"""
Web Server and REST API for Speech Enhancement Prototype
Built using Python's native http.server - zero external framework dependencies.
Provides REST API for:
- Uploading and Denoising custom audio files (WAV, MP3, OGG, WebM)
- Direct microphone recording denoising
- Synthetic benchmark generation and evaluation
- Spectrogram and waveform analysis payload generation
- Serving modern interactive frontend UI
"""

import os
import sys
import io
import json
import base64
import traceback
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import soundfile as sf
from scipy import signal

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

from core.hybrid_pipeline import HybridSpeechEnhancer
from core.metrics import AudioMetricsEvaluator
from core.synthetic_bench import SyntheticAudioBenchmark

PORT = int(os.environ.get("PORT", 8501))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_outputs")

enhancer = HybridSpeechEnhancer(sr=16000)

class NumpyJSONEncoder(json.JSONEncoder):
    """Encodes NumPy types to standard Python native types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def audio_to_base64_wav(audio: np.ndarray, sr: int = 16000) -> str:
    """Encode float32 audio to base64 WAV data URI."""
    buf = io.BytesIO()
    # Ensure float32 in [-1, 1]
    safe_audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(buf, safe_audio, sr, format='WAV', subtype='PCM_16')
    b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:audio/wav;base64,{b64_str}"

def compute_spectrogram_data(audio: np.ndarray, sr: int = 16000, n_fft: int = 512, hop: int = 256):
    """Generate downsampled 2D spectrogram magnitude matrix for web canvas visualization."""
    if len(audio) == 0:
        return []
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-6)
    # Normalize between 0 and 255 for compact JSON transmission
    min_db = -60.0
    max_db = 0.0
    norm_mag = np.clip((mag_db - min_db) / (max_db - min_db), 0.0, 1.0) * 255.0
    
    step_f = max(1, norm_mag.shape[0] // 64)
    step_t = max(1, norm_mag.shape[1] // 128)
    downsampled = norm_mag[::step_f, ::step_t].astype(int).tolist()
    return downsampled

def compute_waveform_peaks(audio: np.ndarray, num_points: int = 256) -> list:
    """Downsample audio to peak pairs for fast waveform rendering."""
    if len(audio) == 0:
        return []
    step = max(1, len(audio) // num_points)
    peaks = []
    for i in range(0, len(audio), step):
        chunk = audio[i:i + step]
        if len(chunk) > 0:
            peaks.append(round(float(np.max(np.abs(chunk))), 4))
    return peaks

class PrototypeAPIHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            local_path = os.path.join(STATIC_DIR, rel_path)
            ext = os.path.splitext(local_path)[1].lower()
            mime = "text/css" if ext == ".css" else "application/javascript" if ext == ".js" else "text/plain"
            self.serve_file(local_path, mime)
        elif path.startswith("/demo_outputs/"):
            rel_path = path[len("/demo_outputs/"):]
            local_path = os.path.join(DEMO_DIR, rel_path)
            self.serve_file(local_path, "audio/wav")
        elif path == "/api/status":
            self.respond_json({
                "status": "online",
                "prototype": "ECE Multi-Stage Speech Enhancement System",
                "core_version": "1.0.0",
                "supported_noise_types": ["stationary", "non_stationary", "impulsive"]
            })
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        if path == "/api/benchmark":
            try:
                data = json.loads(body.decode('utf-8'))
                target_snr = float(data.get("target_snr", 0.0))
                stat = bool(data.get("include_stationary", True))
                non_stat = bool(data.get("include_non_stationary", True))
                imp = bool(data.get("include_impulsive", True))
                duration = float(data.get("duration", 3.0))

                sr = 16000
                bench = SyntheticAudioBenchmark.create_mixed_benchmark(
                    target_snr_db=target_snr,
                    include_stationary=stat,
                    include_non_stationary=non_stat,
                    include_impulsive=imp,
                    duration=duration,
                    sr=sr
                )

                # Process
                cfg = {
                    "enable_impulsive": imp or bool(data.get("force_impulsive", True)),
                    "enable_stationary": stat or bool(data.get("force_stationary", True)),
                    "enable_non_stationary": non_stat or bool(data.get("force_non_stationary", True)),
                    "impulsive_aggression": float(data.get("impulsive_aggression", 0.85)),
                    "stationary_reduction_db": float(data.get("stationary_reduction_db", 32.0)),
                    "non_stationary_aggression": float(data.get("non_stationary_aggression", 0.85)),
                    "enable_harmonic_boost": bool(data.get("enable_harmonic_boost", True)),
                    "aggressive_mode": bool(data.get("aggressive_mode", False))
                }

                enhanced, report = enhancer.process(bench["noisy"], sr=sr, **cfg)
                metrics = AudioMetricsEvaluator.compute_referenced_metrics(
                    bench["clean"], bench["noisy"], enhanced, sr=sr
                )

                response_payload = {
                    "success": True,
                    "target_snr": target_snr,
                    "components": bench["components"],
                    "metrics": metrics,
                    "report": report,
                    "clean_audio_b64": audio_to_base64_wav(bench["clean"], sr),
                    "noisy_audio_b64": audio_to_base64_wav(bench["noisy"], sr),
                    "enhanced_audio_b64": audio_to_base64_wav(enhanced, sr),
                    "noise_removed_b64": audio_to_base64_wav(bench["noisy"] - enhanced, sr),
                    "noisy_waveform": compute_waveform_peaks(bench["noisy"]),
                    "enhanced_waveform": compute_waveform_peaks(enhanced),
                    "noisy_spectrogram": compute_spectrogram_data(bench["noisy"], sr),
                    "enhanced_spectrogram": compute_spectrogram_data(enhanced, sr)
                }
                self.respond_json(response_payload)
            except Exception as e:
                traceback.print_exc()
                self.respond_json({"success": False, "error": str(e)}, status=500)

        elif path == "/api/denoise":
            try:
                data = json.loads(body.decode('utf-8'))
                audio_b64 = data.get("audio_data")
                if not audio_b64:
                    self.respond_json({"success": False, "error": "No audio_data provided"}, status=400)
                    return

                # Strip data URI header if present
                if "," in audio_b64:
                    audio_b64 = audio_b64.split(",", 1)[1]

                audio_bytes = base64.b64decode(audio_b64)
                audio_io = io.BytesIO(audio_bytes)

                try:
                    audio_arr, sr = sf.read(audio_io, dtype='float32')
                except Exception as read_err:
                    print(f"[!] libsndfile error reading audio: {read_err}. Checking raw PCM...")
                    audio_io.seek(0)
                    # Try raw int16 PCM fallback
                    pcm_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(pcm_data) > 0:
                        audio_arr = pcm_data
                        sr = 16000
                    else:
                        raise ValueError(f"Could not decode audio file: {read_err}")

                # Convert to mono if multi-channel
                if audio_arr.ndim > 1:
                    audio_arr = np.mean(audio_arr, axis=1)

                # Resample to 16kHz if needed
                if sr != 16000:
                    num_target_samples = int(len(audio_arr) * 16000 / sr)
                    audio_arr = signal.resample(audio_arr, num_target_samples)
                    sr = 16000

                cfg = {
                    "enable_impulsive": bool(data.get("enable_impulsive", True)),
                    "enable_stationary": bool(data.get("enable_stationary", True)),
                    "enable_non_stationary": bool(data.get("enable_non_stationary", True)),
                    "impulsive_aggression": float(data.get("impulsive_aggression", 0.85)),
                    "stationary_reduction_db": float(data.get("stationary_reduction_db", 36.0)),
                    "non_stationary_aggression": float(data.get("non_stationary_aggression", 0.85)),
                    "enable_harmonic_boost": bool(data.get("enable_harmonic_boost", True)),
                    "aggressive_mode": bool(data.get("aggressive_mode", False))
                }

                enhanced, report = enhancer.process(audio_arr, sr=sr, **cfg)
                metrics = AudioMetricsEvaluator.compute_blind_metrics(audio_arr, enhanced, sr=sr)

                response_payload = {
                    "success": True,
                    "sr": sr,
                    "duration": float(len(audio_arr) / sr),
                    "metrics": metrics,
                    "report": report,
                    "noisy_audio_b64": audio_to_base64_wav(audio_arr, sr),
                    "enhanced_audio_b64": audio_to_base64_wav(enhanced, sr),
                    "noise_removed_b64": audio_to_base64_wav(audio_arr - enhanced, sr),
                    "noisy_waveform": compute_waveform_peaks(audio_arr),
                    "enhanced_waveform": compute_waveform_peaks(enhanced),
                    "noisy_spectrogram": compute_spectrogram_data(audio_arr, sr),
                    "enhanced_spectrogram": compute_spectrogram_data(enhanced, sr)
                }
                self.respond_json(response_payload)
            except Exception as e:
                traceback.print_exc()
                self.respond_json({"success": False, "error": str(e)}, status=500)
        else:
            self.send_error(404, "Unknown API route")

    def serve_file(self, file_path, content_type):
        if not os.path.exists(file_path):
            self.send_error(404, f"File {file_path} not found")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def respond_json(self, data, status=200):
        try:
            body = json.dumps(data, cls=NumpyJSONEncoder).encode('utf-8')
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            traceback.print_exc()
            err_body = json.dumps({"success": False, "error": str(e)}).encode('utf-8')
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)

def start_server(port=PORT):
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, PrototypeAPIHandler)
    print(f"\n[+] Speech Enhancement Web Server running at: http://0.0.0.0:{port}")
    print(f"[+] Web Dashboard URL: http://localhost:{port}/index.html\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else PORT))
    start_server(port)
