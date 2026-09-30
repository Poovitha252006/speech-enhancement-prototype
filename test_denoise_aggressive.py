"""
Verification of Aggressive Background Noise Removal Engine
Tests stationary, non-stationary, and impulsive noise elimination.
"""

import numpy as np
import soundfile as sf
import noisereduce as nr
from core.synthetic_bench import SyntheticAudioBenchmark
from core.metrics import AudioMetricsEvaluator
from core.impulsive_filter import ImpulsiveNoiseFilter

sr = 16000
duration = 3.5

# Create severe 0 dB SNR cocktail (Speech + Fan Hum + Babble + Clicks)
bench = SyntheticAudioBenchmark.create_mixed_benchmark(
    target_snr_db=0.0,
    include_stationary=True,
    include_non_stationary=True,
    include_impulsive=True,
    duration=duration,
    sr=sr
)

clean = bench["clean"]
noisy = bench["noisy"]

# Stage 1: Impulsive despeckler
imp_filter = ImpulsiveNoiseFilter()
s1_audio, imp_stats = imp_filter.process(noisy, sr=sr, aggression=0.85)

# Stage 2: Aggressive Spectral Gating (Stationary)
s2_audio = nr.reduce_noise(
    y=s1_audio,
    sr=sr,
    stationary=True,
    prop_decrease=0.95,
    n_std_thresh_stationary=1.5,
    n_fft=1024,
    hop_length=256
)

# Stage 3: Dynamic Non-Stationary Spectral Reduction
s3_audio = nr.reduce_noise(
    y=s2_audio,
    sr=sr,
    stationary=False,
    prop_decrease=0.85,
    time_constant_s=0.5,
    n_fft=1024,
    hop_length=256
)

# Stage 4: Soft Downward Expander (Noise Gating on pauses)
frame_len = int(0.02 * sr)
hop_len = int(0.01 * sr)
num_frames = (len(s3_audio) - frame_len) // hop_len
energies = np.array([np.mean(s3_audio[i*hop_len : i*hop_len+frame_len]**2) for i in range(num_frames)])
speech_thresh = np.percentile(energies, 60) * 0.15

gain_curve = np.ones(len(s3_audio))
for i in range(num_frames):
    e = energies[i]
    if e < speech_thresh:
        # Attenuate pause frames deeply
        att = max(0.01, (e / (speech_thresh + 1e-8)) ** 1.5)
        start = i * hop_len
        end = min(len(s3_audio), start + frame_len)
        gain_curve[start:end] = np.minimum(gain_curve[start:end], att)

# Smooth gain curve
b, a = [1.0/15.0]*15, [1.0]
from scipy import signal
smooth_gain = signal.filtfilt(b, a, gain_curve)
final_audio = s3_audio * np.clip(smooth_gain, 0.02, 1.0)

# Normalize
peak = np.max(np.abs(final_audio))
if peak > 0:
    final_audio = final_audio * (0.90 / peak)

metrics = AudioMetricsEvaluator.compute_referenced_metrics(clean, noisy, final_audio, sr=sr)
print("=" * 60)
print("AGGRESSIVE NOISE REDUCTION RESULTS:")
print("=" * 60)
for k, v in metrics.items():
    print(f"  {k:30s}: {v}")
print("=" * 60)
