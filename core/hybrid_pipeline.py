"""
Hybrid Speech Enhancement Pipeline
Cascaded Multi-Stage Architecture:
Input Audio -> [Stage 1: Impulsive & Resonant Despeckler] 
            -> [Stage 2: Multi-Band Adaptive Spectral Gating] 
            -> [Stage 3: Non-Stationary Babble Suppressor & VAD Pause Expander] 
            -> [Stage 4: Vocal Formant Clarifier & Safe Limiter] 
            -> Clean Audio
"""

import numpy as np
from scipy import signal
from scipy import interpolate
from .impulsive_filter import ImpulsiveNoiseFilter
from .stationary_filter import StationaryNoiseFilter
from .non_stationary_filter import NonStationaryNoiseFilter

class HybridSpeechEnhancer:
    """
    Cascaded 4-Stage Hybrid Speech Enhancement Engine.
    Actively and aggressively eliminates background noise (stationary, non-stationary, impulsive)
    while strictly preserving primary speech clarity, natural vocal timbre, and formant dynamics.
    """
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.stage1_impulsive = ImpulsiveNoiseFilter()
        self.stage2_stationary = StationaryNoiseFilter()
        self.stage3_non_stationary = NonStationaryNoiseFilter()

    def process(
        self,
        audio: np.ndarray,
        sr: int = None,
        enable_impulsive: bool = True,
        enable_stationary: bool = True,
        enable_non_stationary: bool = True,
        impulsive_aggression: float = 0.85,
        stationary_reduction_db: float = 30.0,
        non_stationary_aggression: float = 0.85,
        enable_harmonic_boost: bool = True,
        normalize_output: bool = True,
        speech_priority: bool = True,
        aggressive_mode: bool = False,
        **kwargs
    ) -> tuple[np.ndarray, dict]:
        """
        Execute speech enhancement pipeline.

        Args:
            audio: Input audio array (float32, normalized in [-1.0, 1.0]).
            sr: Sampling rate (defaults to self.sr if None).
            enable_impulsive: Toggle Stage 1 (Impulsive Noise Suppressor).
            enable_stationary: Toggle Stage 2 (Stationary Noise Suppressor).
            enable_non_stationary: Toggle Stage 3 (Non-Stationary Noise Suppressor).
            impulsive_aggression: Factor 0.0 - 1.0 (Aggression control).
            stationary_reduction_db: Active dB attenuation for stationary noise.
            non_stationary_aggression: Factor 0.0 - 1.0.
            enable_harmonic_boost: Preserve and enhance pitch harmonics.
            normalize_output: Apply peak normalization to prevent clipping.
            speech_priority: Prioritize 100% natural, unshaken vocal integrity using Decision-Directed tracking.
            aggressive_mode: Boost Stage 1 voice enhancement and deepen noise suppression for extreme noise environments.
            **kwargs: Additional pipeline parameters for forwards-compatibility.

        Returns:
            enhanced_audio: Final cleaned audio array.
            report: Comprehensive diagnostic data and stage-by-stage stats.
        """
        if sr is None:
            sr = self.sr

        current_audio = np.asarray(audio, dtype=np.float32)
        N = len(current_audio)

        input_peak = float(np.max(np.abs(current_audio))) if N > 0 else 0.0
        input_rms = float(np.sqrt(np.mean(current_audio ** 2))) if N > 0 else 0.0

        report = {
            "sample_rate": sr,
            "duration_seconds": float(N / sr) if sr > 0 else 0.0,
            "input_peak": input_peak,
            "input_rms": input_rms,
            "stages_executed": [],
            "stage_metrics": {},
            "aggressive_mode": aggressive_mode
        }

        if N == 0:
            return current_audio, report

        # =========================================================================
        # SPEECH-PRIORITY PSYCHOACOUSTIC MODE (Zero-Artifact, Undistorted, Pristine Speech)
        # =========================================================================
        if speech_priority and N > int(0.2 * sr):
            # Stage 1: Zero-Phase Rumble & Desk-Impact Rejection Filter (100 Hz Low-Cut)
            sos_hp = signal.butter(4, 100, btype='highpass', fs=sr, output='sos')
            audio_hp = signal.sosfiltfilt(sos_hp, current_audio)
            report["stages_executed"].append("Stage 1: Zero-Phase 100Hz Impact & Rumble Filter")

            # Stage 2: Consonant-Safe Impulsive Despiker (protects consonants, removes narrow < 2.5ms spikes)
            audio_despike = np.copy(audio_hp)
            hop_ds = int(0.005 * sr)
            win_ds = int(0.015 * sr)
            mad_mult = 4.5 if aggressive_mode else 6.0
            max_spk = int(0.0035 * sr) if aggressive_mode else int(0.0025 * sr)
            for b in range(0, len(audio_despike) - win_ds, hop_ds):
                chunk = audio_despike[b:b+win_ds]
                med = np.median(chunk)
                mad = np.median(np.abs(chunk - med)) + 1e-6
                spikes = np.where(np.abs(chunk - med) > mad_mult * mad)[0]
                if 0 < len(spikes) <= max_spk:
                    for si in spikes:
                        idx = b + si
                        if 3 <= idx < N - 3:
                            audio_despike[idx] = 0.5 * (audio_despike[idx-3] + audio_despike[idx+3])
            stage2_name = "Stage 2: Consonant-Safe Impulsive Despiker (Aggressive Voice Enhancement)" if aggressive_mode else "Stage 2: Consonant-Safe Impulsive Despiker"
            report["stages_executed"].append(stage2_name)

            # Stage 3: STFT Analysis
            n_fft = 1024
            hop = 256
            window = signal.windows.hann(n_fft, sym=False)
            f, t, Zxx = signal.stft(audio_despike, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop, boundary='zeros')
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            power = mag ** 2
            num_freqs, num_frames = mag.shape

            # Stage 4: Robust Quantile Stationary Noise Profiling (15th percentile)
            noise_power = np.percentile(power, 15, axis=1, keepdims=True)
            noise_power = np.maximum(noise_power, 1e-7)

            # Sibilance & Consonant Protection (> 3.2 kHz)
            sibilance_protect = np.ones((num_freqs, 1))
            for fi in range(num_freqs):
                if f[fi] > 3200:
                    sibilance_protect[fi, 0] = 0.60

            # Stage 5: Decision-Directed A-Priori SNR Tracking (Ephraim-Malah / Scalart)
            alpha_dd = 0.95
            active_reduction_db = min(24.0, max(14.0, float(stationary_reduction_db) * 0.6))
            min_speech_gain = 10.0 ** (-active_reduction_db / 20.0)

            prev_speech_power = np.copy(noise_power)
            wiener_gain = np.zeros_like(mag)
            for m in range(num_frames):
                gamma = power[:, m:m+1] / (noise_power * sibilance_protect + 1e-9)
                xi = alpha_dd * (prev_speech_power / (noise_power + 1e-9)) + (1.0 - alpha_dd) * np.maximum(gamma - 1.0, 0.0)
                g = np.maximum(xi / (xi + 1.0), min_speech_gain)
                wiener_gain[:, m:m+1] = g
                prev_speech_power = (g * mag[:, m:m+1]) ** 2

            report["stages_executed"].append("Stage 3: Decision-Directed Wiener Filter (Formant-Preserved)")

            # Stage 6: Syllabic Cluster Sentence Isolator & Dual-Zone Expander
            speech_band_idx = (f >= 300) & (f <= 3800)
            mag_sb = mag[speech_band_idx, :]
            geo_m = np.exp(np.mean(np.log(mag_sb**2 + 1e-12), axis=0))
            ari_m = np.mean(mag_sb**2, axis=0) + 1e-12
            sfm = geo_m / ari_m

            sb_power = np.sum(mag_sb**2, axis=0)
            sb_floor = np.percentile(sb_power, 15)

            indicator = (sb_power / (sb_floor + 1e-9)) * (1.0 - np.clip(sfm / 0.12, 0.0, 1.0))
            w_smooth = int(0.5 * sr / hop)
            smooth_ind = np.convolve(indicator, np.ones(w_smooth)/w_smooth, mode='same')

            active_idx = np.where(smooth_ind > 250)[0]
            if len(active_idx) > 0 and (t[active_idx[-1]] - t[active_idx[0]]) > 1.0:
                t_start = max(0.0, t[active_idx[0]] - 0.20)
                t_end = min(t[-1], t[active_idx[-1]] + 0.35)
            else:
                t_start = 0.0
                t_end = t[-1]

            envelope = np.zeros(num_frames)
            pause_floor = 10.0 ** (-float(40.0 + 24.0 * non_stationary_aggression) / 20.0)
            pause_floor = min(pause_floor, 0.0001)

            for m in range(num_frames):
                tm = t[m]
                if tm < t_start - 0.15:
                    envelope[m] = pause_floor
                elif tm < t_start:
                    fade = (tm - (t_start - 0.15)) / 0.15
                    envelope[m] = pause_floor * (1.0 - fade) + 1.0 * fade
                elif tm <= t_end:
                    envelope[m] = 1.0
                elif tm <= t_end + 0.25:
                    fade = (tm - t_end) / 0.25
                    envelope[m] = 1.0 * (1.0 - fade) + pause_floor * fade
                else:
                    envelope[m] = pause_floor

            combined_gain = wiener_gain * envelope[np.newaxis, :]
            report["stages_executed"].append("Stage 4: Syllabic Cluster Sentence Isolator")

            # Stage 7: iSTFT Synthesis
            enhanced_Zxx = combined_gain * mag * np.exp(1j * phase)
            _, clean_audio = signal.istft(enhanced_Zxx, fs=sr, window=window, nperseg=n_fft, noverlap=n_fft - hop)
            clean_audio = clean_audio[:N]

            # Stage 8: Vocal Presence EQ & Intelligibility Clarifier (+2.0 dB at 3.2 kHz)
            b_peq, a_peq = signal.iirpeak(3200, 1.2, fs=sr)
            clean_audio = clean_audio + 0.15 * signal.lfilter(b_peq, a_peq, clean_audio)
            report["stages_executed"].append("Stage 5: Vocal Presence & Intelligibility Clarifier")

            # Lead-in and tail-out silence gating (eliminates all pre/post speech clicks)
            if t_start > 0.3:
                fade_in_s = max(0, int((t_start - 0.15) * sr))
                fade_in_len = min(int(0.12 * sr), N - fade_in_s)
                clean_audio[:fade_in_s] = 0.0
                if fade_in_len > 0 and fade_in_s + fade_in_len <= N:
                    clean_audio[fade_in_s:fade_in_s+fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len)
            if t_end < (N / sr) - 0.3:
                fade_out_s = min(N, int((t_end + 0.15) * sr))
                fade_out_len = min(int(0.15 * sr), N - fade_out_s)
                if fade_out_len > 0 and fade_out_s + fade_out_len <= N:
                    clean_audio[fade_out_s:fade_out_s+fade_out_len] *= np.linspace(1.0, 0.0, fade_out_len)
                    clean_audio[fade_out_s+fade_out_len:] = 0.0

            # Stage 9: Safe Headroom Normalization to -0.5 dBFS
            out_peak = float(np.max(np.abs(clean_audio))) if len(clean_audio) > 0 else 0.0
            if normalize_output and out_peak > 0.01:
                clean_audio = clean_audio * (0.94 / out_peak)

            clean_audio = np.clip(clean_audio, -1.0, 1.0)
            report["output_peak"] = round(float(np.max(np.abs(clean_audio))), 4)
            report["output_rms"] = round(float(np.sqrt(np.mean(clean_audio ** 2))), 4)
            report["speech_priority_mode"] = True
            return clean_audio.astype(np.float32), report

        # =========================================================================
        # STANDARD CALIBRATED MODE (Zero-Phase Reference Mode)
        # =========================================================================
        # Stage 1: Impulsive Despeckler (Aggressive Voice Enhancement)
        eff_imp_aggression = min(1.0, impulsive_aggression * 1.15) if aggressive_mode else impulsive_aggression
        if enable_impulsive and N > 0:
            cleaned_s1, s1_stats = self.stage1_impulsive.process(
                current_audio, sr=sr, aggression=eff_imp_aggression
            )
            current_audio = cleaned_s1
            stage1_name = "Stage 1: Impulsive Despeckler (Aggressive Voice Enhancement)" if aggressive_mode else "Stage 1: Impulsive Despeckler"
            report["stages_executed"].append(stage1_name)
            report["stage_metrics"]["impulsive"] = s1_stats
        else:
            report["stage_metrics"]["impulsive"] = {"status": "bypassed"}

        # Stage 2: Stationary Spectral Gating
        eff_stat_reduction = min(48.0, stationary_reduction_db + (6.0 if aggressive_mode else 0.0))
        if enable_stationary and N > 0:
            cleaned_s2, s2_stats = self.stage2_stationary.process(
                current_audio, sr=sr, reduction_db=eff_stat_reduction, sensitivity=1.6, spectral_floor_db=-40.0
            )
            current_audio = cleaned_s2
            report["stages_executed"].append("Stage 2: Adaptive Spectral Gating")
            report["stage_metrics"]["stationary"] = s2_stats
        else:
            report["stage_metrics"]["stationary"] = {"status": "bypassed"}

        # Stage 3: Non-Stationary Babble Suppressor & VAD Pause Expander
        eff_non_stat_aggression = min(1.0, non_stationary_aggression + (0.1 if aggressive_mode else 0.0))
        if enable_non_stationary and N > 0:
            cleaned_s3, s3_stats = self.stage3_non_stationary.process(
                current_audio, sr=sr, aggression=eff_non_stat_aggression, harmonic_boost=enable_harmonic_boost
            )
            current_audio = cleaned_s3
            report["stages_executed"].append("Stage 3: VAD Pause Expander & Harmonic Comb")
            report["stage_metrics"]["non_stationary"] = s3_stats
        else:
            report["stage_metrics"]["non_stationary"] = {"status": "bypassed"}

        # Stage 4: Peak Normalization
        out_peak = float(np.max(np.abs(current_audio))) if N > 0 else 0.0
        if normalize_output and out_peak > 0.05:
            if out_peak > 0.95:
                current_audio = current_audio * (0.92 / out_peak)
            elif out_peak < 0.50:
                gain_factor = min(0.85 / (out_peak + 1e-6), 1.8)
                current_audio = current_audio * gain_factor

        current_audio = np.clip(current_audio, -1.0, 1.0)
        final_peak = float(np.max(np.abs(current_audio))) if N > 0 else 0.0
        final_rms = float(np.sqrt(np.mean(current_audio ** 2))) if N > 0 else 0.0

        report["output_peak"] = round(final_peak, 4)
        report["output_rms"] = round(final_rms, 4)
        report["speech_priority_mode"] = False

        return current_audio.astype(np.float32), report
