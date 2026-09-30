"""
Stage 1: Impulsive Noise Filter
Addresses clicks, pops, switch bursts, scratches, keyboard taps, door knocks,
and metallic resonant clanks (cutlery, spoon taps, chime ringing).
Combines:
1. Adaptive Hampel Outlier Filter with cubic spline interpolation for micro-transients (< 2ms).
2. High-Frequency Hilbert Envelope & Transient Suppressor for resonant clanks (25 - 50ms).
Preserves zero phase delay so reference speech signals are not distorted.
"""

import numpy as np
from scipy import signal, interpolate

class ImpulsiveNoiseFilter:
    """
    Adaptive Impulsive and Resonant Transient Suppressor.
    Removes clicks, pops, and prolonged metallic clank transients.
    """
    def __init__(self, window_size: int = 15, threshold_factor: float = 3.5, max_gap: int = 32):
        self.window_size = int(window_size)
        self.threshold_factor = float(threshold_factor)
        self.max_gap = int(max_gap)

    def process(self, audio: np.ndarray, sr: int = 16000, aggression: float = 0.85) -> tuple[np.ndarray, dict]:
        """
        Remove impulsive spikes and resonant transients from input audio.

        Args:
            audio: 1D float32 audio array.
            sr: Sampling rate in Hz.
            aggression: 0.0 (gentle) to 1.0 (strict spike suppression).

        Returns:
            cleaned_audio: Filtered audio array.
            stats: Diagnostic performance metrics.
        """
        if audio.ndim > 1:
            out_channels = []
            stats_list = []
            for ch in range(audio.shape[1]):
                cleaned_ch, st = self.process(audio[:, ch], sr=sr, aggression=aggression)
                out_channels.append(cleaned_ch)
                stats_list.append(st)
            stacked = np.column_stack(out_channels)
            combined_stats = {
                "impulses_detected": sum(s["impulses_detected"] for s in stats_list),
                "samples_interpolated": sum(s["samples_interpolated"] for s in stats_list),
                "impulsive_noise_energy_removed_db": float(np.mean([s["impulsive_noise_energy_removed_db"] for s in stats_list]))
            }
            return stacked, combined_stats

        audio = np.asarray(audio, dtype=np.float32)
        N = len(audio)
        if N == 0:
            return audio, {"impulses_detected": 0, "samples_interpolated": 0, "impulsive_noise_energy_removed_db": 0.0}

        cleaned = np.copy(audio)
        scale = max(1, int(sr / 16000))
        win = max(7, int(self.window_size * scale))
        k_factor = max(2.0, self.threshold_factor * (1.6 - 0.8 * np.clip(aggression, 0.0, 1.0)))

        impulses_count = 0
        interpolated_samples_count = 0

        # Step 1: Hampel point-outlier filter (Micro-transients)
        pad_width = win
        padded = np.pad(cleaned, pad_width, mode='reflect')
        med_filtered = signal.medfilt(padded, kernel_size=2 * win + 1)[pad_width:-pad_width]
        abs_diff = np.abs(cleaned - med_filtered)
        padded_diff = np.pad(abs_diff, pad_width, mode='reflect')
        mad = signal.medfilt(padded_diff, kernel_size=2 * win + 1)[pad_width:-pad_width]
        noise_floor = np.maximum(1.4826 * mad, 1e-6)
        diff_sig = np.abs(np.diff(cleaned, prepend=cleaned[0]))
        threshold = k_factor * noise_floor
        outlier_mask = (abs_diff > threshold) & (diff_sig > (0.5 * threshold))

        outlier_indices = np.flatnonzero(outlier_mask)
        if len(outlier_indices) > 0:
            diffs = np.diff(outlier_indices)
            splits = np.flatnonzero(diffs > 2)
            clusters = np.split(outlier_indices, splits + 1)

            for cluster in clusters:
                if len(cluster) == 0:
                    continue
                start_idx = cluster[0]
                end_idx = cluster[-1]
                gap_len = end_idx - start_idx + 1

                if gap_len <= self.max_gap * scale:
                    left_anchor_start = max(0, start_idx - 4 * scale)
                    left_anchor_end = start_idx
                    right_anchor_start = end_idx + 1
                    right_anchor_end = min(N, end_idx + 1 + 4 * scale)

                    x_anchors = np.concatenate([
                        np.arange(left_anchor_start, left_anchor_end),
                        np.arange(right_anchor_start, right_anchor_end)
                    ])
                    y_anchors = cleaned[x_anchors]

                    if len(x_anchors) >= 4:
                        try:
                            cs = interpolate.CubicSpline(x_anchors, y_anchors, bc_type='natural')
                            cleaned[start_idx:end_idx + 1] = np.clip(cs(np.arange(start_idx, end_idx + 1)), -1.0, 1.0)
                        except Exception:
                            cleaned[start_idx:end_idx + 1] = med_filtered[start_idx:end_idx + 1]
                    else:
                        cleaned[start_idx:end_idx + 1] = med_filtered[start_idx:end_idx + 1]
                else:
                    cleaned[start_idx:end_idx + 1] = 0.3 * cleaned[start_idx:end_idx + 1] + 0.7 * med_filtered[start_idx:end_idx + 1]

                impulses_count += 1
                interpolated_samples_count += gap_len

        # Step 2: Resonant Metallic Transient Suppressor (High-pass Hilbert Envelope)
        if N > int(0.3 * sr):
            sos_high = signal.butter(4, 1400, btype='highpass', fs=sr, output='sos')
            high_band = signal.sosfiltfilt(sos_high, cleaned)
            env = np.abs(signal.hilbert(high_band))
            smooth_k = np.ones(max(3, int(0.004 * sr))) / max(3, int(0.004 * sr))
            env_smooth = np.convolve(env, smooth_k, mode='same')

            env_floor = np.percentile(env_smooth, 75)
            peaks, _ = signal.find_peaks(
                env_smooth,
                height=max(0.035, env_floor * 2.3),
                distance=int(0.10 * sr),
                prominence=0.025
            )

            min_lag = int(sr / 450)
            max_lag = int(sr / 75)
            frame_len_pitch = int(0.03 * sr)

            for p in peaks:
                p_before = max(0, p - frame_len_pitch)
                f_b = cleaned[p_before : p_before + frame_len_pitch]
                c_b = signal.correlate(f_b, f_b, mode='full')[len(f_b)-1:]
                r_b = 0.0
                if c_b[0] > 1e-6:
                    reg_b = (c_b / c_b[0])[min_lag:max_lag]
                    if len(reg_b) > 0:
                        r_b = np.max(reg_b)

                p_after = min(N - frame_len_pitch, p + int(0.04 * sr))
                f_a = cleaned[p_after : p_after + frame_len_pitch]
                c_a = signal.correlate(f_a, f_a, mode='full')[len(f_a)-1:]
                r_a = 0.0
                if c_a[0] > 1e-6:
                    reg_a = (c_a / c_a[0])[min_lag:max_lag]
                    if len(reg_a) > 0:
                        r_a = np.max(reg_a)

                is_voiced_speech = (r_b > 0.40 or r_a > 0.40)

                pre_samp = int(0.004 * sr)
                post_samp = int(0.038 * sr)
                idx1 = max(0, p - pre_samp)
                idx2 = min(N, p + post_samp)
                seg_len = idx2 - idx1

                if not is_voiced_speech:
                    cleaned[idx1:idx2] *= 0.02
                else:
                    seg = cleaned[idx1:idx2]
                    sos_lp = signal.butter(4, 1300, btype='lowpass', fs=sr, output='sos')
                    seg_lp = signal.sosfiltfilt(sos_lp, seg)
                    cf_win = signal.windows.hann(seg_len)
                    cleaned[idx1:idx2] = seg * (1.0 - cf_win) + seg_lp * cf_win

                impulses_count += 1
                interpolated_samples_count += seg_len

        diff_energy = np.sum((audio - cleaned) ** 2)
        total_energy = np.sum(audio ** 2) + 1e-12
        removed_ratio = diff_energy / total_energy
        removed_db = 10.0 * np.log10(max(removed_ratio, 1e-8))

        stats = {
            "impulses_detected": int(impulses_count),
            "samples_interpolated": int(interpolated_samples_count),
            "impulsive_noise_energy_removed_db": float(round(removed_db, 2))
        }
        return cleaned.astype(np.float32), stats
