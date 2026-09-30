import numpy as np
import scipy.signal as signal
from scipy import interpolate
from scipy.ndimage import median_filter
import soundfile as sf

class UpgradedImpulsiveFilter:
    def __init__(self, window_size=15, threshold_factor=3.5, max_gap=32):
        self.window_size = int(window_size)
        self.threshold_factor = float(threshold_factor)
        self.max_gap = int(max_gap)

    def process(self, audio: np.ndarray, sr: int = 16000, aggression: float = 0.85):
        if audio.ndim > 1:
            out = []
            st_list = []
            for ch in range(audio.shape[1]):
                c, s = self.process(audio[:, ch], sr=sr, aggression=aggression)
                out.append(c)
                st_list.append(s)
            return np.column_stack(out), st_list[0]

        audio = np.asarray(audio, dtype=np.float32)
        cleaned = np.copy(audio)
        scale = max(1, int(sr / 16000))
        win = max(7, int(self.window_size * scale))
        k_factor = max(2.0, self.threshold_factor * (1.6 - 0.8 * np.clip(aggression, 0.0, 1.0)))
        N = len(audio)

        # 1. Hampel point-outlier filter (Micro-transients)
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
        impulses_count = 0
        interpolated_samples_count = 0

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
                    impulses_count += 1
                    interpolated_samples_count += gap_len

        # 2. Resonant Metallic Transient Suppressor (High-pass Hilbert Envelope)
        sos_high = signal.butter(4, 1400, btype='highpass', fs=sr, output='sos')
        high_band = signal.sosfilt(sos_high, cleaned)
        env = np.abs(signal.hilbert(high_band))
        smooth_k = np.ones(max(3, int(0.004 * sr))) / max(3, int(0.004 * sr))
        env_smooth = np.convolve(env, smooth_k, mode='same')
        
        # Adaptive threshold based on 75th percentile of envelope
        env_floor = np.percentile(env_smooth, 75)
        peaks, _ = signal.find_peaks(
            env_smooth, 
            height=max(0.03, env_floor * 2.5), 
            distance=int(0.10 * sr), 
            prominence=0.02
        )

        min_lag = int(sr / 450)
        max_lag = int(sr / 75)
        frame_len = int(0.03 * sr)

        for p in peaks:
            p_before = max(0, p - frame_len)
            f_b = cleaned[p_before : p_before + frame_len]
            c_b = signal.correlate(f_b, f_b, mode='full')[len(f_b)-1:]
            r_b = 0.0
            if c_b[0] > 1e-6:
                reg_b = (c_b / c_b[0])[min_lag:max_lag]
                if len(reg_b) > 0:
                    r_b = np.max(reg_b)

            p_after = min(N - frame_len, p + int(0.04 * sr))
            f_a = cleaned[p_after : p_after + frame_len]
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
                impulses_count += 1
                interpolated_samples_count += seg_len
            else:
                seg = cleaned[idx1:idx2]
                sos_lp = signal.butter(4, 1300, btype='lowpass', fs=sr, output='sos')
                seg_lp = signal.sosfiltfilt(sos_lp, seg)
                cf_win = signal.windows.hann(seg_len)
                cleaned[idx1:idx2] = seg * (1.0 - cf_win) + seg_lp * cf_win
                impulses_count += 1
                interpolated_samples_count += seg_len

        # 3. Harmonic-Percussive Source Separation (HPSS)
        n_fft = 1024
        hop = 256
        f, t, Zxx = signal.stft(cleaned, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        H = median_filter(mag, size=(1, 15))
        P = median_filter(mag, size=(19, 1))
        margin = 1.0 + 0.4 * float(aggression)
        mask_H = (H ** 2) / (H ** 2 + (margin * P) ** 2 + 1e-10)
        mag_clean = mag * mask_H

        _, hpss_audio = signal.istft(mag_clean * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        cleaned = hpss_audio[:N].astype(np.float32)

        diff_energy = np.sum((audio - cleaned) ** 2)
        total_energy = np.sum(audio ** 2) + 1e-12
        removed_ratio = diff_energy / total_energy
        removed_db = 10.0 * np.log10(max(removed_ratio, 1e-8))

        stats = {
            "impulses_detected": int(impulses_count),
            "samples_interpolated": int(interpolated_samples_count),
            "impulsive_noise_energy_removed_db": float(removed_db)
        }
        return cleaned, stats

if __name__ == '__main__':
    raw, sr = sf.read('raw_input.wav')
    f_test = UpgradedImpulsiveFilter()
    out, st = f_test.process(raw, sr=sr)
    print('Impulsive filter test completed successfully!')
    print('Stats:', st)
    sf.write('test_stage1_output.wav', out, sr)
