"""
Verification of Enhanced Multi-Stage Pipeline on Real Audio and Synthetic Benchmarks.
"""

import numpy as np
import scipy.signal as signal
from scipy import interpolate
from scipy.ndimage import median_filter
import noisereduce as nr
import soundfile as sf

def process_audio_pipeline(audio: np.ndarray, sr: int = 16000, 
                           enable_impulsive=True,
                           enable_stationary=True,
                           enable_non_stationary=True,
                           impulsive_aggression=0.85,
                           stationary_reduction_db=36.0,
                           non_stationary_aggression=0.85,
                           enable_harmonic_boost=True):
    audio = np.asarray(audio, dtype=np.float32)
    current = np.copy(audio)
    N = len(current)
    scale = max(1, int(sr / 16000))

    # =========================================================================
    # STAGE 1: Dual-Domain Impulsive & Resonant Clank Filter
    # =========================================================================
    if enable_impulsive and len(current) > 0:
        # Part A: Adaptive Hampel Filter for micro-clicks/pops
        win = max(7, int(15 * scale))
        k_factor = max(2.0, 3.5 * (1.6 - 0.8 * np.clip(impulsive_aggression, 0.0, 1.0)))
        pad_width = win
        padded = np.pad(current, pad_width, mode='reflect')
        med_filtered = signal.medfilt(padded, kernel_size=2 * win + 1)[pad_width:-pad_width]
        abs_diff = np.abs(current - med_filtered)
        padded_diff = np.pad(abs_diff, pad_width, mode='reflect')
        mad = signal.medfilt(padded_diff, kernel_size=2 * win + 1)[pad_width:-pad_width]
        noise_floor = np.maximum(1.4826 * mad, 1e-6)
        diff_sig = np.abs(np.diff(current, prepend=current[0]))
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
                if gap_len <= 32 * scale:
                    left_anchor_start = max(0, start_idx - 4 * scale)
                    left_anchor_end = start_idx
                    right_anchor_start = end_idx + 1
                    right_anchor_end = min(N, end_idx + 1 + 4 * scale)
                    x_anchors = np.concatenate([
                        np.arange(left_anchor_start, left_anchor_end),
                        np.arange(right_anchor_start, right_anchor_end)
                    ])
                    y_anchors = current[x_anchors]
                    if len(x_anchors) >= 4:
                        try:
                            cs = interpolate.CubicSpline(x_anchors, y_anchors, bc_type='natural')
                            current[start_idx:end_idx + 1] = np.clip(cs(np.arange(start_idx, end_idx + 1)), -1.0, 1.0)
                        except Exception:
                            current[start_idx:end_idx + 1] = med_filtered[start_idx:end_idx + 1]
                    else:
                        current[start_idx:end_idx + 1] = med_filtered[start_idx:end_idx + 1]

        # Part B: Resonant Metallic Transient Suppressor (High-pass Hilbert Envelope)
        sos_high = signal.butter(4, 1400, btype='highpass', fs=sr, output='sos')
        high_band = signal.sosfilt(sos_high, current)
        env = np.abs(signal.hilbert(high_band))
        smooth_k = np.ones(max(3, int(0.004 * sr))) / max(3, int(0.004 * sr))
        env_smooth = np.convolve(env, smooth_k, mode='same')
        
        env_floor = np.percentile(env_smooth, 75)
        peaks, _ = signal.find_peaks(
            env_smooth, 
            height=max(0.03, env_floor * 2.2), 
            distance=int(0.10 * sr), 
            prominence=0.02
        )

        min_lag = int(sr / 450)
        max_lag = int(sr / 75)
        frame_len_pitch = int(0.03 * sr)

        for p in peaks:
            p_before = max(0, p - frame_len_pitch)
            f_b = current[p_before : p_before + frame_len_pitch]
            c_b = signal.correlate(f_b, f_b, mode='full')[len(f_b)-1:]
            r_b = 0.0
            if c_b[0] > 1e-6:
                reg_b = (c_b / c_b[0])[min_lag:max_lag]
                if len(reg_b) > 0:
                    r_b = np.max(reg_b)

            p_after = min(N - frame_len_pitch, p + int(0.04 * sr))
            f_a = current[p_after : p_after + frame_len_pitch]
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
                current[idx1:idx2] *= 0.02
            else:
                seg = current[idx1:idx2]
                sos_lp = signal.butter(4, 1300, btype='lowpass', fs=sr, output='sos')
                seg_lp = signal.sosfiltfilt(sos_lp, seg)
                cf_win = signal.windows.hann(seg_len)
                current[idx1:idx2] = seg * (1.0 - cf_win) + seg_lp * cf_win

        # Part C: Harmonic-Percussive Source Separation (HPSS)
        n_fft_hpss = 1024
        hop_hpss = 256
        f_hpss, t_hpss, Zxx_hpss = signal.stft(current, fs=sr, nperseg=n_fft_hpss, noverlap=n_fft_hpss - hop_hpss)
        mag_hpss = np.abs(Zxx_hpss)
        phase_hpss = np.angle(Zxx_hpss)

        H = median_filter(mag_hpss, size=(1, 15))
        P = median_filter(mag_hpss, size=(19, 1))
        margin = 1.0 + 0.4 * float(impulsive_aggression)
        mask_H = (H ** 2) / (H ** 2 + (margin * P) ** 2 + 1e-10)
        mag_clean_hpss = mag_hpss * mask_H

        _, hpss_audio = signal.istft(mag_clean_hpss * np.exp(1j * phase_hpss), fs=sr, nperseg=n_fft_hpss, noverlap=n_fft_hpss - hop_hpss)
        current = hpss_audio[:N].astype(np.float32)

    # =========================================================================
    # STAGE 2: Stationary Noise Filter (Multi-Band Quantile Spectral Gating)
    # =========================================================================
    if enable_stationary and len(current) > 0:
        # Use noisereduce spectral gating with high suppression depth
        prop_dec = min(1.0, 0.85 + 0.15 * (stationary_reduction_db / 40.0))
        current = nr.reduce_noise(
            y=current,
            sr=sr,
            stationary=True,
            prop_decrease=prop_dec,
            n_std_thresh_stationary=1.5,
            n_fft=1024,
            hop_length=256
        )

    # =========================================================================
    # STAGE 3: Speech-Aware Dynamic VAD & Pause Expander
    # =========================================================================
    if enable_non_stationary and len(current) > 0:
        frame_len = int(0.03 * sr)
        hop_len = int(0.01 * sr)
        n_frames = max(1, (len(current) - frame_len) // hop_len)
        times = np.arange(n_frames) * (hop_len / sr)

        energies = np.array([np.sum((current[i*hop_len : i*hop_len + frame_len])**2) for i in range(n_frames)])
        
        # Candidate speech frames: energy exceeds adaptive threshold
        log_e = np.log10(np.maximum(energies, 1e-8))
        e_floor = np.percentile(energies, 25)
        speech_cand_thresh = max(0.008, e_floor * 5.0)
        is_cand = energies > speech_cand_thresh

        # Bridge intra-word gaps (<= 140ms / 14 frames)
        bridged = np.copy(is_cand)
        gap = 0
        for i in range(n_frames):
            if bridged[i]:
                if 0 < gap <= 14:
                    bridged[i-gap:i] = True
                gap = 0
            else:
                gap += 1

        # Reject isolated noise blips (< 120ms / 12 frames)
        final_vad = np.zeros(n_frames, dtype=bool)
        curr_start = None
        for i in range(n_frames):
            if bridged[i] and curr_start is None:
                curr_start = i
            elif not bridged[i] and curr_start is not None:
                seg_len = i - curr_start
                if seg_len >= 12:
                    final_vad[curr_start:i] = True
                curr_start = None
        if curr_start is not None and (n_frames - curr_start) >= 12:
            final_vad[curr_start:] = True

        # Construct smooth gain curve
        pause_gain = 10.0 ** (-float(36.0 * non_stationary_aggression) / 20.0) # Down to -36dB to -50dB in pauses
        pause_gain = min(pause_gain, 0.001)

        gain_curve = np.full(len(current), pause_gain, dtype=np.float32)
        for i in range(n_frames):
            if final_vad[i]:
                s_idx = i * hop_len
                e_idx = min(len(current), s_idx + frame_len)
                gain_curve[s_idx:e_idx] = 1.0

        # Smooth gain curve with 60ms Hann window
        k_smooth = signal.windows.hann(int(0.06 * sr))
        k_smooth /= np.sum(k_smooth)
        smooth_gain = np.convolve(gain_curve, k_smooth, mode='same')
        smooth_gain = np.clip(smooth_gain, pause_gain, 1.0)

        current = current * smooth_gain

    # =========================================================================
    # STAGE 4: Vocal Formant EQ & Broadcast Peak Normalization
    # =========================================================================
    # High-pass filter at 85 Hz to eliminate sub-bass rumble
    sos_hp = signal.butter(4, 85, btype='highpass', fs=sr, output='sos')
    current = signal.sosfilt(sos_hp, current)

    # Normalize to -0.92 dBFS (0.92 peak)
    peak = np.max(np.abs(current))
    if peak > 0.01:
        current = current * (0.92 / peak)

    return current.astype(np.float32)

if __name__ == '__main__':
    raw, sr = sf.read('raw_input.wav')
    clean = process_audio_pipeline(raw, sr=sr)
    sf.write('final_clean_speech.wav', clean, sr)
    print('Generated final_clean_speech.wav successfully!')
    
    # Analyze final clean speech
    peak = np.max(np.abs(clean))
    rms = np.sqrt(np.mean(clean**2))
    pause_rms = np.sqrt(np.mean(clean[:int(2.0*sr)]**2))
    speech_rms = np.sqrt(np.mean(clean[int(3.0*sr):int(8.5*sr)]**2))
    print(f'Peak: {peak:.4f}, Overall RMS: {rms:.4f}')
    print(f'Pause RMS (0-2s): {pause_rms:.7f} (-{20*np.log10(1/pause_rms):.1f} dBFS)')
    print(f'Speech RMS (3-8.5s): {speech_rms:.4f} (-{20*np.log10(1/speech_rms):.1f} dBFS)')
    print(f'Signal-to-Pause Contrast: {20*np.log10(speech_rms / pause_rms):.1f} dB')
