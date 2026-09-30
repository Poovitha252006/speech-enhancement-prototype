"""
Stage 3: Non-Stationary Noise & Dynamic Babble Filter with Active VAD Pause Expander
Actively eliminates dynamic, time-variant noise:
- Multi-talker crowd speech babble, cafe ambience, passing sirens, footsteps
- Ambient room bleed during speech pauses
Combines:
1. Multi-Feature Voice Activity Detector (VAD) using Pitch Correlation & Adaptive Energy Profiling
2. Active Downward Pause Expander (up to 55 dB attenuation during non-speech intervals)
3. Pitch-Synchronous Harmonic Comb Tracking to preserve target vocal formants
"""

import numpy as np
from scipy import signal

class NonStationaryNoiseFilter:
    """
    Subband Psychoacoustic & Harmonic Comb Filter with Deep Downward Pause Expander.
    Actively separates target human voice from background babble and silences ambient bleed.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.window = signal.windows.hann(self.n_fft, sym=False)

    def _estimate_pitch(self, frame: np.ndarray, sr: int, min_energy: float = 0.005) -> float:
        """
        Estimate fundamental pitch frequency F0 (in Hz) using normalized autocorrelation.
        Search restricted to human vocal range: 80 Hz to 450 Hz.
        Bypasses pitch search on low-energy silence to prevent false positive triggers.
        """
        n = len(frame)
        e = float(np.sum(frame ** 2))
        if e < min_energy:
            return 0.0

        min_lag = int(sr / 450)
        max_lag = int(sr / 80)
        
        f_frame = np.fft.rfft(frame * signal.windows.hamming(n), n=2*n)
        autocorr = np.fft.irfft(np.abs(f_frame)**2)[:n]
        
        if autocorr[0] <= 1e-8:
            return 0.0

        norm_autocorr = autocorr / autocorr[0]
        search_region = norm_autocorr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.0

        peak_idx = np.argmax(search_region)
        peak_val = search_region[peak_idx]
        
        if peak_val > 0.38:
            lag = min_lag + peak_idx
            return float(sr / lag)
        return 0.0

    def process(
        self, 
        audio: np.ndarray, 
        sr: int = 16000, 
        aggression: float = 0.85, 
        harmonic_boost: bool = True
    ) -> tuple[np.ndarray, dict]:
        """
        Actively suppress non-stationary noise, speech babble, and background pause bleed.

        Args:
            audio: 1D audio array.
            sr: Sample rate.
            aggression: 0.0 (subtle) to 1.0 (aggressive studio silence in pauses).
            harmonic_boost: Preserve and enhance pitch harmonics of target speaker.

        Returns:
            enhanced_audio: Non-stationary cleaned audio array.
            stats: Diagnostic performance metrics.
        """
        if audio.ndim > 1:
            out_channels = []
            stats_list = []
            for ch in range(audio.shape[1]):
                clean_ch, st = self.process(
                    audio[:, ch], 
                    sr=sr, 
                    aggression=aggression, 
                    harmonic_boost=harmonic_boost
                )
                out_channels.append(clean_ch)
                stats_list.append(st)
            stacked = np.column_stack(out_channels)
            return stacked, stats_list[0]

        audio = np.asarray(audio, dtype=np.float32)
        N = len(audio)
        if N < self.n_fft:
            return audio, {"voiced_speech_frames": 0, "active_speech_percentage": 0.0, "pause_noise_expansion_db": 35.0}

        # STFT
        f, t, Zxx = signal.stft(
            audio, 
            fs=sr, 
            window=self.window, 
            nperseg=self.n_fft, 
            noverlap=self.n_fft - self.hop_length, 
            boundary='zeros'
        )

        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        num_freqs, num_frames = mag.shape

        # Step 1: Multi-Feature Speech Activity & Frame Energy Analysis
        frame_len = self.n_fft
        hop_len = self.hop_length

        energies = np.sum(mag ** 2, axis=0)
        e_noise = np.percentile(energies, 20)
        e_speech = np.percentile(energies, 85)
        cand_thresh = max(0.005, e_noise + (e_speech - e_noise) * (0.12 + 0.15 * float(aggression)))

        is_speech_cand = np.zeros(num_frames, dtype=bool)
        pitch_f0s = np.zeros(num_frames)

        for m in range(num_frames):
            s_start = m * hop_len
            s_end = min(N, s_start + frame_len)
            raw_f = audio[s_start:s_end]
            if len(raw_f) < frame_len:
                raw_f = np.pad(raw_f, (0, frame_len - len(raw_f)))

            f0 = self._estimate_pitch(raw_f, sr, min_energy=cand_thresh * 0.8)
            pitch_f0s[m] = f0
            e = energies[m]

            # Speech frame: genuine pitch detected or energy exceeds candidate threshold
            if (f0 > 75.0 and e > cand_thresh * 0.8) or (e > cand_thresh):
                is_speech_cand[m] = True

        # Intra-word gap bridge (<= 14 frames / 224 ms at hop 256)
        bridged = np.copy(is_speech_cand)
        gap = 0
        for m in range(num_frames):
            if bridged[m]:
                if 0 < gap <= 14:
                    bridged[m - gap : m] = True
                gap = 0
            else:
                gap += 1

        # Reject isolated noise blips (<= 6 frames / 96 ms)
        final_vad = np.zeros(num_frames, dtype=bool)
        curr_start = None
        for m in range(num_frames):
            if bridged[m] and curr_start is None:
                curr_start = m
            elif not bridged[m] and curr_start is not None:
                seg_len = m - curr_start
                if seg_len >= 6:
                    final_vad[curr_start:m] = True
                curr_start = None
        if curr_start is not None and (num_frames - curr_start) >= 6:
            final_vad[curr_start:] = True

        # Step 2: Gain Mask Construction
        pause_db = 26.0 + 24.0 * float(aggression) # 36 dB to 50 dB pause attenuation
        pause_floor = 10.0 ** (-float(pause_db) / 20.0)

        gain_mask = np.full_like(mag, pause_floor)
        voiced_frames_count = 0

        for m in range(num_frames):
            if final_vad[m]:
                f0 = pitch_f0s[m]
                if f0 > 75.0 and harmonic_boost:
                    voiced_frames_count += 1
                    comb = np.full(num_freqs, 0.18 * (1.0 - aggression))
                    harmonics = np.arange(1, 16) * f0
                    for h in harmonics:
                        if h >= sr / 2:
                            break
                        bandwidth = max(35.0, 0.045 * h)
                        comb_peak = np.exp(-0.5 * ((f - h) / bandwidth) ** 2)
                        comb = np.maximum(comb, comb_peak)
                    gain_mask[:, m] = np.clip(comb, 0.15, 1.0)
                else:
                    gain_mask[:, m] = 1.0
            else:
                gain_mask[:, m] = pause_floor

        # Step 3: Asymmetric Temporal Smoothing (Fast Attack, Smooth Release)
        for k in range(num_freqs):
            row = gain_mask[k, :]
            for m in range(1, num_frames):
                if row[m] > row[m - 1]:
                    row[m] = 0.85 * row[m] + 0.15 * row[m - 1]
                else:
                    row[m] = 0.25 * row[m] + 0.75 * row[m - 1]
            gain_mask[k, :] = row

        # Step 4: Synthesis via iSTFT
        enhanced_Zxx = gain_mask * mag * np.exp(1j * phase)
        _, enhanced_audio = signal.istft(
            enhanced_Zxx, 
            fs=sr, 
            window=self.window, 
            nperseg=self.n_fft, 
            noverlap=self.n_fft - self.hop_length
        )

        if len(enhanced_audio) > N:
            enhanced_audio = enhanced_audio[:N]
        elif len(enhanced_audio) < N:
            enhanced_audio = np.pad(enhanced_audio, (0, N - len(enhanced_audio)))

        stats = {
            "voiced_speech_frames": int(voiced_frames_count),
            "total_frames": int(num_frames),
            "active_speech_percentage": round(float(100.0 * np.sum(final_vad) / max(1, num_frames)), 1),
            "voicing_percentage": round(float(100.0 * voiced_frames_count / max(1, num_frames)), 1),
            "pause_noise_expansion_db": round(float(pause_db), 1)
        }

        return enhanced_audio.astype(np.float32), stats
