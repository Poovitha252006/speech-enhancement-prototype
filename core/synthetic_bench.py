"""
Synthetic Benchmark Generator for ECE Prototype Evaluation
Generates clean speech signals and mixes them with controlled proportions of:
1. Stationary Noise (AWGN, 50Hz AC hum with harmonics, HVAC low rumble)
2. Non-Stationary Noise (Dynamic crowd babble simulation, swept sirens, fluctuating noise)
3. Impulsive Noise (Poisson-distributed clicks, pops, switch bursts)
At specified target SNR levels (-5 dB, 0 dB, +5 dB, +10 dB, +15 dB).
"""

import numpy as np
from scipy import signal

class SyntheticAudioBenchmark:
    """
    Produces calibrated speech and noise cocktails for objective ECE testing and viva demonstration.
    """
    @staticmethod
    def generate_synthetic_speech(duration: float = 3.5, sr: int = 16000) -> np.ndarray:
        """
        Synthesize realistic multi-formant human speech with vowel transitions and natural cadence.
        Formants: /a/ (700, 1200, 2400 Hz), /i/ (300, 2200, 3000 Hz), /o/ (500, 900, 2400 Hz).
        """
        num_samples = int(duration * sr)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Natural vocal pitch contour (gliding between 120 Hz and 150 Hz with vibrato)
        f0 = 135.0 + 15.0 * np.sin(2 * np.pi * 1.5 * t) + 3.0 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sr

        # Pulse train excitation (glottal source)
        glottal_source = np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.35 * np.sin(3 * phase) + 0.2 * np.sin(4 * phase)

        # Formant filters (Resonant bandpass filters simulating vocal tract)
        def apply_formant(src, fc, bw):
            q = fc / bw
            b, a = signal.iirpeak(fc, q, fs=sr)
            return signal.lfilter(b, a, src)

        # Create 3 spoken syllables: /ba/ -> /ti/ -> /ko/
        seg_len = num_samples // 3
        speech = np.zeros(num_samples)

        # Syllable 1: /ba/ (formants ~ 700 Hz, 1220 Hz, 2600 Hz)
        s1 = apply_formant(glottal_source[:seg_len], 700, 100) * 1.2 + \
             apply_formant(glottal_source[:seg_len], 1220, 140) * 0.8 + \
             apply_formant(glottal_source[:seg_len], 2600, 200) * 0.4
        
        # Syllable 2: /ti/ (formants ~ 320 Hz, 2250 Hz, 2900 Hz)
        s2 = apply_formant(glottal_source[seg_len:2*seg_len], 320, 80) * 1.4 + \
             apply_formant(glottal_source[seg_len:2*seg_len], 2250, 160) * 0.9 + \
             apply_formant(glottal_source[seg_len:2*seg_len], 2900, 220) * 0.5

        # Syllable 3: /ko/ (formants ~ 500 Hz, 900 Hz, 2400 Hz)
        s3 = apply_formant(glottal_source[2*seg_len:], 500, 90) * 1.3 + \
             apply_formant(glottal_source[2*seg_len:], 900, 120) * 0.7 + \
             apply_formant(glottal_source[2*seg_len:], 2400, 200) * 0.35

        # Smooth envelope (attack, sustain, release with pauses between syllables)
        env1 = signal.windows.tukey(seg_len, alpha=0.35)
        env2 = signal.windows.tukey(seg_len, alpha=0.35)
        rem = num_samples - 2 * seg_len
        env3 = signal.windows.tukey(rem, alpha=0.35)

        speech[:seg_len] = s1 * env1
        speech[seg_len:2*seg_len] = s2 * env2
        speech[2*seg_len:] = s3 * env3

        # Add unvoiced fricatives (/s/ sound burst) at beginning of syllable 2
        fricative_noise = np.random.normal(0, 0.25, int(0.08 * sr))
        b_hp, a_hp = signal.butter(4, 4000 / (sr / 2), btype='high')
        fricative_filtered = signal.lfilter(b_hp, a_hp, fricative_noise)
        speech[seg_len : seg_len + len(fricative_filtered)] += fricative_filtered * 0.7

        # Normalize to peak 0.8
        speech = speech / (np.max(np.abs(speech)) + 1e-8) * 0.8
        return speech.astype(np.float32)

    @staticmethod
    def generate_stationary_noise(num_samples: int, sr: int = 16000) -> np.ndarray:
        """
        Generate fan hum + thermal white noise + 50Hz electrical hum with harmonics.
        """
        t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)
        # 50 Hz power hum and harmonics (100 Hz, 150 Hz)
        hum = 0.35 * np.sin(2 * np.pi * 50 * t) + 0.18 * np.sin(2 * np.pi * 100 * t) + 0.08 * np.sin(2 * np.pi * 150 * t)
        
        # Thermal white noise
        white = np.random.normal(0, 0.5, num_samples)
        
        # Fan low-pass rumble
        b_lp, a_lp = signal.butter(3, 400 / (sr / 2), btype='low')
        rumble = signal.lfilter(b_lp, a_lp, np.random.normal(0, 0.8, num_samples))
        
        stationary = hum + 0.4 * white + 0.8 * rumble
        return (stationary / (np.max(np.abs(stationary)) + 1e-8)).astype(np.float32)

    @staticmethod
    def generate_non_stationary_noise(num_samples: int, sr: int = 16000) -> np.ndarray:
        """
        Generate dynamic crowd speech babble and modulated acoustic interference.
        """
        t = np.linspace(0, num_samples / sr, num_samples, endpoint=False)
        
        # Multiple interfering pseudo-voices with wandering pitch (crowd babble)
        babble = np.zeros(num_samples)
        for seed_f0, rate in [(190, 0.8), (240, 1.2), (110, 0.5), (170, 1.7)]:
            pitch_w = seed_f0 + 25.0 * np.sin(2 * np.pi * rate * t)
            ph = 2 * np.pi * np.cumsum(pitch_w) / sr
            # Modulate amplitude dynamically (chatter bursts)
            amp_mod = np.clip(np.sin(2 * np.pi * (rate * 1.5) * t) ** 2, 0.1, 1.0)
            voice_int = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * amp_mod
            babble += voice_int

        # Add swept siren component in the background
        siren_freq = 600 + 400 * np.sin(2 * np.pi * 0.7 * t)
        siren_phase = 2 * np.pi * np.cumsum(siren_freq) / sr
        siren = 0.3 * np.sin(siren_phase)

        non_stat = babble + siren
        return (non_stat / (np.max(np.abs(non_stat)) + 1e-8)).astype(np.float32)

    @staticmethod
    def generate_impulsive_noise(num_samples: int, sr: int = 16000, num_clicks: int = 14) -> np.ndarray:
        """
        Generate sparse high-amplitude impulse clicks, mic pops, and switch bursts.
        """
        impulse = np.zeros(num_samples, dtype=np.float32)
        click_positions = np.random.choice(np.arange(100, num_samples - 100), size=num_clicks, replace=False)

        for pos in click_positions:
            # Click width between 3 and 12 samples
            width = np.random.randint(3, 10)
            polarity = np.random.choice([-1.0, 1.0])
            amplitude = np.random.uniform(0.7, 0.98) * polarity
            
            # Sharp decay waveform
            click_shape = amplitude * np.exp(-np.linspace(0, 3, width))
            impulse[pos : pos + width] += click_shape

        return impulse

    @classmethod
    def create_mixed_benchmark(
        cls,
        target_snr_db: float = 0.0,
        include_stationary: bool = True,
        include_non_stationary: bool = True,
        include_impulsive: bool = True,
        duration: float = 3.5,
        sr: int = 16000
    ) -> dict:
        """
        Create calibrated audio cocktail with exact SNR and mixture breakdown.
        """
        clean_speech = cls.generate_synthetic_speech(duration=duration, sr=sr)
        N = len(clean_speech)

        total_noise = np.zeros(N, dtype=np.float32)
        components_added = []

        if include_stationary:
            stat_noise = cls.generate_stationary_noise(N, sr=sr)
            total_noise += 0.5 * stat_noise
            components_added.append("Stationary (Fan Hum, 50Hz AC, White Hiss)")

        if include_non_stationary:
            non_stat_noise = cls.generate_non_stationary_noise(N, sr=sr)
            total_noise += 0.5 * non_stat_noise
            components_added.append("Non-Stationary (Crowd Babble, Modulated Siren)")

        # Scale continuous noise to achieve target SNR
        if len(components_added) > 0:
            speech_power = np.mean(clean_speech ** 2)
            noise_power = np.mean(total_noise ** 2) + 1e-12
            required_noise_power = speech_power / (10.0 ** (target_snr_db / 10.0))
            scale_factor = np.sqrt(required_noise_power / noise_power)
            total_noise = total_noise * scale_factor

        # Add impulsive noise spikes on top
        if include_impulsive:
            impulsive = cls.generate_impulsive_noise(N, sr=sr, num_clicks=15)
            total_noise += impulsive
            components_added.append("Impulsive (Sharp Pops, Clicks, Micro-Transients)")

        noisy_speech = clean_speech + total_noise
        # Normalize to prevent digital hard clipping
        max_val = np.max(np.abs(noisy_speech))
        if max_val > 0.98:
            norm_factor = 0.98 / max_val
            clean_speech = clean_speech * norm_factor
            total_noise = total_noise * norm_factor
            noisy_speech = noisy_speech * norm_factor

        return {
            "clean": clean_speech.astype(np.float32),
            "noise": total_noise.astype(np.float32),
            "noisy": noisy_speech.astype(np.float32),
            "target_snr_db": target_snr_db,
            "components": components_added,
            "sr": sr,
            "duration": duration
        }
