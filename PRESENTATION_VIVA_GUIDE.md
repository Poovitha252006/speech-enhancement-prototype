# Viva Voce Defense & Presentation Guide: Hybrid Speech Enhancement Prototype

---

## Part 1: Slide-by-Slide Defense Presentation Outline

### Slide 1: Title Slide
* **Title:** Hybrid Multi-Stage Speech Enhancement System for Stationary, Non-Stationary, and Impulsive Acoustic Environments
* **Domain:** Digital Signal Processing (DSP) & Communication Systems
* **Presenter:** Student Name / Roll Number | Department of Electronics & Communication Engineering (ECE)
* **Guide / Supervisor:** Faculty Guide Name

### Slide 2: Problem Statement & Motivation
* **Key Points:**
  * Real-world audio capture in smartphones, hearing aids, VoIP, and conference systems is degraded by multiple concurrent noise sources.
  * Noise falls into 3 mathematically distinct classes: Impulsive (clicks/pops), Stationary (fan/AC/hiss), and Non-Stationary (human babble/sirens).
  * Existing single-filter systems fail: spectral subtraction introduces "musical noise"; Wiener filters smear impulses into prolonged thuds; deep learning requires massive GPU power.
* **Speaker Script:** *"In real environments, background noise is rarely purely stationary or purely impulsive. When a user speaks in a cafe with fans and typing sounds, classical filters either corrupt the voice or leave significant noise. Our objective was to design a hybrid multi-stage architecture that cleanly handles all three noise categories simultaneously."*

### Slide 3: Proposed Hybrid System Architecture
* **Key Points:**
  * Cascaded Multi-Stage Pipeline:
    1. **Stage 1 (Time Domain):** Adaptive Hampel Filter (MAD) + Cubic Spline Interpolator.
    2. **Stage 2 (STFT Domain):** Minimum Statistics Noise PSD Tracking + Decision-Directed (DD) Wiener Filter.
    3. **Stage 3 (Subband & Pitch Domain):** 32-Band Bark Filterbank + Pitch-Synchronous Harmonic Comb Filter.
    4. **Stage 4 (Synthesis):** Overlap-Add (OLA) + Perceptual Soft Knee Limiter.
* **Speaker Script:** *"Instead of attempting to solve all noise phenomena in a single domain, our system decouples the signals: sharp time-domain transients are repaired first, followed by frequency-domain stationary removal, and finally pitch-synchronous harmonic isolation for speech babble."*

### Slide 4: Stage 1 – Impulsive Noise Despeckler
* **Key Points:**
  * Clicks and pops are non-Gaussian outliers with steep derivatives.
  * Rolling Median: \( \mu_{1/2}[n] \).
  * Median Absolute Deviation: \( \text{MAD}[n] = 1.4826 \times \text{median}(|x[n+k] - \mu_{1/2}[n]|) \).
  * Threshold condition: \( |x[n] - \mu_{1/2}| > \tau \cdot \text{MAD} \).
  * Reconstruction: Contiguous corrupted spans are reconstructed using Cubic Hermite Splines anchored on clean boundary points.
* **Speaker Script:** *"Because linear filters smear impulses across time, we detect clicks directly in the time domain using the robust Median Absolute Deviation. Detected click intervals are then seamlessly bridged via cubic spline interpolation without losing speech formants."*

### Slide 5: Stage 2 – Stationary Noise Suppressor
* **Key Points:**
  * 1024-point STFT, Hann window, 75% overlap.
  * Martin's Minimum Statistics tracks noise floor \( \lambda_d(k, m) \) during continuous speech without requiring an explicit silence detector.
  * Decision-Directed (DD) a priori SNR estimation (\( \alpha_{dd} = 0.96 \)):
    \[
    \xi(k, m) = \alpha \frac{|\hat{S}(k, m-1)|^2}{\lambda_d(k, m-1)} + (1-\alpha) \max(\gamma(k, m) - 1, 0)
    \]
  * Wiener Gain with Spectral Floor \( \beta_{\text{floor}} = -32\text{ dB} \) prevents musical noise tone bursts.
* **Speaker Script:** *"Stationary noise like fan hum and thermal hiss is removed using an Ephraim-Malah Decision-Directed Wiener filter. By setting a strict spectral floor of -32 dB and smoothing the a priori SNR, we completely eliminate musical noise."*

### Slide 6: Stage 3 – Non-Stationary Noise & Babble Suppressor
* **Key Points:**
  * Background babble shares frequencies with speech, making stationary subtraction ineffective.
  * Voiced speech has quasi-periodic glottal pulse excitation with fundamental pitch \( F_0 \in [80, 450]\text{ Hz} \).
  * Normalized Autocorrelation estimates \( F_0 \).
  * Pitch-Synchronous Harmonic Comb Filter passes integer multiples \( h \cdot F_0 \) while attenuating inter-harmonic valleys containing diffuse babble energy.
  * 32-Band Bark critical filterbank suppresses high-flux non-speech frames.
* **Speaker Script:** *"Babble noise is notoriously difficult because it is composed of human speech. We exploit the harmonic structure of the target speaker: voiced vowels have distinct spectral peaks at multiples of F0. Our harmonic comb filter preserves these voice formants while attenuating the inter-harmonic valleys where background babble resides."*

### Slide 7: Experimental Setup & Benchmarking
* **Key Points:**
  * Tested at 16 kHz sampling rate across 4 calibrated scenarios.
  * Target SNRs evaluated from -10 dB (extreme) up to +15 dB (mild).
  * Objective Metrics: Global SNR, Segmental SNR (SSNR), Noise Reduction Factor (NRF), Speech Distortion Index (SDI), Estimated PESQ (MOS), and STOI.

### Slide 8: Results & Performance Table
* **Key Points:**
  * Impulsive Clicks: **+6.60 dB gain**, STOI = 0.990 (Near perfect preservation).
  * Stationary Hum/Hiss: **+8.16 dB gain**, SSNR gain = +8.33 dB, PESQ = 3.64.
  * Non-Stationary Babble: **+3.09 dB gain**, PESQ = 3.03.
  * **All 3 Noises Combined (0 dB SNR Extreme Stress Test):**
    * Input SNR: -0.01 dB \(\rightarrow\) Output SNR: **+6.32 dB (+6.33 dB Gain)**
    * Segmental SNR Gain: **+6.23 dB**
    * Intelligibility (STOI): **0.821**

### Slide 9: Spectrogram & Waveform Visual Analysis
* **Key Points:**
  * Time domain waveforms demonstrate elimination of vertical impulse spikes and noise envelope compression.
  * Dual STFT Spectrogram Heatmaps (0 to 8000 Hz) clearly display complete eradication of the 50 Hz hum rail and broad-band hiss while preserving vocal formant trajectories (F1, F2, F3).

### Slide 10: Real-Time Hardware Feasibility & Edge Deployment
* **Key Points:**
  * Latency: 16 ms frame processing delay (hop size 256 samples at 16 kHz).
  * Complexity: \( O(N \log N) \), runs faster than 0.15× real-time on standard CPUs (Raspberry Pi, ARM Cortex-A53, Intel Core).
  * Memory: Less than 18 MB RAM consumption. Zero GPU requirement.

### Slide 11: Live Prototype Demonstration
* **Key Points:**
  * Web Dashboard Walkthrough (`http://localhost:8501`).
  * Live microphone recording test.
  * Real-time file upload & before/after A-B audio playback.
  * Synthetic benchmark cocktail generator.

### Slide 12: Conclusion & Future Scope
* **Key Points:**
  * Successfully delivered an autonomous, mathematically robust speech enhancement prototype.
  * Solved impulsive, stationary, and non-stationary noises concurrently.
  * Future Work: Dual-channel spatial beamforming, subband TinyML neural post-filtering.

---

## Part 2: Comprehensive Viva Voce Questions & Answers

### DSP & Theoretical Foundations

#### Q1: Why is the scale factor 1.4826 used in the Median Absolute Deviation (MAD)?
**Answer:** In a standard normal distribution \( X \sim \mathcal{N}(0, \sigma^2) \), the 75th percentile (the upper quartile) is approximately \( 0.6745 \cdot \sigma \). Consequently, the median of the absolute values is:
\[
\text{median}(|X|) = \Phi^{-1}(0.75) \cdot \sigma \approx 0.6745 \cdot \sigma
\]
To make MAD an unbiased estimator of the standard deviation \( \sigma \) for Gaussian processes, we multiply by the reciprocal:
\[
\frac{1}{0.6745} \approx 1.4826
\]
This allows the Hampel filter to accurately gauge background noise variance without being corrupted by the outliers it is trying to detect.

#### Q2: What is the difference between A Posteriori SNR and A Priori SNR?
**Answer:**
* **A Posteriori SNR (\( \gamma_k \)):** Computed directly from the observed noisy frame:
  \[
  \gamma_k = \frac{|Y_k|^2}{\lambda_{d, k}}
  \]
  It measures the instantaneous ratio of the received signal power to the estimated noise power in frequency bin \( k \).
* **A Priori SNR (\( \xi_k \)):** Represents the expected clean speech power relative to noise power:
  \[
  \xi_k = \frac{E[|S_k|^2]}{\lambda_{d, k}}
  \]
  Because the clean signal \( S_k \) is unknown, Ephraim and Malah proposed the **Decision-Directed (DD)** recursion:
  \[
  \xi_k(m) = \alpha \frac{|\hat{S}_k(m-1)|^2}{\lambda_{d, k}(m-1)} + (1-\alpha) \max(\gamma_k(m) - 1, 0)
  \]
  This links the previous frame's estimate with the current observation, eliminating the rapid gain fluctuations responsible for musical noise.

#### Q3: Why is 75% overlap (hop size = N/4) used in the STFT rather than 50%?
**Answer:** A Hann window satisfies the Constant-Overlap-Add (COLA) condition at both 50% and 75% overlap. However, when non-linear frequency-domain spectral modifications (such as Wiener gains or masking) are applied, the modified spectrum no longer matches a valid Fourier transform of a single time-domain signal. A 75% overlap (hop size = 256 for window = 1024) provides quadrupled temporal redundancy, which smooths boundary discontinuities and prevents frame modulation artifacts upon inverse STFT (iSTFT).

#### Q4: How does Martin's Minimum Statistics work without a Voice Activity Detector (VAD)?
**Answer:** Traditional speech enhancers rely on a VAD to update noise estimates only during speech pauses. If the VAD fails (e.g., during low SNR), speech is erroneously classified as noise and wiped out. Minimum Statistics observes that speech power in any individual frequency bin fluctuates rapidly between phonemes and pauses. Even during active continuous speech, speech energy in any given bin drops to the noise floor within a window of 0.5 to 1.5 seconds. By continuously tracking the minimum value of smoothed power over a sliding buffer, the noise PSD is reliably estimated without ever needing an explicit hard-decision VAD.

#### Q5: How is the pitch frequency \( F_0 \) extracted in the presence of noise?
**Answer:** Normalized autocorrelation in the time domain is highly robust to additive zero-mean noise. Because additive noise is uncorrelated with the periodic vocal signal, the noise autocorrelation concentrates almost entirely at lag \( k = 0 \). At lags corresponding to the pitch period \( T_0 = \frac{f_s}{F_0} \), the periodic speech signal produces distinct secondary correlation peaks. By searching within the physiological human vocal bounds (80 Hz to 450 Hz) and validating that the normalized peak exceeds 0.35, the fundamental pitch \( F_0 \) is accurately identified even under heavy noise.

#### Q6: What is the Speech Distortion Index (SDI)?
**Answer:** The Speech Distortion Index measures how much the enhanced signal \( \hat{s}[n] \) deviates from the original speech \( s[n] \), normalized by the clean speech power:
\[
\text{SDI} = \frac{\sum_n (s[n] - \alpha_{\text{opt}} \hat{s}[n])^2}{\sum_n s^2[n]}
\]
A lower SDI (close to 0) indicates that vocal formants, consonants, and speech timbre have been preserved without muffling or mutilation.

---

## Part 3: Live Demonstration Walkthrough Script (For Examiners)

1. **Launch Web Server:**
   ```bash
   python server.py
   ```
   Open browser at: `http://localhost:8501`

2. **Step 1: Calibrated Benchmark Stress Test:**
   * Click on the **"Calibrated Benchmark & Stress Test"** tab.
   * Check all 3 noise types: Stationary (50Hz + fan), Non-Stationary (Babble), Impulsive (clicks).
   * Set Target SNR to **0 dB (Severe Noise)**.
   * Click **"Generate Synthetic Cocktail & Test Denoising"**.
   * Play the three audio tracks:
     * *Clean Speech:* Listen to the clear sentence.
     * *Noisy Cocktail:* Listen to the completely masked, unintelligible audio with clicks, hum, and babble.
     * *Enhanced Output:* Listen to how all clicks, hum, and babble are removed, restoring clear intelligible voice.
   * Point the examiner to the **+6.33 dB SNR gain** and **+6.23 dB SSNR gain** in the live table.

3. **Step 2: Real-World Audio / Live Microphone Test:**
   * Switch to the **"Denoising Studio"** tab.
   * Click **"Start Microphone Recording"**, speak a sentence while tapping a pen on the desk (impulsive click) or with fan noise.
   * Click **"Stop Recording"** then **"Clean Audio"**.
   * Play the raw and cleaned audio side-by-side.
   * Show the **Time-Domain Waveform** and **Dual Spectrogram Heatmaps** illustrating the precise frequency bands where noise energy was stripped away.
