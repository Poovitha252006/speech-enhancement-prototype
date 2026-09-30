"""
Standalone Demonstration Runner for Speech Enhancement Prototype
Generates benchmark samples, runs isolated and cascaded denoising,
saves audio files to 'demo_outputs/', and displays the ECE evaluation score table.
"""

import os
import sys
import numpy as np
import soundfile as sf

from core.hybrid_pipeline import HybridSpeechEnhancer
from core.metrics import AudioMetricsEvaluator
from core.synthetic_bench import SyntheticAudioBenchmark

def run_demonstration():
    print("=" * 75)
    print(" ECE SPEECH ENHANCEMENT PROTOTYPE - BENCHMARK & DEMO RUNNER")
    print(" Targeting: Stationary, Non-Stationary, and Impulsive Background Noises")
    print("=" * 75)

    output_dir = os.path.join(os.path.dirname(__file__), "demo_outputs")
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    duration = 3.5

    enhancer = HybridSpeechEnhancer(sr=sr)

    # Scenarios to test
    scenarios = [
        {
            "name": "1_Impulsive_Clicks_Only",
            "desc": "Clean Speech + Poisson-distributed sharp clicks and pops",
            "target_snr": 8.0,
            "stat": False, "non_stat": False, "imp": True,
            "cfg": {"enable_impulsive": True, "enable_stationary": False, "enable_non_stationary": False, "impulsive_aggression": 0.9}
        },
        {
            "name": "2_Stationary_Hum_Hiss",
            "desc": "Clean Speech + 50Hz AC hum, fan low rumble, and thermal white hiss",
            "target_snr": 3.0,
            "stat": True, "non_stat": False, "imp": False,
            "cfg": {"enable_impulsive": False, "enable_stationary": True, "enable_non_stationary": False, "stationary_reduction_db": 32.0}
        },
        {
            "name": "3_NonStationary_Babble",
            "desc": "Clean Speech + Multi-talker crowd babble and wandering siren",
            "target_snr": 4.0,
            "stat": False, "non_stat": True, "imp": False,
            "cfg": {"enable_impulsive": False, "enable_stationary": False, "enable_non_stationary": True, "non_stationary_aggression": 0.85}
        },
        {
            "name": "4_All_Three_Noises_StressTest",
            "desc": "Clean Speech + Stationary + Non-Stationary + Impulsive (0 dB SNR)",
            "target_snr": 0.0,
            "stat": True, "non_stat": True, "imp": True,
            "cfg": {"enable_impulsive": True, "enable_stationary": True, "enable_non_stationary": True, "impulsive_aggression": 0.85, "stationary_reduction_db": 32.0, "non_stationary_aggression": 0.85}
        }
    ]

    results = []

    for item in scenarios:
        print(f"\n[*] Processing Scenario: {item['name']}")
        print(f"    Description: {item['desc']}")

        bench = SyntheticAudioBenchmark.create_mixed_benchmark(
            target_snr_db=item["target_snr"],
            include_stationary=item["stat"],
            include_non_stationary=item["non_stat"],
            include_impulsive=item["imp"],
            duration=duration,
            sr=sr
        )

        clean = bench["clean"]
        noisy = bench["noisy"]

        # Run enhancer
        enhanced, report = enhancer.process(
            noisy,
            sr=sr,
            **item["cfg"]
        )

        # Compute objective metrics
        metrics = AudioMetricsEvaluator.compute_referenced_metrics(clean, noisy, enhanced, sr=sr)
        results.append({
            "name": item["name"],
            "in_snr": metrics["input_snr_db"],
            "out_snr": metrics["output_snr_db"],
            "delta_snr": metrics["delta_snr_db"],
            "delta_ssnr": metrics["delta_ssnr_db"],
            "nrf": metrics["noise_reduction_factor_db"],
            "mos": metrics["estimated_pesq_mos"],
            "stoi": metrics["estimated_stoi"],
            "rating": metrics["quality_rating"]
        })

        # Save audio files
        clean_file = os.path.join(output_dir, f"{item['name']}_0_clean.wav")
        noisy_file = os.path.join(output_dir, f"{item['name']}_1_noisy.wav")
        enhanced_file = os.path.join(output_dir, f"{item['name']}_2_enhanced.wav")
        noise_diff_file = os.path.join(output_dir, f"{item['name']}_3_removed_noise.wav")

        sf.write(clean_file, clean, sr)
        sf.write(noisy_file, noisy, sr)
        sf.write(enhanced_file, enhanced, sr)
        sf.write(noise_diff_file, noisy - enhanced, sr)

        print(f"    -> In SNR: {metrics['input_snr_db']} dB | Out SNR: {metrics['output_snr_db']} dB | Delta: +{metrics['delta_snr_db']} dB")
        print(f"    -> PESQ/MOS Est: {metrics['estimated_pesq_mos']} | STOI: {metrics['estimated_stoi']}")
        print(f"    -> Audio saved: {enhanced_file}")

    print("\n" + "=" * 95)
    print(f"{'SCENARIO':<32} | {'IN SNR':<8} | {'OUT SNR':<8} | {'GAIN':<8} | {'SSNR GAIN':<10} | {'PESQ':<6} | {'STOI':<6}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<32} | {r['in_snr']:>6.1f} dB | {r['out_snr']:>6.1f} dB | +{r['delta_snr']:>5.2f} dB | +{r['delta_ssnr']:>7.2f} dB | {r['mos']:>5.2f}  | {r['stoi']:>5.2f}")
    print("=" * 95)
    print(f"\n[OK] Demonstration complete! All audio files written to:\n     {output_dir}\n")

if __name__ == '__main__':
    run_demonstration()
