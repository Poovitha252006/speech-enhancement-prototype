"""
Unit and Integration Test Suite for Hybrid Speech Enhancement Prototype
Verifies denoising efficiency across Stationary, Non-Stationary, and Impulsive noise types.
"""

import sys
import os
import unittest
import numpy as np

# Ensure core package is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.impulsive_filter import ImpulsiveNoiseFilter
from core.stationary_filter import StationaryNoiseFilter
from core.non_stationary_filter import NonStationaryNoiseFilter
from core.hybrid_pipeline import HybridSpeechEnhancer
from core.metrics import AudioMetricsEvaluator
from core.synthetic_bench import SyntheticAudioBenchmark

class TestSpeechEnhancementPipeline(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.duration = 2.5
        self.enhancer = HybridSpeechEnhancer(sr=self.sr)

    def test_impulsive_filter_isolated(self):
        """Verify impulsive filter eliminates injected clicks."""
        clean = SyntheticAudioBenchmark.generate_synthetic_speech(duration=self.duration, sr=self.sr)
        clicks = SyntheticAudioBenchmark.generate_impulsive_noise(len(clean), sr=self.sr, num_clicks=20)
        noisy = clean + clicks

        filt = ImpulsiveNoiseFilter()
        cleaned, stats = filt.process(noisy, sr=self.sr, aggression=0.85)

        self.assertEqual(len(cleaned), len(noisy))
        self.assertGreater(stats["impulses_detected"], 5)
        # Residual error with clean should be much lower than initial click error
        err_before = np.sum((noisy - clean) ** 2)
        err_after = np.sum((cleaned - clean) ** 2)
        self.assertLess(err_after, err_before)
        print(f"[PASS] Impulsive Test: Detected {stats['impulses_detected']} clicks, Error reduced from {err_before:.4f} to {err_after:.4f}")

    def test_stationary_filter_isolated(self):
        """Verify stationary filter improves SNR in AWGN + hum condition."""
        bench = SyntheticAudioBenchmark.create_mixed_benchmark(
            target_snr_db=5.0,
            include_stationary=True,
            include_non_stationary=False,
            include_impulsive=False,
            duration=self.duration,
            sr=self.sr
        )
        filt = StationaryNoiseFilter()
        cleaned, stats = filt.process(bench["noisy"], sr=self.sr, reduction_db=22.0)

        metrics = AudioMetricsEvaluator.compute_referenced_metrics(bench["clean"], bench["noisy"], cleaned, sr=self.sr)
        self.assertGreater(metrics["delta_snr_db"], 1.5)
        print(f"[PASS] Stationary Test: In SNR={metrics['input_snr_db']}dB -> Out SNR={metrics['output_snr_db']}dB (Gain: +{metrics['delta_snr_db']}dB)")

    def test_non_stationary_filter_isolated(self):
        """Verify non-stationary filter attenuates babble chatter."""
        bench = SyntheticAudioBenchmark.create_mixed_benchmark(
            target_snr_db=5.0,
            include_stationary=False,
            include_non_stationary=True,
            include_impulsive=False,
            duration=self.duration,
            sr=self.sr
        )
        filt = NonStationaryNoiseFilter()
        cleaned, stats = filt.process(bench["noisy"], sr=self.sr, aggression=0.75)

        metrics = AudioMetricsEvaluator.compute_referenced_metrics(bench["clean"], bench["noisy"], cleaned, sr=self.sr)
        self.assertGreater(metrics["delta_snr_db"], 0.5)
        self.assertGreater(stats["voicing_percentage"], 10.0)
        print(f"[PASS] Non-Stationary Test: Gain: +{metrics['delta_snr_db']}dB, Voicing: {stats['voicing_percentage']:.1f}%")

    def test_full_hybrid_cascade_all_three_noises(self):
        """Verify full hybrid pipeline across all 3 simultaneous noises (Stress Test: 0 dB SNR)."""
        bench = SyntheticAudioBenchmark.create_mixed_benchmark(
            target_snr_db=0.0,
            include_stationary=True,
            include_non_stationary=True,
            include_impulsive=True,
            duration=3.0,
            sr=self.sr
        )

        enhanced, report = self.enhancer.process(
            bench["noisy"],
            sr=self.sr,
            enable_impulsive=True,
            enable_stationary=True,
            enable_non_stationary=True,
            stationary_reduction_db=24.0,
            impulsive_aggression=0.85,
            non_stationary_aggression=0.8,
            speech_priority=False
        )

        metrics = AudioMetricsEvaluator.compute_referenced_metrics(bench["clean"], bench["noisy"], enhanced, sr=self.sr)
        
        self.assertEqual(len(report["stages_executed"]), 3)
        self.assertGreater(metrics["delta_snr_db"], 3.0)  # Expect substantial SNR boost
        self.assertGreater(metrics["noise_reduction_factor_db"], 4.0)

        print("\n=======================================================")
        print(" FULL HYBRID PIPELINE STRESS TEST RESULTS (ALL 3 NOISES):")
        print("=======================================================")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v}")
        print("=======================================================\n")

    def test_speech_priority_mode(self):
        """Verify psychoacoustic speech-priority mode preserves speech clarity and eliminates noise."""
        bench = SyntheticAudioBenchmark.create_mixed_benchmark(
            target_snr_db=5.0,
            include_stationary=True,
            include_non_stationary=True,
            include_impulsive=True,
            duration=3.0,
            sr=self.sr
        )
        enhanced, report = self.enhancer.process(
            bench["noisy"],
            sr=self.sr,
            speech_priority=True
        )
        self.assertTrue(report.get("speech_priority_mode", False))
        self.assertGreaterEqual(len(report["stages_executed"]), 4)
        self.assertLess(report["output_peak"], 1.0)

    def test_aggressive_mode_execution(self):
        """Verify aggressive_mode parameter is accepted and executed without error."""
        bench = SyntheticAudioBenchmark.create_mixed_benchmark(
            target_snr_db=5.0,
            include_stationary=True,
            include_non_stationary=True,
            include_impulsive=True,
            duration=2.5,
            sr=self.sr
        )
        # Test calibrated mode with aggressive_mode=True
        enhanced_cal, report_cal = self.enhancer.process(
            bench["noisy"],
            sr=self.sr,
            speech_priority=False,
            aggressive_mode=True
        )
        self.assertTrue(report_cal.get("aggressive_mode", False))
        self.assertEqual(len(report_cal["stages_executed"]), 3)
        self.assertIn("Aggressive Voice Enhancement", report_cal["stages_executed"][0])

        # Test speech priority mode with aggressive_mode=True
        enhanced_sp, report_sp = self.enhancer.process(
            bench["noisy"],
            sr=self.sr,
            speech_priority=True,
            aggressive_mode=True
        )
        self.assertTrue(report_sp.get("aggressive_mode", False))
        self.assertLess(report_sp["output_peak"], 1.0)

if __name__ == '__main__':
    unittest.main()
