"""Numerical/shape tests use geometric patterns and tones, never enrollment data."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler

from commands import recognize_command
from features import (extract_face_features, extract_voice_features, extract_fingerprint_features,
                      create_fusion_features, preprocess_audio)
from train_models import calibrate_threshold, binary_counts
from verification import verify_face, verify_voice, verify_fusion, verify_fingerprint


class FeatureTests(unittest.TestCase):
    def setUp(self):
        # Diagnostic checkerboard, not a fabricated face or fingerprint.
        self.image = ((np.indices((200, 200)).sum(axis=0) // 7) % 2 * 255).astype(np.uint8)

    def test_face_hog_lbp_dimensions_and_histogram(self):
        vector = extract_face_features(self.image)
        self.assertEqual(vector.shape, (4366,))
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(vector[-10:].sum(), 1)

    def test_fingerprint_dimensions(self):
        vector = extract_fingerprint_features(self.image)
        self.assertEqual(vector.shape, (8110,))
        self.assertAlmostEqual(vector[-10:].sum(), 1)

    def test_voice_30_features_and_stereo_resampling(self):
        rate = 8000
        t = np.arange(rate * 2) / rate
        tone = (0.2 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
        vector = extract_voice_features(np.column_stack((tone, tone)), rate)
        self.assertEqual(vector.shape, (30,))
        self.assertTrue(np.isfinite(vector).all())

    def test_file_and_array_preprocessing_are_identical(self):
        tone = (0.2 * np.sin(2 * np.pi * 300 * np.arange(32000) / 16000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic_tone.wav"
            sf.write(path, tone, 16000, subtype="FLOAT")
            np.testing.assert_allclose(extract_voice_features(path), extract_voice_features(tone), atol=1e-6)

    def test_invalid_audio_rejected(self):
        for audio in ([], np.zeros(32000), np.full(32000, np.nan), np.ones(3), np.ones(32000) * 10):
            with self.subTest(shape=np.shape(audio)), self.assertRaises(ValueError):
                extract_voice_features(audio)

    def test_invalid_images_rejected(self):
        for image in (None, np.zeros((100, 100)), np.zeros((2, 2)), np.full((100, 100), np.nan)):
            with self.subTest(shape=np.shape(image)), self.assertRaises(ValueError):
                extract_face_features(image)

    def test_fusion_scales_separately_then_concatenates(self):
        face_scaler = StandardScaler().fit([[0, 10], [2, 30]])
        voice_scaler = StandardScaler().fit([[100], [300]])
        vector = create_fusion_features([2, 30], [100], face_scaler, voice_scaler)
        np.testing.assert_allclose(vector, [1, 1, -1])
        # Transforming an extreme inference sample never refits either scaler.
        create_fusion_features([200, 300], [10000], face_scaler, voice_scaler)
        np.testing.assert_allclose(face_scaler.mean_, [1, 20])
        np.testing.assert_allclose(voice_scaler.mean_, [200])

    def test_missing_artifacts_reject_all_modalities_and_command(self):
        with tempfile.TemporaryDirectory() as folder:
            for verifier in (verify_face, verify_voice, verify_fusion, verify_fingerprint):
                result = verifier([1.0], "user_01", folder)
                self.assertFalse(result["accepted"])
            self.assertEqual(recognize_command([1.0], folder)["recognized_command"], "UNKNOWN")

    def test_corrupt_bundle_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "face.joblib").write_bytes(b"not a model")
            self.assertFalse(verify_face([1], "user_01", folder)["accepted"])

    def test_threshold_uses_validation_genuine_and_wrong_claims(self):
        threshold, counts = calibrate_threshold(np.array([[0.1], [9.9]]),
            np.array(["a", "b"]), np.array(["a", "b"]), {"a": np.array([0.0]), "b": np.array([10.0])})
        self.assertAlmostEqual(threshold, 0.1)
        self.assertEqual((counts["TP"], counts["FP"], counts["FN"], counts["TN"]), (2, 0, 0, 2))

    def test_prediction_disagreement_rejects_even_at_small_distance(self):
        from sklearn.neighbors import KNeighborsClassifier
        from verification import decide_normalized
        model = KNeighborsClassifier(n_neighbors=3).fit([[0], [0.1], [0.2]], ["b", "b", "b"])
        artifact = {"dimension": 1, "models": {"selected": model}, "selected": "selected",
                    "templates": {"a": np.array([0.0])}, "threshold": 100.0}
        self.assertFalse(decide_normalized([0.0], "a", artifact)["accepted"])

    def test_zero_denominators_are_explicit(self):
        counts = binary_counts([], [])
        self.assertIsNone(counts["FAR"])
        self.assertIsNone(counts["FRR"])
