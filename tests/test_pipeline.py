"""Branch tests use explicit doubles; these are NOT biometric accuracy results."""
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

import pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.calls = {}
        values = {
            "capture_image": np.zeros((2, 2)), "record_audio": np.zeros(2),
            "capture_fingerprint": np.zeros((2, 2)),
            "extract_face_features": np.array([1.0]), "extract_voice_features": np.array([2.0]),
            "extract_fingerprint_features": np.array([3.0]), "create_fusion_features": np.array([1.0, 2.0]),
            "recognize_command": {"accepted": True, "recognized_command": "UNLOCK", "reason": "test"},
            "execute_command": {"message": "test action", "session_ended": False},
        }
        for name in ("verify_face", "verify_voice", "verify_fusion", "verify_fingerprint"):
            values[name] = {"accepted": True, "reason": "test acceptance"}
        for name, value in values.items():
            self.calls[name] = self.stack.enter_context(patch.object(pipeline, name, return_value=value))

    def test_verified_admin_unlock_allowed(self):
        result = pipeline.run_pipeline("user_01")
        self.assertTrue(result["authorized"])
        self.calls["execute_command"].assert_called_once_with("UNLOCK")

    def test_verified_employee_unlock_denied(self):
        result = pipeline.run_pipeline("user_02")
        self.assertTrue(result["biometrics_accepted"])
        self.assertEqual(result["rejected_stage"], "permission")
        self.calls["execute_command"].assert_not_called()

    def test_verified_guest_report_denied(self):
        self.calls["recognize_command"].return_value["recognized_command"] = "REPORT"
        self.assertFalse(pipeline.run_pipeline("user_04")["authorized"])
        self.calls["execute_command"].assert_not_called()

    def test_verified_employee_status_allowed(self):
        self.calls["recognize_command"].return_value["recognized_command"] = "STATUS"
        self.assertTrue(pipeline.run_pipeline("user_02")["authorized"])
        self.calls["execute_command"].assert_called_once_with("STATUS")

    def test_every_failed_gate_stops_remaining_stages(self):
        checks = ["verify_face", "verify_voice", "verify_fusion", "verify_fingerprint"]
        for index, name in enumerate(checks):
            with self.subTest(gate=name):
                for mock in self.calls.values():
                    mock.reset_mock()
                self.calls[name].return_value = {"accepted": False, "reason": "test rejection"}
                result = pipeline.run_pipeline("user_01")
                self.assertEqual(result["rejected_stage"], name.removeprefix("verify_"))
                self.assertFalse(result["biometrics_accepted"])
                for remaining in checks[index + 1:] + ["recognize_command", "execute_command"]:
                    self.calls[remaining].assert_not_called()
                if index == 0:
                    self.calls["record_audio"].assert_not_called()
                if index < 3:
                    self.calls["capture_fingerprint"].assert_not_called()
                self.calls[name].return_value = {"accepted": True, "reason": "test acceptance"}

    def test_unknown_command_stops_action(self):
        self.calls["recognize_command"].return_value = {"accepted": False, "recognized_command": "UNKNOWN", "reason": "distance"}
        self.assertEqual(pipeline.run_pipeline("user_01")["rejected_stage"], "command")
        self.calls["execute_command"].assert_not_called()

    def test_unverified_unlock_prediction_cannot_execute(self):
        self.calls["verify_face"].return_value = {"accepted": False, "reason": "wrong face"}
        pipeline.run_pipeline("user_01")
        self.calls["recognize_command"].assert_not_called()
        self.calls["execute_command"].assert_not_called()

    def test_invalid_biometric_stops_action(self):
        self.calls["extract_voice_features"].side_effect = ValueError("silent")
        self.assertEqual(pipeline.run_pipeline("user_01")["rejected_stage"], "voice")
        self.calls["execute_command"].assert_not_called()

    def test_missing_real_model_rejects_pipeline(self):
        from verification import verify_face
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            self.calls["verify_face"].side_effect = lambda x, user, _: verify_face(x, user, folder)
            self.assertFalse(pipeline.run_pipeline("user_01")["authorized"])
        self.calls["execute_command"].assert_not_called()

    def test_same_voice_vector_and_one_recording(self):
        pipeline.run_pipeline("user_01")
        self.calls["record_audio"].assert_called_once()
        self.calls["extract_voice_features"].assert_called_once()
        vector = self.calls["verify_voice"].call_args.args[0]
        self.assertIs(self.calls["recognize_command"].call_args.args[0], vector)
        self.assertIs(self.calls["create_fusion_features"].call_args.args[1], vector)

    def test_offline_trials_never_execute(self):
        result = pipeline.run_pipeline("user_01", samples={"face": "held-out", "voice": "held-out", "fingerprint": "held-out"})
        self.assertTrue(result["authorized"])
        self.assertTrue(result["offline"])
        self.calls["execute_command"].assert_not_called()
        self.calls["capture_image"].assert_not_called()
        self.calls["record_audio"].assert_not_called()

    def test_unknown_claim_rejects_before_capture(self):
        self.assertFalse(pipeline.run_pipeline("intruder")["authorized"])
        self.calls["capture_image"].assert_not_called()
        self.calls["execute_command"].assert_not_called()
