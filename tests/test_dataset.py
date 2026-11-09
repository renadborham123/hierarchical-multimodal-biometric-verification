"""Manifest tests use temporary marker bytes, not a generated biometric dataset."""
import csv
import tempfile
import unittest
from pathlib import Path

import config
from collect_data import ensure_new_fingerprint
from dataset import validate_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.csv"
        self.row = {"user_id": "user_01", "session_id": "session_01", "command": "STATUS", "trial_id": "01",
                    "face_path": "raw/user_01/session_01/STATUS_face.jpg",
                    "voice_path": "raw/user_01/session_01/STATUS_voice.wav",
                    "fingerprint_path": "raw/user_01/session_01/fingerprint.jpg", "split": "TRAIN"}
        for modality in ("face", "voice", "fingerprint"):
            path = self.root / self.row[f"{modality}_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"integrity-test-{modality}".encode())

    def report(self, rows, complete=False):
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=config.MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return validate_dataset(self.manifest, require_complete=complete)

    def test_partial_integrity_valid_and_counts(self):
        report = self.report([self.row])
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["counts"]["pairs_per_user"]["user_01"], 1)

    def test_incomplete_data_cannot_train(self):
        self.assertFalse(self.report([self.row], complete=True)["valid"])

    def test_face_voice_identity_mismatch(self):
        self.row["voice_path"] = "raw/user_02/session_01/STATUS_voice.wav"
        self.assertTrue(any("mismatch" in error for error in self.report([self.row])["errors"]))

    def test_session_in_train_and_test_rejected(self):
        other = {**self.row, "split": "TEST"}
        errors = self.report([self.row, other])["errors"]
        self.assertTrue(any("crosses splits" in error for error in errors))

    def test_session_policy(self):
        self.assertEqual([config.session_split(f"session_{n:02d}") for n in (1, 6, 7, 8, 9, 10)],
                         ["TRAIN", "TRAIN", "VALIDATION", "VALIDATION", "TEST", "TEST"])
        with self.assertRaises(ValueError):
            config.session_split("session_11")

    def test_missing_file(self):
        (self.root / self.row["face_path"]).unlink()
        self.assertTrue(any("missing face" in error for error in self.report([self.row])["errors"]))

    def test_path_escape(self):
        self.row["face_path"] = "../private.jpg"
        self.assertFalse(self.report([self.row])["valid"])

    def test_duplicate_content_across_sessions(self):
        other = {**self.row, "session_id": "session_02", "trial_id": "02"}
        for modality in ("face", "voice", "fingerprint"):
            other[f"{modality}_path"] = self.row[f"{modality}_path"].replace("session_01", "session_02")
            path = self.root / other[f"{modality}_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.root / self.row[f"{modality}_path"]).read_bytes())
        self.assertTrue(any("duplicate file content" in e for e in self.report([self.row, other])["errors"]))

    def test_one_fingerprint_reference_per_command_is_allowed(self):
        other = {**self.row, "command": "REPORT"}
        for modality in ("face", "voice"):
            other[f"{modality}_path"] = self.row[f"{modality}_path"].replace("STATUS", "REPORT")
            (self.root / other[f"{modality}_path"]).write_bytes(f"other-{modality}".encode())
        report = self.report([self.row, other])
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["counts"]["fingerprints_per_user"]["user_01"], 1)

    def test_live_import_rejects_enrollment_file_and_copy(self):
        enrolled = self.root / self.row["fingerprint_path"]
        with self.assertRaises(ValueError):
            ensure_new_fingerprint(enrolled, self.root)
        copied = self.root / "copied.jpg"
        copied.write_bytes(enrolled.read_bytes())
        with self.assertRaises(ValueError):
            ensure_new_fingerprint(copied, self.root)

    def test_empty_manifest_rejected(self):
        self.assertFalse(self.report([])["valid"])
