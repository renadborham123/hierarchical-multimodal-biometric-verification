"""Small, editable project settings. Paths are independent of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Remove one entry for a three-person team. Collect all sessions for each entry.
USER_ROLES = {
    "user_01": "Admin",
    "user_02": "Employee",
    "user_03": "Employee",
    "user_04": "Guest",
}
COMMANDS = ("STATUS", "REPORT", "UNLOCK", "LOGOUT")
ROLE_PERMISSIONS = {
    "Admin": set(COMMANDS),
    "Employee": {"STATUS", "REPORT", "LOGOUT"},
    "Guest": {"STATUS", "LOGOUT"},
}
MANIFEST_FIELDS = (
    "user_id", "session_id", "command", "trial_id",
    "face_path", "voice_path", "fingerprint_path", "split",
)
TASKS = ("face", "voice", "fusion", "fingerprint", "command")
IDENTITY_TASKS = TASKS[:-1]
SAMPLE_RATE = 16000
RECORD_SECONDS = 2.0
SILENCE_RMS = 0.005
CAMERA_INDEX = 0
FACE_CROP_FRACTION = 0.60
FINGERPRINT_CROP_FRACTION = 0.65
FEATURE_VERSION = "hog-lbp-mfcc30-v1"


def session_split(session_id):
    """The session, rather than an individual image, determines its split."""
    if session_id not in {f"session_{n:02d}" for n in range(1, 11)}:
        raise ValueError("Session must be session_01 through session_10")
    number = int(session_id[-2:])
    return "TRAIN" if number <= 6 else "VALIDATION" if number <= 8 else "TEST"
