"""Actual camera/microphone capture; no generated or substituted biometric samples."""
import argparse
import importlib
from pathlib import Path

import config
from dataset import file_hash, rebuild_manifest


def check_hardware():
    """Probe availability without opening preview windows or recording audio."""
    status = {}
    for name in ("numpy", "cv2", "skimage", "librosa", "sounddevice", "soundfile", "sklearn", "joblib", "matplotlib"):
        try:
            importlib.import_module(name)
            status[name] = "available"
        except (ImportError, OSError) as exc:
            status[name] = f"unavailable: {exc}"
    if status["cv2"] == "available":
        import cv2
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        try:
            status["camera"] = "available (not visually tested)" if camera.isOpened() and camera.read()[0] else "unavailable"
        finally:
            camera.release()
    else:
        status["camera"] = "unavailable: OpenCV missing"
    if status["sounddevice"] == "available":
        import sounddevice as sd
        try:
            inputs = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
            sd.check_input_settings(channels=1, samplerate=config.SAMPLE_RATE)
            status["microphone"] = f"input devices: {inputs}; not recording-tested"
        except (sd.PortAudioError, ValueError, OSError) as exc:
            status["microphone"] = f"unavailable: {exc}"
    else:
        status["microphone"] = "unavailable: sounddevice missing"
    return status


def capture_image(modality="face"):
    import cv2
    from features import crop_bounds
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Webcam unavailable. Connect a camera and check CAMERA_INDEX in config.py.")
    title = f"{modality}: align in guide; SPACE capture; ESC cancel"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Webcam returned no image")
            x1, y1, x2, y2 = crop_bounds(frame.shape, modality)
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, "SPACE capture / ESC cancel", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow(title, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                raise RuntimeError("Capture cancelled")
            if key == 32:
                return frame  # features.py applies the identical guide crop offline and live
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("Capture window closed")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def record_audio():
    import sounddevice as sd
    print(f"Speak ONE command now ({config.RECORD_SECONDS:g} seconds).")
    try:
        audio = sd.rec(int(config.RECORD_SECONDS * config.SAMPLE_RATE),
                       samplerate=config.SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        return audio[:, 0]
    except (sd.PortAudioError, ValueError, OSError) as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc


def ensure_new_fingerprint(path, data_dir=config.DATA_DIR):
    """Reject any enrolled path or byte-identical enrolled file for a live import."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("New fingerprint image does not exist")
    raw = (Path(data_dir) / "raw").resolve()
    if path.is_relative_to(raw):
        raise ValueError("Use a newly acquired fingerprint outside data/raw; enrollment files are forbidden")
    digest = file_hash(path)
    for enrolled in raw.rglob("*"):
        if enrolled.is_file() and file_hash(enrolled) == digest:
            raise ValueError("This file duplicates an enrolled sample")
    return path


def capture_fingerprint(new_image=None, prompt=False):
    from features import read_image
    if prompt and new_image is None:
        new_image = input("NEW ink impression image path (Enter for webcam): ").strip().strip('"') or None
    if new_image:
        return read_image(ensure_new_fingerprint(new_image))
    print("Show a NEW ink impression to the camera, not a photograph of a fingertip.")
    return capture_image("fingerprint")


def collect_session(user_id, session_id, fingerprint_image=None):
    """Collect four matched pairs plus one independently acquired ink impression."""
    import cv2
    import soundfile as sf
    from features import extract_face_features, extract_voice_features, extract_fingerprint_features
    if user_id not in config.USER_ROLES:
        raise ValueError("Unknown user")
    config.session_split(session_id)
    folder = config.DATA_DIR / "raw" / user_id / session_id
    folder.mkdir(parents=True, exist_ok=True)
    print("Keep the SAME participant for all samples in this session. Existing files are preserved.")
    try:
        for command in config.COMMANDS:
            face_path, voice_path = folder / f"{command}_face.jpg", folder / f"{command}_voice.wav"
            if face_path.exists() and voice_path.exists():
                print(f"Keeping existing {command} pair")
                continue
            if face_path.exists() or voice_path.exists():
                raise ValueError(f"Incomplete {command} pair: inspect and remove that incomplete pair before retrying")
            input(f"{user_id} / {session_id}: prepare to say {command}; press Enter.")
            face = capture_image("face")
            audio = record_audio()
            extract_face_features(face)
            extract_voice_features(audio)
            if not cv2.imwrite(str(face_path), face):
                raise OSError(f"Could not save {face_path}")
            sf.write(str(voice_path), audio, config.SAMPLE_RATE, subtype="PCM_16")
        fingerprint_path = folder / "fingerprint.jpg"
        if not fingerprint_path.exists():
            print("Use a separately acquired ink impression for EACH session.")
            fingerprint = capture_fingerprint(fingerprint_image)
            extract_fingerprint_features(fingerprint)
            # Compare the saved encoding too: prevents importing a prior saved impression.
            ok, encoded = cv2.imencode(".jpg", fingerprint)
            if not ok:
                raise OSError("Could not encode fingerprint")
            import hashlib
            digest = hashlib.sha256(encoded.tobytes()).hexdigest()
            if any(file_hash(p) == digest for p in (config.DATA_DIR / "raw").rglob("fingerprint.jpg")):
                raise ValueError("Duplicate fingerprint impression; acquire a new one")
            fingerprint_path.write_bytes(encoded.tobytes())
    finally:
        rebuild_manifest()
    print("Session saved. Run dataset.py to check counts and integrity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-hardware", action="store_true")
    parser.add_argument("--user", choices=list(config.USER_ROLES))
    parser.add_argument("--session", type=int, choices=range(1, 11))
    parser.add_argument("--fingerprint", type=Path, help="New ink impression image outside data/raw")
    args = parser.parse_args()
    if args.check_hardware:
        for item, value in check_hardware().items():
            print(f"{item}: {value}")
    elif args.user and args.session:
        try:
            collect_session(args.user, f"session_{args.session:02d}", args.fingerprint)
        except (RuntimeError, ValueError, OSError) as exc:
            parser.exit(1, f"Collection stopped: {exc}\n")
    else:
        parser.error("Supply --user and --session, or --check-hardware")
