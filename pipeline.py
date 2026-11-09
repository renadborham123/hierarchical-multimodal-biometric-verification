"""One ordered pipeline. No subsequent stage runs after a failed check."""
import config
from actions import execute_command
from collect_data import capture_image, record_audio, capture_fingerprint
from commands import recognize_command
from features import (extract_face_features, extract_voice_features,
                      create_fusion_features, extract_fingerprint_features)
from permissions import get_user_role, is_command_allowed
from verification import verify_face, verify_voice, verify_fusion, verify_fingerprint


def run_pipeline(claimed_user_id, samples=None, fingerprint_image=None, model_dir=config.MODEL_DIR):
    """Live capture by default. Supplied held-out samples ALWAYS mean a dry run.

    samples = {'face': image/path, 'voice': waveform/path, 'fingerprint': image/path}
    is used by offline evaluation. It can never execute an action.
    """
    result = {"claimed_user": claimed_user_id, "verified_user": None, "decisions": {},
              "biometrics_accepted": False, "recognized_command": "UNKNOWN", "role": None,
              "permission_allowed": False, "authorized": False, "action": None,
              "rejected_stage": None, "reason": "", "offline": samples is not None}
    stage = "identity"
    try:
        if claimed_user_id not in config.USER_ROLES:
            raise ValueError("Unknown claimed identity")
        stage = "face"
        face_image = capture_image("face") if samples is None else samples["face"]
        face = extract_face_features(face_image)
        decision = verify_face(face, claimed_user_id, model_dir)
        result["decisions"][stage] = decision
        if not decision["accepted"]:
            raise ValueError(decision["reason"])

        stage = "voice"
        audio = record_audio() if samples is None else samples["voice"]
        voice = extract_voice_features(audio)  # ONCE; same object reaches command recognition
        decision = verify_voice(voice, claimed_user_id, model_dir)
        result["decisions"][stage] = decision
        if not decision["accepted"]:
            raise ValueError(decision["reason"])

        stage = "fusion"
        fusion = create_fusion_features(face, voice, model_dir=model_dir)
        decision = verify_fusion(fusion, claimed_user_id, model_dir)
        result["decisions"][stage] = decision
        if not decision["accepted"]:
            raise ValueError(decision["reason"])

        stage = "fingerprint"
        fingerprint_image = (capture_fingerprint(fingerprint_image, prompt=fingerprint_image is None)
                             if samples is None else samples["fingerprint"])
        fingerprint = extract_fingerprint_features(fingerprint_image)
        decision = verify_fingerprint(fingerprint, claimed_user_id, model_dir)
        result["decisions"][stage] = decision
        if not decision["accepted"]:
            raise ValueError(decision["reason"])
        result["biometrics_accepted"] = True
        result["verified_user"] = claimed_user_id

        stage = "command"
        command = recognize_command(voice, model_dir)
        result["decisions"][stage] = command
        result["recognized_command"] = command["recognized_command"]
        if not command["accepted"] or command["recognized_command"] not in config.COMMANDS:
            raise ValueError(command["reason"])

        stage = "permission"
        result["role"] = get_user_role(result["verified_user"])
        result["permission_allowed"] = is_command_allowed(result["verified_user"], command["recognized_command"])
        if not result["permission_allowed"]:
            raise ValueError("ACCESS DENIED: INSUFFICIENT PERMISSIONS")
        result["authorized"] = True
        if samples is None:
            stage = "action"
            result["action"] = execute_command(command["recognized_command"])
        result["reason"] = "AUTHORIZED (offline: no action executed)" if samples is not None else "AUTHORIZED"
    except Exception as exc:
        result["rejected_stage"] = stage
        result["reason"] = f"ACCESS DENIED at {stage}: {exc}"
    return result


def print_result(result):
    print(f"Claimed user: {result['claimed_user']}")
    for stage in ("face", "voice", "fusion", "fingerprint"):
        decision = result["decisions"].get(stage)
        print(f"{stage.title()} result: {decision if decision else 'NOT RUN'}")
    print(f"Recognized command: {result['recognized_command']}")
    print(f"User role: {result['role'] or 'NOT CHECKED'}")
    print(f"Permission result: {result['permission_allowed']}")
    print(f"Final action: {result['action'] or 'NONE'}")
    print(result["reason"])
