"""CLI demonstrations. Only the full live pipeline can execute simulated commands."""
import config
from collect_data import collect_session, capture_image, record_audio, capture_fingerprint
from dataset import validate_dataset, print_report
from features import (extract_face_features, extract_voice_features,
                      create_fusion_features, extract_fingerprint_features)
from pipeline import run_pipeline, print_result
from verification import verify_face, verify_voice, verify_fusion, verify_fingerprint

MENU = """
1. Collect Dataset
2. Validate Dataset
3. Train Models
4. Evaluate Models
5. Face-Only Verification
6. Voice-Only Verification
7. Face + Voice Fusion Verification
8. Fingerprint-Only Verification
9. Full Hierarchical Live Demo
10. Exit
"""


def claimed_user():
    print("Users:", ", ".join(config.USER_ROLES))
    user = input("Claimed user ID: ").strip()
    if user not in config.USER_ROLES:
        raise ValueError("Unknown user ID")
    return user


def new_fingerprint_path():
    return input("NEW ink impression image path (Enter for webcam): ").strip().strip('"') or None


def main():
    while True:
        print(MENU)
        try:
            option = input("Choose: ").strip()
            if option == "10":
                return
            if option == "1":
                user = claimed_user()
                session = int(input("Session number (1-10): "))
                collect_session(user, f"session_{session:02d}", new_fingerprint_path())
            elif option == "2":
                print_report(validate_dataset())
            elif option == "3":
                from train_models import train_all
                train_all()
            elif option == "4":
                from evaluate import evaluate_all
                evaluate_all()
            elif option in {"5", "6", "7", "8", "9"}:
                user = claimed_user()
                if option == "9":
                    print_result(run_pipeline(user))
                elif option == "5":
                    print(verify_face(extract_face_features(capture_image("face")), user))
                elif option == "6":
                    print(verify_voice(extract_voice_features(record_audio()), user))
                elif option == "7":
                    face = extract_face_features(capture_image("face"))
                    voice = extract_voice_features(record_audio())
                    print(verify_fusion(create_fusion_features(face, voice), user))
                else:
                    print(verify_fingerprint(extract_fingerprint_features(capture_fingerprint(new_fingerprint_path())), user))
                if option != "9":
                    print("Individual verification only; no command or action is run.")
            else:
                print("Choose 1 through 10.")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            return
        except Exception as exc:
            print(f"Operation rejected/stopped: {exc}")


if __name__ == "__main__":
    main()
