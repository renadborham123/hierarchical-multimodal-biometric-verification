"""Claimed identity + classifier agreement + validation-calibrated distance gate."""
from pathlib import Path

import joblib
import numpy as np

import config
from features import feature_settings, valid_vector


def load_artifact(task, model_dir=config.MODEL_DIR):
    """Load only locally trained artifacts. Missing or inconsistent state rejects."""
    path = Path(model_dir) / f"{task}.joblib"
    if not path.is_file():
        raise ValueError(f"Missing {task} model bundle. Collect real data and run train_models.py.")
    artifact = joblib.load(path)
    required = {"task", "models", "selected", "scalers", "templates", "threshold", "dimension",
                "feature_settings", "users", "dataset_signature", "label_mapping"}
    if not isinstance(artifact, dict) or not required <= artifact.keys():
        raise ValueError("Incomplete model bundle")
    if artifact["task"] != task or artifact["feature_settings"] != feature_settings():
        raise ValueError("Model feature settings differ; retrain before use")
    if artifact["users"] != sorted(config.USER_ROLES):
        raise ValueError("Enrolled users changed; retrain before use")
    expected_labels = set(config.COMMANDS if task == "command" else config.USER_ROLES)
    if set(artifact["templates"]) != expected_labels:
        raise ValueError("Missing label template")
    if set(artifact["label_mapping"]) != expected_labels:
        raise ValueError("Missing label mapping")
    if set(artifact["models"]) != {f"{task}_knn", f"{task}_svm"}:
        raise ValueError("Both fitted KNN and SVM are required")
    if artifact["selected"] not in artifact["models"]:
        raise ValueError("Selected model missing")
    for model in artifact["models"].values():
        if not hasattr(model, "classes_") or set(model.classes_) != expected_labels:
            raise ValueError("Model is unfitted or has inconsistent labels")
        if model.n_features_in_ != artifact["dimension"]:
            raise ValueError("Classifier feature dimension mismatch")
    threshold = artifact["threshold"]
    if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Missing or invalid rejection threshold")
    expected_scalers = {"face", "voice"} if task == "fusion" else {task}
    if set(artifact["scalers"]) != expected_scalers:
        raise ValueError("Missing normalization scaler")
    for scaler in artifact["scalers"].values():
        if (not hasattr(scaler, "mean_") or not np.isfinite(scaler.mean_).all()
                or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0)):
            raise ValueError("Missing or invalid fitted scaler")
    if sum(s.n_features_in_ for s in artifact["scalers"].values()) != artifact["dimension"]:
        raise ValueError("Scaler feature dimension mismatch")
    for template in artifact["templates"].values():
        valid_vector(template, artifact["dimension"])
    return artifact


def decide_normalized(vector, claimed_user_id, artifact):
    """Shared decision rule for live verification and held-out evaluation."""
    vector = valid_vector(vector, artifact["dimension"])
    if claimed_user_id not in artifact["templates"]:
        raise ValueError("Claimed user has no training template")
    predicted = str(artifact["models"][artifact["selected"]].predict([vector])[0])
    distance = float(np.linalg.norm(vector - artifact["templates"][claimed_user_id]))
    accepted = predicted == claimed_user_id and distance <= artifact["threshold"]
    reason = "Accepted" if accepted else ("Predicted identity differs from claimed identity"
              if predicted != claimed_user_id else "Distance exceeds validation threshold")
    return {"accepted": bool(accepted), "predicted_user": predicted, "claimed_user": claimed_user_id,
            "distance": distance, "threshold": float(artifact["threshold"]), "reason": reason}


def verify(task, biometric_features, claimed_user_id, model_dir=config.MODEL_DIR):
    try:
        if claimed_user_id not in config.USER_ROLES:
            raise ValueError("Unknown claimed user")
        artifact = load_artifact(task, model_dir)
        vector = valid_vector(biometric_features, artifact["dimension"])
        if task != "fusion":
            vector = artifact["scalers"][task].transform([vector])[0]
        # Fusion has already been independently scaled before concatenation.
        return decide_normalized(vector, claimed_user_id, artifact)
    except Exception as exc:
        # A failed load, transform, or prediction must never grant access.
        return {"accepted": False, "predicted_user": None, "claimed_user": claimed_user_id,
                "distance": None, "threshold": None, "reason": f"Verification unavailable/invalid: {exc}"}


def verify_face(face_features, claimed_user_id, model_dir=config.MODEL_DIR):
    return verify("face", face_features, claimed_user_id, model_dir)


def verify_voice(voice_features, claimed_user_id, model_dir=config.MODEL_DIR):
    return verify("voice", voice_features, claimed_user_id, model_dir)


def verify_fusion(fusion_features, claimed_user_id, model_dir=config.MODEL_DIR):
    return verify("fusion", fusion_features, claimed_user_id, model_dir)


def verify_fingerprint(fingerprint_features, claimed_user_id, model_dir=config.MODEL_DIR):
    return verify("fingerprint", fingerprint_features, claimed_user_id, model_dir)
