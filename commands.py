"""Recognize the spoken word, independently of speaker identity. Executes nothing."""
import numpy as np

import config
from features import valid_vector
from verification import load_artifact


def recognize_command(voice_features, model_dir=config.MODEL_DIR):
    try:
        artifact = load_artifact("command", model_dir)
        vector = valid_vector(voice_features, artifact["dimension"])
        normalized = artifact["scalers"]["command"].transform([vector])[0]
        predicted = str(artifact["models"][artifact["selected"]].predict([normalized])[0])
        if predicted not in config.COMMANDS:
            raise ValueError("Unrecognized command label")
        distance = float(np.linalg.norm(normalized - artifact["templates"][predicted]))
        accepted = np.isfinite(distance) and distance <= artifact["threshold"]
        return {"recognized_command": predicted if accepted else "UNKNOWN", "accepted": bool(accepted),
                "distance": distance, "threshold": float(artifact["threshold"]),
                "reason": "Accepted" if accepted else "Command distance exceeds validation threshold"}
    except Exception as exc:
        return {"recognized_command": "UNKNOWN", "accepted": False,
                "distance": None, "threshold": None, "reason": f"Invalid/unavailable command recognition: {exc}"}
