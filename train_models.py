"""Fit exactly KNN(k=3) and linear SVM per task, using TRAIN only."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from dataset import require_valid_dataset, sample_path, dataset_signature
from features import (extract_face_features, extract_voice_features,
                      extract_fingerprint_features, feature_settings)


def extract_tables(rows, data_dir=config.DATA_DIR):
    """One fingerprint row per session; each voice file is extracted only once."""
    face, voice, fingerprints, fingerprint_rows = [], [], [], []
    seen_fingerprints = set()
    for index, row in enumerate(rows, 1):
        try:
            face.append(extract_face_features(sample_path(row, "face", data_dir)))
            voice.append(extract_voice_features(sample_path(row, "voice", data_dir)))
            key = (row["user_id"], row["session_id"])
            if key not in seen_fingerprints:
                fingerprints.append(extract_fingerprint_features(sample_path(row, "fingerprint", data_dir)))
                fingerprint_rows.append(row)
                seen_fingerprints.add(key)
        except (ValueError, OSError, RuntimeError) as exc:
            raise ValueError(f"Invalid sample {row['user_id']}/{row['session_id']}/{row['command']}: {exc}") from exc
        if index % 20 == 0:
            print(f"Extracted {index}/{len(rows)} paired samples")
    def table(values, records, label="user_id"):
        return {"X": np.asarray(values), "y": np.array([r[label] for r in records]),
                "split": np.array([r["split"] for r in records]), "rows": records}
    return {"face": table(face, rows), "voice": table(voice, rows),
            "command": table(voice, rows, "command"),
            "fingerprint": table(fingerprints, fingerprint_rows)}


def classification_scores(expected, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(expected, predicted)),
            "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1)}


def binary_counts(genuine, accepted):
    genuine, accepted = np.asarray(genuine, dtype=bool), np.asarray(accepted, dtype=bool)
    tp = int(np.sum(genuine & accepted))
    fn = int(np.sum(genuine & ~accepted))
    fp = int(np.sum(~genuine & accepted))
    tn = int(np.sum(~genuine & ~accepted))
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "accuracy": (tp + tn) / len(genuine) if len(genuine) else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "FAR": fp / (fp + tn) if fp + tn else None,
            "FRR": fn / (fn + tp) if fn + tp else None,
            "trials": len(genuine)}


def calibrate_threshold(X, y, predicted, templates):
    """Validation genuine + every wrong-label claim; minimize (FAR+FRR)/2.

    Ties prefer lower FAR, then the smaller threshold. Prediction must agree
    with the claim when computing BOTH rates, exactly as at inference.
    """
    distances, genuine, gates = [], [], []
    for vector, actual, prediction in zip(X, y, predicted):
        for claim, template in templates.items():
            distances.append(float(np.linalg.norm(vector - template)))
            genuine.append(actual == claim)
            gates.append(prediction == claim)
    distances, gates = np.asarray(distances), np.asarray(gates)
    if not distances.size or not np.isfinite(distances).all():
        raise ValueError("Cannot calibrate without valid validation trials")
    best = None
    for threshold in np.unique(np.append(distances, 0.0)):
        counts = binary_counts(genuine, gates & (distances <= threshold))
        key = ((counts["FAR"] + counts["FRR"]) / 2, counts["FAR"], float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), counts)
    return best[1], best[2]


def fit_task(task, table, normalized, scalers, signature):
    train = table["split"] == "TRAIN"
    validation = table["split"] == "VALIDATION"
    X_train, y_train = normalized[train], table["y"][train]
    X_val, y_val = normalized[validation], table["y"][validation]
    models = {f"{task}_knn": KNeighborsClassifier(n_neighbors=3),
              f"{task}_svm": SVC(kernel="linear")}
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = classification_scores(y_val, model.predict(X_val))
    # Stable insertion order makes KNN win an exact tie.
    selected = max(models, key=lambda name: (scores[name]["f1_macro"], scores[name]["accuracy"]))
    templates = {str(label): X_train[y_train == label].mean(axis=0) for label in np.unique(y_train)}
    threshold, calibration = calibrate_threshold(X_val, y_val, models[selected].predict(X_val), templates)
    print(f"{task}: selected {selected}; validation macro F1={scores[selected]['f1_macro']:.4f}; threshold={threshold:.4f}")
    return {"task": task, "models": models, "selected": selected, "scalers": scalers,
            "templates": templates, "threshold": threshold, "dimension": normalized.shape[1],
            "label_mapping": {label: i for i, label in enumerate(sorted(templates))},
            "validation_scores": scores, "calibration": calibration,
            "threshold_rule": "minimize validation (FAR+FRR)/2, then FAR, then threshold",
            "feature_settings": feature_settings(), "users": sorted(config.USER_ROLES),
            "dataset_signature": signature}


def train_all(manifest_path=config.MANIFEST_PATH, model_dir=config.MODEL_DIR,
              results_dir=config.RESULTS_DIR):
    rows = require_valid_dataset(manifest_path)
    data_dir = Path(manifest_path).parent
    tables = extract_tables(rows, data_dir)
    signature = dataset_signature(rows, data_dir)
    artifacts, normalized, scalers = {}, {}, {}
    for task in ("face", "voice", "fingerprint", "command"):
        table = tables[task]
        scaler = StandardScaler().fit(table["X"][table["split"] == "TRAIN"])
        scalers[task] = scaler
        normalized[task] = scaler.transform(table["X"])
        artifacts[task] = fit_task(task, table, normalized[task], {task: scaler}, signature)
    # Actual EARLY fusion: separate training scalers, then concatenate features.
    normalized["fusion"] = np.concatenate((normalized["face"], normalized["voice"]), axis=1)
    tables["fusion"] = {**tables["face"], "X": normalized["fusion"]}
    artifacts["fusion"] = fit_task("fusion", tables["fusion"], normalized["fusion"],
                                    {"face": scalers["face"], "voice": scalers["voice"]}, signature)
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for task, artifact in artifacts.items():
        temporary = destination / f"{task}.joblib.tmp"
        joblib.dump(artifact, temporary)
        temporary.replace(destination / f"{task}.joblib")
    summary = {task: {key: a[key] for key in ("selected", "threshold", "validation_scores", "calibration", "threshold_rule")}
               for task, a in artifacts.items()}
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved all ten fitted models in five task bundles. Evaluating frozen models on TEST.")
    from evaluate import evaluate_all
    evaluate_all(manifest_path, model_dir, results_dir, tables=tables)
    return summary


if __name__ == "__main__":
    try:
        train_all()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"Training stopped: {exc}")
