"""Frozen-model TEST metrics and the actual matched-session hierarchical pipeline."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import config
from dataset import require_valid_dataset, dataset_signature, sample_path
from features import create_fusion_features
from pipeline import run_pipeline, print_result
from train_models import extract_tables, classification_scores, binary_counts
from verification import load_artifact, decide_normalized


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def evaluate_all(manifest_path=config.MANIFEST_PATH, model_dir=config.MODEL_DIR,
                 results_dir=config.RESULTS_DIR, tables=None):
    rows = require_valid_dataset(manifest_path)
    data_dir = Path(manifest_path).parent
    artifacts = {task: load_artifact(task, model_dir) for task in config.TASKS}
    signature = dataset_signature(rows, data_dir)
    if any(a["dataset_signature"] != signature for a in artifacts.values()):
        raise ValueError("Dataset changed since training; results would not describe the saved experiment")
    tables = extract_tables(rows, data_dir) if tables is None else tables
    fusion_scalers = artifacts["fusion"]["scalers"]
    tables["fusion"] = {**tables["face"], "X": np.array([
        create_fusion_features(f, v, fusion_scalers["face"], fusion_scalers["voice"])
        for f, v in zip(tables["face"]["X"], tables["voice"]["X"])])}
    destination = Path(results_dir)
    matrices = destination / "confusion_matrices"
    matrices.mkdir(parents=True, exist_ok=True)
    classification, verification = [], []
    for task in config.TASKS:
        artifact, table = artifacts[task], tables[task]
        mask = table["split"] == "TEST"
        X, y = table["X"][mask], table["y"][mask]
        normalized = X if task == "fusion" else artifact["scalers"][task].transform(X)
        labels = sorted(artifact["templates"])
        for name, model in artifact["models"].items():
            predicted = model.predict(normalized)
            classification.append({"task": task, "model": name, "split": "TEST", "samples": len(y),
                                   **classification_scores(y, predicted), "selected": name == artifact["selected"]})
            matrix = confusion_matrix(y, predicted, labels=labels)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, colorbar=False, xticks_rotation=30)
            ax.set_title(f"{name} — TEST (n={len(y)})")
            fig.tight_layout()
            fig.savefig(matrices / f"{name}.png", dpi=160)
            plt.close(fig)
        if task in config.IDENTITY_TASKS:
            genuine, accepted = [], []
            for vector, user in zip(normalized, y):
                for claim in sorted(config.USER_ROLES):
                    genuine.append(user == claim)
                    accepted.append(decide_normalized(vector, claim, artifact)["accepted"])
            verification.append({"task": task, "model": artifact["selected"], "split": "TEST",
                                 **binary_counts(genuine, accepted)})

    genuine, accepted, pipeline_trials = [], [], []
    for row in rows:
        if row["split"] != "TEST":
            continue
        samples = {modality: sample_path(row, modality, data_dir) for modality in ("face", "voice", "fingerprint")}
        for claim in sorted(config.USER_ROLES):
            result = run_pipeline(claim, samples=samples, model_dir=model_dir)
            genuine.append(claim == row["user_id"])
            accepted.append(result["biometrics_accepted"])
            pipeline_trials.append({"sample_user": row["user_id"], "session_id": row["session_id"],
                                    "spoken_command": row["command"], "claimed_user": claim,
                                    "biometrics_accepted": result["biometrics_accepted"],
                                    "recognized_command": result["recognized_command"],
                                    "authorized": result["authorized"],
                                    "correct_command": result["recognized_command"] == row["command"],
                                    "rejected_stage": result["rejected_stage"], "reason": result["reason"]})
    verification.append({"task": "hierarchical", "model": "selected four identity verifiers", "split": "TEST",
                         **binary_counts(genuine, accepted)})
    for record in verification:
        matrix = np.array([[record["TN"], record["FP"]], [record["FN"], record["TP"]]])
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matrix).plot(ax=ax, colorbar=False)
        ax.set_xticks([0, 1], ["Rejected", "Accepted"])
        ax.set_yticks([0, 1], ["Impostor", "Genuine"])
        ax.set_title(f"{record['task']} verification — TEST")
        fig.tight_layout()
        fig.savefig(matrices / f"{record['task']}_verification.png", dpi=160)
        plt.close(fig)
    write_csv(destination / "classification_metrics.csv", classification)
    write_csv(destination / "verification_metrics.csv", verification)
    write_csv(destination / "pipeline_trials.csv", pipeline_trials)
    setup = {"split": "TEST", "dataset_signature": signature,
             "users": sorted(config.USER_ROLES), "test_sessions_per_user": [9, 10],
             "identity_claims": "one genuine plus every other enrolled user per sample",
             "hierarchical_trials": len(pipeline_trials),
             "fingerprint_reuse": "One genuine impression per user/session is reused across its four command trials and all claims; these are not independent fingerprint captures.",
             "hierarchical_acceptance": "All four biometric gates; command and permissions recorded separately to avoid counting authorized denials as biometric FRR.",
             "zero_denominator": "FAR/FRR is blank in CSV (null in JSON) if denominator is zero.",
             "unknown_command_limit": "No real out-of-vocabulary recordings provided; wrong-known-label validation trials are only a rejection proxy.",
             "actions_executed": False}
    (destination / "test_setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    print("\nTEST classification:")
    for record in classification:
        print(f"{record['model']:18s} accuracy={record['accuracy']:.3f} macro F1={record['f1_macro']:.3f}")
    print("\nTEST verification:")
    for record in verification:
        print(f"{record['task']:14s} trials={record['trials']} FAR={record['FAR']} FRR={record['FRR']}")
    print(f"Results saved to {destination}")
    return {"classification": classification, "verification": verification, "setup": setup}


def evaluate_trial(user_id, session_id, command, claimed_user_id):
    """An explicit held-out sample dry run, even when hardware is unavailable."""
    rows = require_valid_dataset()
    row = next((r for r in rows if r["user_id"] == user_id and r["session_id"] == session_id
                and r["command"] == command and r["split"] == "TEST"), None)
    if row is None:
        raise ValueError("Select a real TEST row (session 09 or 10)")
    samples = {modality: sample_path(row, modality) for modality in ("face", "voice", "fingerprint")}
    result = run_pipeline(claimed_user_id, samples=samples)
    print_result(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-user", choices=list(config.USER_ROLES))
    parser.add_argument("--session", type=int, choices=(9, 10), default=9)
    parser.add_argument("--command", choices=config.COMMANDS, default="STATUS")
    parser.add_argument("--claim", choices=list(config.USER_ROLES))
    args = parser.parse_args()
    try:
        if args.trial_user:
            evaluate_trial(args.trial_user, f"session_{args.session:02d}", args.command, args.claim or args.trial_user)
        else:
            evaluate_all()
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation stopped: {exc}\n")
