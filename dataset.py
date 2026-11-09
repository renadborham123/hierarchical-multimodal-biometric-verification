"""CSV manifest, session splits, and checks before any training or evaluation."""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import config


def load_manifest(manifest_path=config.MANIFEST_PATH):
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(f"Manifest missing: {path}. Collect data first.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != config.MANIFEST_FIELDS:
            raise ValueError("Manifest must have exactly these columns: " + ", ".join(config.MANIFEST_FIELDS))
        return list(reader)


def sample_path(row, modality, data_dir=config.DATA_DIR):
    """Resolve a CSV-relative path and forbid paths outside data/raw."""
    root = Path(data_dir).resolve()
    path = (root / row[f"{modality}_path"]).resolve()
    if not path.is_relative_to(root / "raw"):
        raise ValueError(f"{modality} path must stay inside data/raw")
    return path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rebuild_manifest(data_dir=config.DATA_DIR):
    """Index existing sessions, including unfinished ones so missing samples show up."""
    root = Path(data_dir)
    rows = []
    for user in config.USER_ROLES:
        for number in range(1, 11):
            session = f"session_{number:02d}"
            folder = root / "raw" / user / session
            if not folder.is_dir() or not any(folder.iterdir()):
                continue
            prefix = f"raw/{user}/{session}"
            for command in config.COMMANDS:
                rows.append(dict(zip(config.MANIFEST_FIELDS, (
                    user, session, command, f"{number:02d}",
                    f"{prefix}/{command}_face.jpg", f"{prefix}/{command}_voice.wav",
                    f"{prefix}/fingerprint.jpg", config.session_split(session),
                ))))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=config.MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_dataset(manifest_path=config.MANIFEST_PATH, data_dir=None, require_complete=True):
    """Return errors and counts. Training requires the complete configured dataset."""
    root = Path(data_dir) if data_dir is not None else Path(manifest_path).parent
    errors = []
    try:
        rows = load_manifest(manifest_path)
    except (ValueError, OSError) as exc:
        return {"valid": False, "errors": [str(exc)], "counts": {}, "rows": []}
    if len(config.USER_ROLES) not in (3, 4) or any(
        role not in config.ROLE_PERMISSIONS for role in config.USER_ROLES.values()
    ):
        errors.append("Configure three or four users with valid roles")
    if not rows:
        errors.append("No biometric samples collected")
    seen_keys = set()
    sessions = defaultdict(set)
    paths_seen = {}
    hashes_seen = {}
    checked_paths = set()
    fingerprints = defaultdict(set)
    counts = Counter()
    command_counts = Counter()
    split_counts = Counter()
    for index, row in enumerate(rows, 2):
        tag = f"CSV line {index}"
        if any(not row.get(field) for field in config.MANIFEST_FIELDS) or None in row:
            errors.append(f"{tag}: missing/extra field")
            continue
        user, session, command = row["user_id"], row["session_id"], row["command"]
        if user not in config.USER_ROLES or command not in config.COMMANDS:
            errors.append(f"{tag}: unknown user or command")
        try:
            if row["split"] != config.session_split(session):
                errors.append(f"{tag}: wrong session split (possible split leakage)")
            if row["trial_id"] != session[-2:]:
                errors.append(f"{tag}: trial_id must match the session number")
        except ValueError as exc:
            errors.append(f"{tag}: {exc}")
        key = (user, session, command)
        if key in seen_keys:
            errors.append(f"{tag}: duplicate user/session/command")
        seen_keys.add(key)
        sessions[(user, session)].add(row["split"])
        counts[user] += 1
        command_counts[(user, command)] += 1
        split_counts[row["split"]] += 1
        for modality, suffix in (("face", f"{command}_face.jpg"),
                                 ("voice", f"{command}_voice.wav"),
                                 ("fingerprint", "fingerprint.jpg")):
            expected = Path("raw") / user / session / suffix
            if Path(row[f"{modality}_path"]).as_posix() != expected.as_posix():
                errors.append(f"{tag}: {modality} identity/session/command path mismatch")
            try:
                path = sample_path(row, modality, root)
                if not path.is_file():
                    errors.append(f"{tag}: missing {modality}: {path}")
                    continue
                if modality == "fingerprint":
                    fingerprints[user].add(str(path))
                owner = (user, session, "" if modality == "fingerprint" else command, row["split"])
                if path in paths_seen and paths_seen[path] != owner:
                    errors.append(f"{tag}: shared file across samples/splits: {path}")
                paths_seen[path] = owner
                if path in checked_paths:
                    continue  # one fingerprint intentionally referenced by four command rows
                checked_paths.add(path)
                digest = file_hash(path)
                if digest in hashes_seen:
                    errors.append(f"{tag}: duplicate file content: {path} and {hashes_seen[digest]}")
                hashes_seen[digest] = path
            except (OSError, ValueError) as exc:
                errors.append(f"{tag}: {exc}")
    for key, splits in sessions.items():
        if len(splits) > 1:
            errors.append(f"Session {key} crosses splits: {sorted(splits)}")
    if require_complete:
        expected_keys = {(u, f"session_{n:02d}", c)
                         for u in config.USER_ROLES for n in range(1, 11) for c in config.COMMANDS}
        missing = expected_keys - seen_keys
        if missing:
            errors.append(f"Missing {len(missing)} face/audio pairs; need 40 per configured user")
        if seen_keys - expected_keys:
            errors.append("Unexpected user/session/command rows")
    return {
        "valid": not errors, "errors": errors, "rows": rows,
        "counts": {"pairs_per_user": {u: counts[u] for u in config.USER_ROLES},
                   "commands_per_user": {f"{u}/{c}": command_counts[(u, c)]
                                         for u in config.USER_ROLES for c in config.COMMANDS},
                   "fingerprints_per_user": {u: len(fingerprints[u]) for u in config.USER_ROLES},
                   "pairs_per_split": {s: split_counts[s] for s in ("TRAIN", "VALIDATION", "TEST")}},
    }


def require_valid_dataset(manifest_path=config.MANIFEST_PATH):
    report = validate_dataset(manifest_path)
    if not report["valid"]:
        raise ValueError("Dataset rejected:\n" + "\n".join(report["errors"][:25]))
    return report["rows"]


def dataset_signature(rows, data_dir=config.DATA_DIR):
    """Detect accidental data edits after training, before reporting test metrics."""
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (r["user_id"], r["session_id"], r["command"])):
        digest.update(str(tuple(row[field] for field in config.MANIFEST_FIELDS)).encode())
    paths = {sample_path(row, modality, data_dir)
             for row in rows for modality in ("face", "voice", "fingerprint")}
    for path in sorted(paths):
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def print_report(report):
    print("Dataset VALID" if report["valid"] else "Dataset INVALID / INCOMPLETE")
    for name, counts in report["counts"].items():
        print(f"{name}: {counts}")
    for error in report["errors"][:25]:
        print(" -", error)
    if len(report["errors"]) > 25:
        print(f" ... {len(report['errors']) - 25} further errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--partial", action="store_true", help="Check existing rows without requiring all sessions")
    args = parser.parse_args()
    if args.rebuild:
        rebuild_manifest()
    report = validate_dataset(require_complete=not args.partial)
    print_report(report)
    raise SystemExit(0 if report["valid"] else 1)
