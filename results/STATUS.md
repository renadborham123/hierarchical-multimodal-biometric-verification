# Implementation and evidence report

Date: November 8, 2026 (Africa/Cairo). This report distinguishes implemented code from real biometric evidence.

## Implemented files

`main.py`, `config.py`, `collect_data.py`, `features.py`, `dataset.py`, `train_models.py`, `verification.py`, `commands.py`, `permissions.py`, `actions.py`, `pipeline.py`, and `evaluate.py` are implemented. The four required test files, requirements, ignore rules, empty manifest, README, and English/Egyptian-Arabic oral defense are included. Raw/model/result directories are prepared. There are no invented participant files or placeholder measurements.

## Actual checks performed

- Python 3.13.5; all nine direct project dependencies installed/importable in `.venv`.
- `python -m unittest discover -s tests -v`: **40 tests passed**. Numerical features were checked on geometric patterns and diagnostic audio tones. Permission/branch tests use explicit doubles and do not measure biometric accuracy.
- Compilation/import checks for all application modules and tests passed.
- `dataset.py`: correctly rejects the empty dataset; 160 paired face/audio rows and 40 genuine impressions are missing for the default four users.
- `train_models.py`: correctly refuses to train on the empty dataset; no artifacts saved.
- `evaluate.py`: correctly refuses to evaluate the empty dataset; no metrics fabricated.
- `main.py`: ten-option menu displayed and Exit completed through piped CLI input.
- `collect_data.py --check-hardware`: all dependencies importable; **camera unavailable** at the configured device index. Microphone input devices were enumerated and default mono/16 kHz input settings accepted. Actual recording was not tested.

The final unit-test transcript is saved as `results/test_run.txt`. None of these checks demonstrates real-person identification, verification, command recognition, or a complete live biometric demo.

## Ten required models

| Model | Code | Real training status |
|---|---|---|
| face_knn | KNeighborsClassifier(n_neighbors=3) | BLOCKED: no real face images |
| face_svm | SVC(kernel="linear") | BLOCKED: no real face images |
| voice_knn | KNeighborsClassifier(n_neighbors=3) | BLOCKED: no real recordings |
| voice_svm | SVC(kernel="linear") | BLOCKED: no real recordings |
| fusion_knn | KNeighborsClassifier(n_neighbors=3) | BLOCKED: no matched real face/audio pairs |
| fusion_svm | SVC(kernel="linear") | BLOCKED: no matched real face/audio pairs |
| fingerprint_knn | KNeighborsClassifier(n_neighbors=3) | BLOCKED: no genuine ink impressions |
| fingerprint_svm | SVC(kernel="linear") | BLOCKED: no genuine ink impressions |
| command_knn | KNeighborsClassifier(n_neighbors=3) | BLOCKED: no labeled spoken commands |
| command_svm | SVC(kernel="linear") | BLOCKED: no labeled spoken commands |

After collection, training saves both fitted models for each task in one bundle, selects by validation macro F1/accuracy, calibrates thresholds on validation only, and runs frozen TEST evaluation. No fitted biometric bundle exists yet.

## Integrated pipeline and bonuses

Claim ID → live face features/claim verification → one recording/voice features/claim verification → separately normalized concatenated features/separate fusion claim verification → new ink impression/fingerprint claim verification → command recognition from the same voice vector → verified user's role/permission → simulated action. Every failed gate stops the later stages. Supplied offline samples always disable actions.

| Bonus | Implementation status | Real demonstration status |
|---|---|---|
| 1: spoken commands | Connected after four successful biometric gates | BLOCKED by missing recordings/models |
| 2: live webcam/microphone | Real capture functions and availability checks implemented | BLOCKED by unavailable camera and missing trained models; microphone recording untested |
| 3: custom collection | Session collection, manifest, integrity/count/leakage checks implemented | BLOCKED until participants collect genuine data |
| 4: fingerprint | HOG/LBP, real algorithms, claim/distance gate, mandatory pipeline stage implemented | BLOCKED until genuine impressions/models are available |

## Definition-of-done audit

| Requirement | Evidence / status |
|---|---|
| Face feature extraction | PASS on a diagnostic pattern: 4366 finite values; real face validation pending |
| Voice feature extraction | PASS on diagnostic tones/file/stereo/resampling: 30 finite values; real speech validation pending |
| Fingerprint feature extraction | PASS on a diagnostic pattern: 8110 finite values; real fingerprint validation pending |
| All ten models actually trained | BLOCKED: no genuine custom dataset |
| Face-only verification with real samples | BLOCKED: no trained face bundle |
| Voice-only verification with real samples | BLOCKED: no trained voice bundle |
| Genuine early concatenation | PASS: scaler/concatenation tests; 4396-dimensional production path implemented |
| Fusion verification with real samples | BLOCKED: no trained fusion bundle |
| Fingerprint verification with real samples | BLOCKED: no genuine samples/trained bundle |
| Live webcam | BLOCKED: configured camera probe unavailable |
| Live microphone | Devices/settings checked; actual recording UNTESTED |
| Same recording/vector reused | PASS in pipeline control-flow tests |
| Unauthorized commands denied | PASS in required Admin/Employee/Guest branch tests |
| Any failed biometric check stops | PASS in all four stage tests; later recognition/actions not called |
| Real matched face/audio dataset | BLOCKED: participants must supply 160 images + 160 recordings |
| Real fingerprint dataset | BLOCKED: participants must supply 40 separate genuine impressions |
| Real TEST metrics and confusion matrices | BLOCKED: no real data/fitted artifacts; generation code implemented |
| Four bonuses integrated | Implementation connected; real end-to-end evidence BLOCKED |
| README and oral defense | Written, including every module and Egyptian Arabic explanations |
| Automated tests | PASS: 40 tests |

## Students' exact next steps

1. Review user IDs/roles in `config.py`; keep four users or configure three. For three users the missing totals become 120 pairs and 30 impressions.
2. Connect a webcam, verify `CAMERA_INDEX`, and check a microphone. Run `.\.venv\Scripts\python.exe collect_data.py --check-hardware`.
3. For each configured user and each session 1–10, run `.\.venv\Scripts\python.exe collect_data.py --user user_01 --session 1`, changing user/session each time. Record all four words and collect a separately acquired same-finger ink impression per session. README includes phone/scanner acquisition instructions.
4. Run `.\.venv\Scripts\python.exe dataset.py` and resolve all reported issues. Do not train with incomplete or artificial data.
5. Run `.\.venv\Scripts\python.exe train_models.py`. Review generated validation evidence and TEST tables/images. Do not change thresholds to improve TEST performance.
6. Run `.\.venv\Scripts\python.exe main.py`, choose 9, claim a user, capture a new face, speak one command, and present/import a new ink impression after prior gates accept.
7. Demonstrate an Admin requesting UNLOCK and an Employee requesting UNLOCK; record actual successes/rejections honestly. Use options 5–8 for separate no-action comparisons.

No deep learning or pretrained biometric/speech model was used. This assignment's real-data and hardware requirements remain explicit blockers; the project is not claimed fully complete.
