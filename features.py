"""The same fixed preprocessing and feature functions are used offline and live."""
from pathlib import Path

import cv2
import librosa
import numpy as np
import soundfile as sf
from skimage.feature import hog, local_binary_pattern

import config


def feature_settings():
    return {"version": config.FEATURE_VERSION, "sample_rate": config.SAMPLE_RATE,
            "seconds": config.RECORD_SECONDS, "silence_rms": config.SILENCE_RMS,
            "face_crop": config.FACE_CROP_FRACTION, "fingerprint_crop": config.FINGERPRINT_CROP_FRACTION}


def valid_vector(values, expected_size=None):
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Features must be a nonempty, finite, one-dimensional vector")
    if expected_size is not None and vector.size != expected_size:
        raise ValueError(f"Expected {expected_size} features, got {vector.size}")
    return vector


def read_image(image):
    if isinstance(image, (str, Path)):
        # imdecode also supports Windows paths with non-ASCII characters.
        image = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unreadable image")
    image = np.asarray(image)
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 32 or image.size == 0:
        raise ValueError("Image must be at least 32x32 pixels")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Expected grayscale, BGR, or BGRA image")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("Image pixels must be finite and between 0 and 255")
    return image.astype(np.uint8)


def crop_bounds(shape, modality):
    if modality not in ("face", "fingerprint"):
        raise ValueError("Unknown image modality")
    fraction = config.FACE_CROP_FRACTION if modality == "face" else config.FINGERPRINT_CROP_FRACTION
    height, width = shape[:2]
    size = int(min(height, width) * fraction)
    x, y = (width - size) // 2, (height - size) // 2
    return x, y, x + size, y + size


def image_features(image, modality, size):
    image = read_image(image)
    x1, y1, x2, y2 = crop_bounds(image.shape, modality)
    cropped = image[y1:y2, x1:x2]
    if cropped.ndim == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGRA2GRAY if cropped.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    if float(gray.std()) < 1:
        raise ValueError(f"{modality} crop is blank or has insufficient contrast")
    gradient = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    histogram, _ = np.histogram(lbp, bins=np.arange(11))
    histogram = histogram.astype(float) / histogram.sum()
    return valid_vector(np.concatenate((gradient, histogram)))


def extract_face_features(image):
    """Centered guide crop -> grayscale 96x96 -> 4356 HOG + 10 LBP."""
    return image_features(image, "face", 96)


def extract_fingerprint_features(image):
    """Centered ridge-area crop -> grayscale 128x128 -> 8100 HOG + 10 LBP."""
    return image_features(image, "fingerprint", 128)


def preprocess_audio(audio, sample_rate=config.SAMPLE_RATE):
    if isinstance(audio, (str, Path)):
        audio, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2 and 1 <= audio.shape[1] <= 8:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio is empty, nonfinite, or has an invalid channel shape")
    if not isinstance(sample_rate, (int, float)) or not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Invalid sample rate")
    if audio.size / sample_rate < 0.25:
        raise ValueError("Recording is too short (minimum 0.25 seconds)")
    if np.max(np.abs(audio)) > 1.01:
        raise ValueError("Audio must use floating point amplitudes between -1 and 1")
    if sample_rate != config.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=config.SAMPLE_RATE)
    length = int(config.SAMPLE_RATE * config.RECORD_SECONDS)
    audio = audio[:length]
    if float(np.sqrt(np.mean(audio.astype(float) ** 2))) < config.SILENCE_RMS:
        raise ValueError("Recording is silent or too quiet; record again")
    return np.pad(audio, (0, max(0, length - audio.size)))


def extract_voice_features(audio, sample_rate=config.SAMPLE_RATE):
    """Return 13 MFCC means/stds, ZCR mean/std, RMS mean/std: exactly 30."""
    audio = preprocess_audio(audio, sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=config.SAMPLE_RATE, n_mfcc=13,
                                n_fft=512, hop_length=160)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512, hop_length=160)
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=160)
    return valid_vector(np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1),
                                       zcr.mean(axis=1), zcr.std(axis=1),
                                       rms.mean(axis=1), rms.std(axis=1))), 30)


def create_fusion_features(face_features, voice_features, face_scaler=None,
                           voice_scaler=None, model_dir=config.MODEL_DIR):
    """Normalize each modality with TRAIN scalers, then concatenate real features."""
    if face_scaler is None or voice_scaler is None:
        from verification import load_artifact
        artifact = load_artifact("fusion", model_dir)
        face_scaler, voice_scaler = artifact["scalers"]["face"], artifact["scalers"]["voice"]
    face = valid_vector(face_features, face_scaler.n_features_in_)
    voice = valid_vector(voice_features, voice_scaler.n_features_in_)
    return valid_vector(np.concatenate((face_scaler.transform([face])[0], voice_scaler.transform([voice])[0])))
