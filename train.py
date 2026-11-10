"""Machine Learning Gesture Training Pipeline for Vision-Gesture.

Provides:
- Ingestion of real landmark CSV datasets from `data/raw_landmarks/` (or synthetic benchmark
  generator fallback if no recorded CSVs are present).
- Pose-invariant feature extraction via features.normalize_landmarks.
- Stratified 5-Fold Cross Validation.
- High-performance ensemble training (Random Forest Classifier).
- Precision, Recall, F1 evaluation & confusion matrix reporting.
- Serialization of model (`gesture_model.pkl`) and encoder (`label_encoder.pkl`) to both
  `models/` and root directory for seamless CLI and package compatibility.
"""

import glob
import math
import os
from typing import Tuple

import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from features import normalize_landmarks


def generate_synthetic_benchmark_dataset(samples_per_class: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-fidelity synthetic biomechanical hand landmarks for canonical gestures.

    Gestures modeled:
    - 'Pointer': Index extended, thumb/middle/ring/pinky curled.
    - 'Pinch': Thumb and index fingertip touching, other fingers curled.
    - 'Peace_Scroll': Index and middle extended, ring and pinky curled.
    - 'Open_Palm': All 5 fingers extended outward.
    - 'Fist_Clutch': All 5 fingers tightly curled toward palm.
    - 'Thumbs_Up': Thumb extended upward, other fingers curled.
    """
    classes = ["Pointer", "Pinch", "Peace_Scroll", "Open_Palm", "Fist_Clutch", "Thumbs_Up"]
    X = []
    y = []

    # Canonical 21 landmark base template (wrist at 0, 0, 0)
    base_lms = np.array([
        [0.0, 0.0, 0.0],          # 0: wrist
        [-0.08, -0.06, 0.0],      # 1: thumb CMC
        [-0.14, -0.14, 0.0],      # 2: thumb MCP
        [-0.18, -0.22, 0.0],      # 3: thumb IP
        [-0.20, -0.28, 0.0],      # 4: thumb tip
        [-0.06, -0.22, 0.0],      # 5: index MCP
        [-0.07, -0.32, 0.0],      # 6: index PIP
        [-0.07, -0.40, 0.0],      # 7: index DIP
        [-0.07, -0.48, 0.0],      # 8: index tip
        [0.00, -0.24, 0.0],       # 9: middle MCP
        [0.00, -0.35, 0.0],       # 10: middle PIP
        [0.00, -0.44, 0.0],       # 11: middle DIP
        [0.00, -0.52, 0.0],       # 12: middle tip
        [0.06, -0.22, 0.0],       # 13: ring MCP
        [0.07, -0.32, 0.0],       # 14: ring PIP
        [0.07, -0.40, 0.0],       # 15: ring DIP
        [0.07, -0.48, 0.0],       # 16: ring tip
        [0.12, -0.18, 0.0],       # 17: pinky MCP
        [0.14, -0.26, 0.0],       # 18: pinky PIP
        [0.15, -0.33, 0.0],       # 19: pinky DIP
        [0.16, -0.40, 0.0],       # 20: pinky tip
    ], dtype=np.float64)

    rng = np.random.default_rng(42)

    for cls in classes:
        for _ in range(samples_per_class):
            lms = base_lms.copy()

            if cls == "Pointer":
                lms[[10, 11, 12], 1] = lms[9, 1] + 0.05
                lms[[14, 15, 16], 1] = lms[13, 1] + 0.05
                lms[[18, 19, 20], 1] = lms[17, 1] + 0.05
                lms[[3, 4], 0] = -0.08
            elif cls == "Pinch":
                lms[4] = [-0.04, -0.35, 0.0]
                lms[8] = [-0.04, -0.35, 0.0]
                lms[[10, 11, 12], 1] = lms[9, 1] + 0.05
                lms[[14, 15, 16], 1] = lms[13, 1] + 0.05
                lms[[18, 19, 20], 1] = lms[17, 1] + 0.05
            elif cls == "Peace_Scroll":
                lms[[14, 15, 16], 1] = lms[13, 1] + 0.05
                lms[[18, 19, 20], 1] = lms[17, 1] + 0.05
                lms[[3, 4], 0] = -0.06
            elif cls == "Open_Palm":
                lms[[4, 8, 12, 16, 20], 1] -= 0.02
            elif cls == "Fist_Clutch":
                lms[[6, 7, 8], 1] = lms[5, 1] + 0.04
                lms[[10, 11, 12], 1] = lms[9, 1] + 0.04
                lms[[14, 15, 16], 1] = lms[13, 1] + 0.04
                lms[[18, 19, 20], 1] = lms[17, 1] + 0.04
                lms[[3, 4], 0] = -0.05
            elif cls == "Thumbs_Up":
                lms[4] = [-0.10, -0.42, 0.0]
                lms[[6, 7, 8], 1] = lms[5, 1] + 0.04
                lms[[10, 11, 12], 1] = lms[9, 1] + 0.04
                lms[[14, 15, 16], 1] = lms[13, 1] + 0.04
                lms[[18, 19, 20], 1] = lms[17, 1] + 0.04

            noise = rng.normal(0, 0.012, size=lms.shape)
            lms += noise

            angle = rng.uniform(-0.52, 0.52)
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            lms = lms @ R.T

            scale = rng.uniform(0.8, 1.25)
            lms *= scale
            lms += rng.uniform(-0.1, 0.1, size=(1, 3))

            feat = normalize_landmarks(lms)
            X.append(feat)
            y.append(cls)

    return np.array(X, dtype=np.float64), np.array(y)


def train_model() -> None:
    """Train and evaluate the gesture classifier."""
    print("=" * 65)
    print(" Vision-Gesture: ML Model Training & Validation Pipeline")
    print("=" * 65)

    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn or joblib not installed in environment.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_patterns = [
        os.path.join(script_dir, "*.csv"),
        os.path.join(script_dir, "data", "*.csv"),
        os.path.join(script_dir, "data", "raw_landmarks", "*.csv"),
        os.path.join(script_dir, "dataset", "*.csv"),
    ]
    csv_files = []
    for pat in csv_patterns:
        csv_files.extend(glob.glob(pat))
    csv_files = sorted(set(csv_files))

    X, y = None, None

    if csv_files:
        print(f"Loading datasets from {len(csv_files)} CSV file(s)...")
        records_x, records_y = [], []
        for file in csv_files:
            try:
                data = np.loadtxt(file, delimiter=",", skiprows=1, dtype=object)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                for row in data:
                    label = str(row[-1]).strip()
                    coords = np.array(row[:-1], dtype=np.float64)
                    if coords.size == 63:
                        lms = coords.reshape(21, 3)
                        feat = normalize_landmarks(lms)
                        records_x.append(feat)
                        records_y.append(label)
            except Exception as e:
                print(f"  Warning: could not parse {file}: {e}")
        if records_x:
            X = np.array(records_x)
            y = np.array(records_y)
            print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes from CSV.")

    if X is None or len(X) == 0:
        print("No user CSV datasets found (use `python collect_dataset.py` to record real subjects).")
        print("Generating synthetic biomechanical benchmark gesture dataset...")
        X, y = generate_synthetic_benchmark_dataset(samples_per_class=200)
        print(f"Generated {len(X)} benchmark samples across {len(np.unique(y))} gesture classes.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nTarget Classes ({len(le.classes_)}): {', '.join(le.classes_)}")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y_encoded, cv=skf, scoring="accuracy")

    print("\n--- 5-Fold Stratified Cross-Validation ---")
    for fold, score in enumerate(scores, 1):
        print(f"  Fold {fold}: Accuracy = {score * 100:.2f}%")
    print(f"  Mean Accuracy: {scores.mean() * 100:.2f}% (std: {scores.std() * 100:.2f}%)")

    clf.fit(X, y_encoded)
    preds = clf.predict(X)
    print("\n--- Classification Report ---")
    print(classification_report(y_encoded, preds, target_names=le.classes_))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_encoded, preds))

    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for target_dir in (models_dir, script_dir):
        joblib.dump(clf, os.path.join(target_dir, "gesture_model.pkl"))
        joblib.dump(le, os.path.join(target_dir, "label_encoder.pkl"))

    print(f"\n✓ Model serialized to:         {os.path.join(models_dir, 'gesture_model.pkl')}")
    print(f"✓ Label encoder serialized to: {os.path.join(models_dir, 'label_encoder.pkl')}")
    print("=" * 65)


if __name__ == "__main__":
    train_model()
