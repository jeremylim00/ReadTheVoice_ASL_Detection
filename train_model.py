
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading ASL dataset...")

df = pd.read_csv("dataset/asl_sequences.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Separate labels and features
y = df["label"]

# Exclude label and session_id from features
X = df.drop(columns=["label", "session_id"],
            errors="ignore")

# Check for missing values
X = X.fillna(0)

print("\nNumber of samples:", len(X))
print("Number of features:", X.shape[1])

print("\nClass distribution:")
print(y.value_counts())

# ==========================================
# 2. SPLIT TRAINING AND TESTING DATA
# ==========================================

print("\nSplitting dataset...")

if "session_id" in df.columns and df["session_id"].nunique() >= 2:

    splitter = GroupShuffleSplit(
        n_splits=20,
        test_size=0.2,
        random_state=42
    )

    groups = df["session_id"]

    # Find a split containing every class
    for train_idx, test_idx in splitter.split(X, y, groups):

        if (set(y.iloc[train_idx]) == set(y) and
                set(y.iloc[test_idx]) == set(y)):

            break
    else:
        raise ValueError(
            "Could not find a session-based split "
            "containing all classes. Collect more sessions."
        )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 3. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 4. TRAIN MODEL
# ==========================================

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 5. TEST MODEL
# ==========================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 6. SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, "asl_model.pkl")

print("\nModel saved successfully!")
print("Saved file: asl_model.pkl")

print("\nASL MODEL TRAINING FINISHED!")
