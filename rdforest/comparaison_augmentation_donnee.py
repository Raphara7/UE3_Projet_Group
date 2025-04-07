import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])

# Séparation X et y
X_raw = df.drop(columns=["class", "label_encoded"])
y = df["label_encoded"]

# Standardisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# Split train/test
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ============================
# Pipeline SANS augmentation (X brut)
# ============================
model_raw = RandomForestClassifier(n_estimators=100, random_state=42)
model_raw.fit(X_train_raw, y_train_raw)
y_pred_raw = model_raw.predict(X_test)

metrics_raw = {
    "Accuracy": accuracy_score(y_test, y_pred_raw),
    "Precision": precision_score(y_test, y_pred_raw, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred_raw, average="weighted"),
    "F1 Score": f1_score(y_test, y_pred_raw, average="weighted"),
}

# ============================
# Pipeline AVEC augmentation (Gaussian noise sur X brut)
# ============================
# Ajout de bruit à X_train_raw
noise = np.random.normal(0, 0.05, X_train_raw.shape)
X_train_noisy = X_train_raw + noise
y_train_noisy = y_train_raw.copy()

# Fusion original + bruité
X_train_aug = pd.concat([X_train_raw, pd.DataFrame(X_train_noisy, columns=X_train_raw.columns)])
y_train_aug = pd.concat([y_train_raw, y_train_noisy])

# Entraînement
model_aug = RandomForestClassifier(n_estimators=100, random_state=42)
model_aug.fit(X_train_aug, y_train_aug)
y_pred_aug = model_aug.predict(X_test)

metrics_augmented = {
    "Accuracy": accuracy_score(y_test, y_pred_aug),
    "Precision": precision_score(y_test, y_pred_aug, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred_aug, average="weighted"),
    "F1 Score": f1_score(y_test, y_pred_aug, average="weighted"),
}

# ============================
# Comparaison visuelle
# ============================
metrics_df = pd.DataFrame({
    "Données brutes": metrics_raw,
    "Données + Gaussian noise": metrics_augmented
})

print(metrics_df)

# Histogramme
metrics_df.plot.bar(rot=0, figsize=(10, 6))
plt.title("Comparaison des métriques : Données brutes vs. Données augmentées")
plt.ylabel("Score")
plt.tight_layout()
plt.show()
