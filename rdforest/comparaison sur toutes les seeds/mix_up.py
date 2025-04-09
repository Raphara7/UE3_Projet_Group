import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
import random

# === Chargement et prétraitement des données ===
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])
X_raw = df.drop(columns=["class", "label_encoded"])
y_raw = df["label_encoded"]

# Calcul des dérivées
X_deriv1 = X_raw.diff(axis=1).fillna(0)
X_deriv1.columns = [f"{col}_d1" for col in X_raw.columns]
X_deriv2 = X_raw.diff(axis=1).diff(axis=1).fillna(0)
X_deriv2.columns = [f"{col}_d2" for col in X_raw.columns]

X_combined = pd.concat([X_raw, X_deriv1, X_deriv2], axis=1)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)

# === Augmentation Mixup ===
def mixup_augmentation(X, y, n_samples, alpha=0.2, seed=None, gaussian_std=0.01, jitter_factor=0.02, prob=0.5):
    np.random.seed(seed)
    X = X.to_numpy()
    y = y.to_numpy()
    idx1 = np.random.randint(0, len(X), size=n_samples)
    idx2 = np.random.randint(0, len(X), size=n_samples)
    lam = np.random.beta(alpha, alpha, size=n_samples).reshape(-1, 1)
    X_mix = lam * X[idx1] + (1 - lam) * X[idx2]
    if np.random.rand() < prob:
        noise = np.random.normal(0, gaussian_std, X_mix.shape)
        jitter = np.random.uniform(1 - jitter_factor, 1 + jitter_factor, X_mix.shape)
        X_mix = (X_mix + noise) * jitter
    y_mix = y[idx1]
    return pd.DataFrame(X_mix, columns=X_scaled.columns), pd.Series(y_mix, name="label_encoded")

# === Fonction d'entraînement ===
def train_and_evaluate(seed, params, save_model=False, model_id=None):
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=seed)
    df_train = pd.concat([X_train_base, y_train_base], axis=1)
    max_count = df_train["label_encoded"].value_counts().max()
    balanced_train = []
    for label in df_train["label_encoded"].unique():
        class_subset = df_train[df_train["label_encoded"] == label]
        n_to_generate = max_count - len(class_subset)
        if n_to_generate > 0:
            X_class = class_subset.drop(columns=["label_encoded"])
            y_class = class_subset["label_encoded"]
            X_synth, y_synth = mixup_augmentation(X_class, y_class, n_to_generate, alpha=0.2, seed=seed)
            class_subset = pd.concat([class_subset, pd.concat([X_synth, y_synth], axis=1)])
        balanced_train.append(class_subset)
    df_train_bal = pd.concat(balanced_train)
    df_train_bal = shuffle(df_train_bal, random_state=seed)
    X_train = df_train_bal.drop(columns=["label_encoded"])
    y_train = df_train_bal["label_encoded"]
    model = RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    if save_model and model_id is not None:
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(model, f"saved_models/model_{model_id}.joblib")
    y_pred = model.predict(X_test_base)
    return accuracy_score(y_test_base, y_pred)

# === Génération des 300 combinaisons ===
param_grid = []
for _ in range(300):
    combo = {
        "n_estimators": random.choice([50, 100, 150, 200]),
        "max_depth": random.choice([10, 20, 30, None]),
        "min_samples_split": random.choice([2, 4, 6]),
        "min_samples_leaf": random.choice([1, 2, 4]),
        "max_features": random.choice(["sqrt", "log2"]),
        "criterion": random.choice(["gini", "entropy"])
    }
    param_grid.append(combo)

# === Recherche des meilleurs hyperparamètres ===
results = []
start = time.time()
for i, params in enumerate(param_grid):
    accuracies = []
    for seed in range(100):
        acc = train_and_evaluate(seed, params)
        accuracies.append(acc)
    avg_acc = np.mean(accuracies)
    results.append({"params": params, "avg_accuracy": avg_acc, "accuracies": accuracies})
    elapsed = time.time() - start
    remaining = (elapsed / (i + 1)) * (len(param_grid) - (i + 1))
    print(f"[{i+1}/300] Accuracy moyenne: {avg_acc:.4f} | Temps écoulé: {elapsed/60:.2f} min | Temps restant estimé: {remaining/60:.2f} min")

# === Sauvegarde des 10 meilleurs modèles ===
top_10 = sorted(results, key=lambda x: x["avg_accuracy"], reverse=True)[:10]
for idx, result in enumerate(top_10):
    print(f"\n\n\u2728 Sauvegarde du modèle #{idx+1} (accuracy: {result['avg_accuracy']:.4f})")
    train_and_evaluate(0, result["params"], save_model=True, model_id=f"top_{idx+1}")

# === Meilleure combinaison ===
best_result = top_10[0]
print("\nMeilleure accuracy moyenne:", best_result["avg_accuracy"])
print("Paramètres:", best_result["params"])

# === Heatmap des 100 accuracies de la meilleure combinaison ===
acc_matrix = np.array(best_result["accuracies"]).reshape(10, 10)
plt.figure(figsize=(8, 6))
sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Accuracy'})
plt.title("Accuracy sur les 100 seeds - Meilleurs hyperparamètres")
plt.xlabel("Colonne (seed % 10)")
plt.ylabel("Ligne (seed // 10)")
plt.tight_layout()
plt.show()

# === Tableau récapitulatif ===
summary_df = pd.DataFrame([{
    **res["params"],
    "avg_accuracy": res["avg_accuracy"]
} for res in results])

import ace_tools as tools
tools.display_dataframe_to_user(name="Résultats des hyperparamètres", dataframe=summary_df.sort_values(by="avg_accuracy", ascending=False))
