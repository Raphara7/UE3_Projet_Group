# -*- coding: utf-8 -*-
"""
PLS + Logistic Regression on Hyperspectral Data (Brut + Dérivées 1, 2 & 3)
Author: ChatGPT & [Schwer Noé]
"""

# 📦 Importation des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

# 📂 Chargement des données
file_path = 'C:/Users/schwe/Documents/Saves/Python/combined_data.csv'
df = pd.read_csv(file_path)

# 🎯 Variables X et y
X_raw = df.drop(columns=["class"]).values
y = df["class"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 🧪 Calcul des dérivées
X_deriv1 = savgol_filter(X_raw, 11, 2, deriv=1)
X_deriv2 = savgol_filter(X_raw, 11, 2, deriv=2)
X_deriv3 = savgol_filter(X_raw, 11, 2, deriv=3)

# 🔀 Concaténation des données
X_combined = np.concatenate([X_raw, X_deriv1, X_deriv2, X_deriv3], axis=1)

# 📉 Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# 🔄 Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚙️ PLS + Régression Logistique
n_components = 5
pls = PLSRegression(n_components=n_components)
X_train_pls = pls.fit_transform(X_train_scaled, y_train)[0]
X_test_pls = pls.transform(X_test_scaled)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_pls, y_train)
y_pred = clf.predict(X_test_pls)

# ✅ Évaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy (Brut + dérivées 1, 2, 3) : {accuracy:.2%}\n")
print("📊 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 📉 Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de confusion – PLS + Logistic Regression (avec dérivées)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# 📊 Variance expliquée par composantes PLS
explained_variance = np.var(X_train_pls, axis=0) / np.sum(np.var(X_train_scaled, axis=0))
plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, color="cornflowerblue")
plt.xlabel("Composante PLS")
plt.ylabel("Variance expliquée (%)")
plt.title("1️⃣ Variance expliquée par composante PLS")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📈 Projection 2D (2 premières composantes PLS)
X_pls_2D = X_train_pls[:, :2]
y_labels = le.inverse_transform(y_train)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pls_2D[:, 0], y=X_pls_2D[:, 1], hue=y_labels,
                palette="Set2", s=80, edgecolor="black")
plt.title("2️⃣ Projection des échantillons (composantes PLS 1 & 2)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 📈 Poids spectrales
pls_weights = pls.x_weights_
longueurs_donde = np.arange(X_combined.shape[1])  # index générique
plt.figure(figsize=(10, 6))
plt.plot(longueurs_donde, pls_weights[:, 0], label='Composante 1', linewidth=2)
plt.plot(longueurs_donde, pls_weights[:, 1], label='Composante 2', linewidth=2)
plt.xlabel("Index des variables (brut + dérivées)")
plt.ylabel("Poids")
plt.title("3️⃣ Poids des variables dans les composantes PLS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧪 Courbe Accuracy vs nombre de composantes
accuracies = []
for n in range(1, 21):
    pls_tmp = PLSRegression(n_components=n)
    X_pls_tmp = pls_tmp.fit_transform(X_train_scaled, y_train)[0]
    clf_tmp = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf_tmp, X_pls_tmp, y_train, cv=5, scoring='accuracy')
    accuracies.append(scores.mean())

plt.figure(figsize=(8, 4))
plt.plot(range(1, 21), accuracies, marker='o', color="teal")
plt.title("4️⃣ Accuracy vs. nombre de composantes PLS")
plt.xlabel("Nombre de composantes PLS")
plt.ylabel("Accuracy moyenne (CV 5-fold)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 📉 MSE vs nombre de composantes
mse_scores = []
for n in range(1, 21):
    pls_tmp = PLSRegression(n_components=n)
    X_train_tmp = pls_tmp.fit_transform(X_train_scaled, y_train)[0]
    X_test_tmp = pls_tmp.transform(X_test_scaled)
    clf_tmp = LogisticRegression(max_iter=1000)
    clf_tmp.fit(X_train_tmp, y_train)
    y_pred_tmp = clf_tmp.predict(X_test_tmp)
    mse = mean_squared_error(y_test, y_pred_tmp)
    mse_scores.append(mse)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 21), mse_scores, marker='s', color='crimson')
plt.title("5️⃣ MSE vs. nombre de composantes PLS")
plt.xlabel("Nombre de composantes PLS")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 🔍 Nombre de paramètres dans la régression logistique ===
n_weights = pls.coef_.size          # Poids : shape = (n_classes, n_components)
n_biases = pls.intercept_.size      # Biais : 1 par classe
total_params = n_weights + n_biases

print(f"Nombre de poids (régression logistique) : {n_weights}")
print(f"Nombre de biais                          : {n_biases}")
print(f"Nombre total de paramètres              : {total_params}")

