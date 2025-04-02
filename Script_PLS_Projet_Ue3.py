# -*- coding: utf-8 -*-
"""
PLS + Logistic Regression on Hyperspectral Data
Author: ChatGPT & [Schwer Noé]
"""

# 📦 Importation des librairies nécessaires
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📂 Chemin d'accès vers le fichier CSV
file_path = r"C:\Users\schwe\Documents\Saves\Python\combined_data.csv"

# 🧪 Chargement des données
df = pd.read_csv(file_path)

# 🎯 Séparation des variables explicatives (X) et de la cible (y)
X = df.drop(columns=["class"]).values
y = df["class"].values

# 🔢 Encodage des classes (ex: 'canola' → 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 📉 Séparation en données d’entraînement/test (70/30 stratifié)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 🔄 Standardisation (obligatoire pour PLS)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚙️ Entraînement du modèle PLS + Logistic Regression
n_components = 10  # nombre de composantes PLS
pls = PLSRegression(n_components=n_components)
X_train_pls = pls.fit_transform(X_train_scaled, y_train)[0]
X_test_pls = pls.transform(X_test_scaled)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_pls, y_train)

# 🔍 Prédictions
y_pred = clf.predict(X_test_pls)

# ✅ Évaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy du modèle PLS + Régression Logistique : {accuracy:.2%}\n")

print("📊 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 📉 Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Matrice de confusion – PLS + Logistic Regression")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()



# ==========================================================
# 🔬 ANALYSES COMPLÉMENTAIRES DU MODÈLE PLS
# ==========================================================

# 1. 🎯 Variance expliquée par les composantes PLS
explained_variance = np.var(X_train_pls, axis=0) / np.sum(np.var(X_train_scaled, axis=0))

plt.figure(figsize=(8, 4))
plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, color="cornflowerblue")
plt.xlabel("Composante PLS")
plt.ylabel("Variance expliquée (%)")
plt.title("1️⃣ Variance expliquée par composante PLS")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. 🌀 Visualisation des échantillons projetés (2D)
X_pls_2D = X_train_pls[:, :2]
y_labels = le.inverse_transform(y_train)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pls_2D[:, 0], y=X_pls_2D[:, 1], hue=y_labels, palette="Set2", s=80, edgecolor="black")
plt.title("2️⃣ Projection des échantillons (2 premières composantes PLS)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.legend(title="Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. 📈 Importance des variables spectrales (poids dans les composantes)
pls_weights = pls.x_weights_  # Coefficients de projection

longueurs_donde = df.columns[:-1].astype(float)

plt.figure(figsize=(10, 6))
plt.plot(longueurs_donde, pls_weights[:, 0], label='Composante 1', linewidth=2)
plt.plot(longueurs_donde, pls_weights[:, 1], label='Composante 2', linewidth=2)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Poids des variables")
plt.title("3️⃣ Poids des longueurs d’onde dans les composantes PLS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 🧠 Optionnel – performance vs. nombre de composantes
# (utile si tu veux justifier le choix de 10 composantes)
from sklearn.model_selection import cross_val_score
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
