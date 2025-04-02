# -*- coding: utf-8 -*-
"""
Random Forest sur données hyperspectrales
Auteur : ChatGPT & [Ton nom]
"""

# 📦 Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# 📂 Chargement des données
file_path = r"C:\Users\schwe\Documents\Saves\Python\combined_data.csv"
df = pd.read_csv(file_path)

# 🎯 Séparation des variables
X = df.drop(columns=["class"]).values
y = df["class"].values

# 🔢 Encodage des classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ✂️ Séparation en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 🌲 Entraînement du modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 🔍 Prédictions et évaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy du modèle Random Forest : {accuracy:.2%}\n")
print("📊 Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 📉 Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de confusion – Random Forest")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.tight_layout()
plt.show()

# 🔬 Importance des variables (top 20)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
top_features = df.columns[:-1].astype(float)[indices[:top_n]]
top_importances = importances[indices[:top_n]]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features)
plt.title("Importance des 20 longueurs d’onde les plus discriminantes")
plt.xlabel("Importance")
plt.ylabel("Longueur d’onde (nm)")
plt.tight_layout()
plt.show()

# 📈 Accuracy vs. nombre d’arbres
n_estimators_range = range(10, 210, 10)
cv_scores = []

for n in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, cv_scores, marker='o', color='green')
plt.title("Accuracy vs. nombre d’arbres (CV 5-fold)")
plt.xlabel("Nombre d’arbres")
plt.ylabel("Accuracy moyenne")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🌳 Visualisation de quelques arbres de décision
for idx in range(3):  # arbres n°1 à 3
    tree = rf_model.estimators_[idx]
    plt.figure(figsize=(20, 10))
    plot_tree(tree,
              feature_names=df.columns[:-1],
              class_names=le.classes_,
              filled=True,
              rounded=True,
              max_depth=3,
              fontsize=8)
    plt.title(f"Arbre de décision n°{idx + 1} (profondeur max = 3)")
    plt.tight_layout()
    plt.show()
