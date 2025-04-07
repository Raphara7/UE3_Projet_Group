# ============================
# Importation des dépendances
# ============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.fft import fft
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Chargement et préparation des données
# ============================
df = pd.read_csv("combined_data.csv")
# Encodage de la variable "class" en entier non ordonné
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])

# Séparation des données brutes et du label
X_raw = df.drop(columns=["class", "label_encoded"])
y = df["label_encoded"]

# Standardisation des données brutes
scaler = StandardScaler()
X_raw_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# ============================
# 2. Pipeline sans augmentation de données
# ============================
print("=== Pipeline sans augmentation de données ===")
# Séparation en jeu d'entraînement et de test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Entraînement du modèle sur les données brutes standardisées
rf_model_raw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_raw.fit(X_train_raw, y_train)

# Prédictions et évaluation
y_pred_raw = rf_model_raw.predict(X_test_raw)
accuracy_raw = accuracy_score(y_test, y_pred_raw)
print(f"🎯 Accuracy sans augmentation : {accuracy_raw:.2%}")
print("📊 Rapport de classification (sans augmentation) :")
print(classification_report(y_test, y_pred_raw))

# Matrice de confusion
conf_matrix_raw = confusion_matrix(y_test, y_pred_raw)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_raw, annot=True, fmt='d', cmap="Greens")
plt.title("Matrice de confusion – Random Forest (sans augmentation)")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.tight_layout()
plt.show()

# ============================
# 3. Pipeline avec augmentation de données
# ============================
print("\n=== Pipeline avec augmentation de données ===")
# Calcul de la transformée de Fourier (FFT) et récupération de la magnitude
fft_features = np.abs(fft(X_raw_scaled.values, axis=1))
fft_columns = ["fft_" + str(i) for i in range(fft_features.shape[1])]
df_fft = pd.DataFrame(fft_features, columns=fft_columns)

# Calcul de la transformée de Hilbert et récupération de l'enveloppe
hilbert_features = np.abs(hilbert(X_raw_scaled.values, axis=1))
hilbert_columns = ["hilbert_" + str(i) for i in range(hilbert_features.shape[1])]
df_hilbert = pd.DataFrame(hilbert_features, columns=hilbert_columns)

# Concaténation des caractéristiques dérivées
derived_features = pd.concat([df_fft, df_hilbert], axis=1)

# Réduction de dimension avec PCA sur les données dérivées
pca = PCA(n_components=10, random_state=42)
derived_pca = pd.DataFrame(pca.fit_transform(derived_features),
                           columns=[f"pca_{i}" for i in range(10)])

# Constitution du jeu de données combiné (données brutes + données dérivées réduites)
X_combined = pd.concat([X_raw_scaled, derived_pca], axis=1)

# Séparation en jeu d'entraînement et de test pour le jeu combiné
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.3, random_state=42, stratify=y
)

# Entraînement du modèle sur le jeu de données combiné
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy avec augmentation : {accuracy:.2%}")
print("📊 Rapport de classification (avec augmentation) :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Matrice de confusion – Random Forest (avec augmentation)")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.tight_layout()
plt.show()

# ============================
# 4. Importance des variables (top 20) – Pipeline avec augmentation
# ============================
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
feature_names = X_combined.columns
top_features = feature_names[indices][:top_n]
top_importances = importances[indices][:top_n]

print("Les 20 variables les plus discriminantes (avec augmentation) :")
for feature, importance in zip(top_features, top_importances):
    print(f"{feature} : {importance}")

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features)
plt.title("Importance des 20 variables (avec augmentation)")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# ============================
# 5. Evaluation de l'accuracy en fonction du nombre d'arbres (optionnel)
# ============================
n_estimators_range = range(10, 210, 10)
cv_scores = []

for n in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, cv_scores, marker='o', color='green')
plt.title("Accuracy vs. nombre d'arbres (CV 5-fold, avec augmentation)")
plt.xlabel("Nombre d'arbres")
plt.ylabel("Accuracy moyenne")
plt.grid(True)
plt.tight_layout()
plt.show()
