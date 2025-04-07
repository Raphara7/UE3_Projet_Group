# ============================
# Importation des dépendances
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

from scipy.fft import fft
from scipy.signal import hilbert

# Pour le suréchantillonnage
from imblearn.over_sampling import SMOTE

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
# 2. Création des données spectrales dérivées
# ============================
# Calcul de la transformée de Fourier (FFT)
fft_features = np.abs(fft(X_raw_scaled.values, axis=1))
fft_columns = ["fft_" + str(i) for i in range(fft_features.shape[1])]
df_fft = pd.DataFrame(fft_features, columns=fft_columns)

# Calcul de la transformée de Hilbert
hilbert_features = np.abs(hilbert(X_raw_scaled.values, axis=1))
hilbert_columns = ["hilbert_" + str(i) for i in range(hilbert_features.shape[1])]
df_hilbert = pd.DataFrame(hilbert_features, columns=hilbert_columns)

# Optionnel : Réduction de dimension des données dérivées via PCA
derived_features = pd.concat([df_fft, df_hilbert], axis=1)
pca = PCA(n_components=10, random_state=42)
derived_pca = pd.DataFrame(pca.fit_transform(derived_features),
                           columns=[f"pca_{i}" for i in range(10)])

# ============================
# 3. Constitution du jeu de données combiné
# ============================
# Concaténation des données brutes standardisées et des données dérivées (PCA)
X_combined = pd.concat([X_raw_scaled, derived_pca], axis=1)

# ============================
# 4. Visualisation de la répartition initiale des classes
# ============================
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribution des classes avant augmentation")
plt.xlabel("Classe (encodée)")
plt.ylabel("Nombre d'échantillons")
plt.show()

# ============================
# 5. Suréchantillonnage avec SMOTE
# ============================
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_combined, y)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_balanced)
plt.title("Distribution des classes après SMOTE")
plt.xlabel("Classe (encodée)")
plt.ylabel("Nombre d'échantillons")
plt.show()

# ============================
# 6. Augmentation des données spectrales pour les classes minoritaires
# ============================
# Pour illustrer, nous allons sélectionner un échantillon provenant de la classe minoritaire
class_counts = y.value_counts()
minority_class = class_counts.idxmin()  # classe avec le moins d'exemples
sample_index = df[df["label_encoded"] == minority_class].index[0]
original_fft_sample = fft_features[sample_index, :]

# Fonctions d'augmentation spectrale
def augment_spectral_data(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def shift_spectral_data(X, shift=1):
    return np.roll(X, shift, axis=0)

# Génération de données augmentées : ajout de bruit et décalage
fft_sample_noisy = augment_spectral_data(original_fft_sample, noise_level=0.05)
fft_sample_shifted = shift_spectral_data(original_fft_sample, shift=5)

# Visualisation de l'augmentation sur un échantillon FFT
plt.figure(figsize=(10, 6))
plt.plot(original_fft_sample, label="Original FFT", lw=2)
plt.plot(fft_sample_noisy, label="FFT avec bruit", lw=2)
plt.plot(fft_sample_shifted, label="FFT décalé", lw=2)
plt.title("Augmentation spectrale d'un échantillon (FFT)")
plt.xlabel("Index de fréquence")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

# Vous pouvez ensuite intégrer ces nouvelles données dans votre jeu de données global
# (ici, nous utilisons SMOTE sur le jeu de données combiné pour équilibrer les classes)

# ============================
# 7. Séparation en jeu d'entraînement et de test
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# ============================
# 8. Entraînement du modèle Random Forest avec pondération des classes
# ============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# ============================
# 9. Prédictions et évaluation du modèle
# ============================
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy du modèle Random Forest : {accuracy:.2%}\n")
print("📊 Rapport de classification :")
print(classification_report(y_test, y_pred))

# ============================
# 10. Matrice de confusion
# ============================
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.title("Matrice de confusion – Random Forest")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.tight_layout()
plt.show()

# ============================
# 11. Importance des variables (top 20)
# ============================
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20
top_features = X_combined.columns[indices][:top_n]
top_importances = importances[indices][:top_n]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features)
plt.title("Importance des 20 variables les plus discriminantes")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# ============================
# 12. Evaluation de l'accuracy en fonction du nombre d'arbres
# ============================
n_estimators_range = range(10, 210, 10)
cv_scores = []

for n in n_estimators_range:
    model = RandomForestClassifier(n_estimators=n, random_state=42, class_weight='balanced')
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, cv_scores, marker='o', color='green')
plt.title("Accuracy vs. nombre d'arbres (CV 5-fold)")
plt.xlabel("Nombre d'arbres")
plt.ylabel("Accuracy moyenne")
plt.grid(True)
plt.tight_layout()
plt.show()
