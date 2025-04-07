import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.fft import fft
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# --- Chargement et préparation des données ---
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])
X_raw = df.drop(columns=["class", "label_encoded"])
y = df["label_encoded"]

# Standardisation des données brutes
scaler_raw = StandardScaler()
X_raw_scaled = scaler_raw.fit_transform(X_raw)

# === 1. PCA sur les données brutes ===
pca_raw = PCA(n_components=2)
X_raw_pca = pca_raw.fit_transform(X_raw_scaled)

plt.figure()
plt.scatter(X_raw_pca[:, 0], X_raw_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA sur données brutes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# === 2. PCA sur les données transformées par FFT ===
# Transformation FFT sur chaque ligne et conservation de la magnitude
X_fft = np.abs(fft(X_raw_scaled, axis=1))
scaler_fft = StandardScaler()
X_fft_scaled = scaler_fft.fit_transform(X_fft)

pca_fft = PCA(n_components=2)
X_fft_pca = pca_fft.fit_transform(X_fft_scaled)

plt.figure()
plt.scatter(X_fft_pca[:, 0], X_fft_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA sur données transformées par FFT")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# === 3. PCA sur les données transformées par Hilbert ===
# Transformation Hilbert sur chaque ligne et conservation de la magnitude
X_hilbert = np.abs(hilbert(X_raw_scaled, axis=1))
scaler_hilbert = StandardScaler()
X_hilbert_scaled = scaler_hilbert.fit_transform(X_hilbert)

pca_hilbert = PCA(n_components=2)
X_hilbert_pca = pca_hilbert.fit_transform(X_hilbert_scaled)

plt.figure()
plt.scatter(X_hilbert_pca[:, 0], X_hilbert_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA sur données transformées par Hilbert")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# === 4. PCA sur les données dérivées ===
# Calcul de la première dérivée numérique sur chaque ligne
X_deriv = np.gradient(X_raw_scaled, axis=1)
scaler_deriv = StandardScaler()
X_deriv_scaled = scaler_deriv.fit_transform(X_deriv)

pca_deriv = PCA(n_components=2)
X_deriv_pca = pca_deriv.fit_transform(X_deriv_scaled)

plt.figure()
plt.scatter(X_deriv_pca[:, 0], X_deriv_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA sur données dérivées")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# === 5. PCA sur la concaténation de toutes les représentations ===
# Concaténation des données brutes, FFT, Hilbert et dérivée
X_concat = np.concatenate((X_raw_scaled, X_fft_scaled, X_hilbert_scaled, X_deriv_scaled), axis=1)
pca_concat = PCA(n_components=2)
X_concat_pca = pca_concat.fit_transform(X_concat)

plt.figure()
plt.scatter(X_concat_pca[:, 0], X_concat_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("PCA sur données concaténées (Brutes + FFT + Hilbert + Dérivée)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()
