import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.fft import fft
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

# --- Chargement et préparation des données ---
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])
X_raw = df.drop(columns=["class", "label_encoded"])
y = df["label_encoded"]

# Standardisation des données brutes
scaler_raw = StandardScaler()
X_raw_scaled = scaler_raw.fit_transform(X_raw)



# === 4. PCA sur les données dérivées ===
# Calcul de la première dérivée numérique sur chaque ligne
X_deriv = np.gradient(X_raw_scaled, axis=1)
scaler_deriv = StandardScaler()
X_deriv_scaled = scaler_deriv.fit_transform(X_deriv)

pca_deriv = PCA(n_components=2)
X_deriv_pca = pca_deriv.fit_transform(X_deriv_scaled)

# Récupérer les noms des classes originales
class_names = LabelEncoder().fit(df["class"]).classes_

# Define cluster labels and colors based on existing class names
unique_classes = np.unique(y)
cluster_labels = [class_names[cls] for cls in unique_classes]
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

# Plot PCA with class-guided clusters and ellipses
plt.figure()
for i, (cls, color) in enumerate(zip(unique_classes, colors)):
    cluster_points = X_deriv_pca[y == cls]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=cluster_labels[i], color=color, alpha=0.7)
    
    # Add ellipses for each class
    if len(cluster_points) > 1:  # Ensure enough points for covariance
        cov = np.cov(cluster_points, rowvar=False)
        mean = cluster_points.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color, facecolor='none', linewidth=2)
        plt.gca().add_patch(ellipse)

plt.title("PCA sur les dérivées")
plt.xlabel("Dim1 (PC1)")
plt.ylabel("Dim2 (PC2)")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.legend(title="Classes", loc="upper left", ncol=2)  # Display legend on two lines
plt.show()


