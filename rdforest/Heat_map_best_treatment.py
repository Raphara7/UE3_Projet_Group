import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.fft import fft
from scipy.signal import hilbert

# Gestion du support wavelet
try:
    from scipy.signal import cwt, ricker
    WAVELET_AVAILABLE = True
except ImportError:
    print("⚠️ Wavelet non disponible, elle sera ignorée.")
    WAVELET_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Chargement des données ===
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])  # Encode labels in a non-ordered way

X_raw = df.drop(columns=["class", "label_encoded"])
y = df["label_encoded"]

scaler = StandardScaler()

# === 2. Définitions des transformations ===
def brute(X): return X.copy()

def deriv1(X): return X.diff(axis=1).fillna(0)

def deriv2(X): return deriv1(X).diff(axis=1).fillna(0)

def pca(X):
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(PCA(n_components=0.95).fit_transform(X_scaled))

def fft_amp(X):
    return pd.DataFrame(X.apply(lambda row: np.abs(fft(row.to_numpy())), axis=1).tolist())

def wavelet(X):
    widths = np.arange(1, 31)
    X_cwt = X.apply(lambda row: cwt(row.to_numpy(), ricker, widths).flatten(), axis=1)
    return pd.DataFrame(np.vstack(X_cwt))

def hilbert_env(X):
    return pd.DataFrame(X.apply(lambda row: np.abs(hilbert(row.to_numpy())), axis=1).tolist())

# === 3. Dictionnaire des méthodes ===
transformations = {
    "Brute": brute,
    "Dérivée 1": deriv1,
    "Dérivée 2": deriv2,
    "PCA": pca,
    "FFT": fft_amp,
    "Hilbert": hilbert_env
}

if WAVELET_AVAILABLE:
    transformations["Wavelet"] = wavelet

# === 4. Matrice des scores ===
methods = list(transformations.keys())
results = pd.DataFrame(index=methods, columns=methods)

# === 5. Évaluation des combinaisons avec hyperparamètres optimisés ===
for i, name_i in enumerate(methods):
    for j, name_j in enumerate(methods):
        print(f"🧪 Évaluation : {name_i} → {name_j}")
        try:
            X_trans = transformations[name_i](X_raw)
            if i != j:
                X_trans = transformations[name_j](X_trans)

            X_final = scaler.fit_transform(X_trans)
            X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

            # 💡 Utilisation des meilleurs hyperparamètres
            model = RandomForestClassifier(
                criterion='gini',
                max_depth=None,
                max_features='sqrt',
                min_samples_leaf=1,
                min_samples_split=10,
                n_estimators=300,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
        except Exception as e:
            print(f"❌ Erreur avec {name_i} → {name_j} : {e}")
            acc = np.nan

        results.loc[name_i, name_j] = acc

# === 6. Heatmap verte avec labels en haut ===
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    results.astype(float),
    annot=True,
    fmt=".2f",
    cmap="Greens",         # 🌿 Dégradé vert clair → foncé
    linewidths=0.5,
    cbar_kws={'label': 'Accuracy'}
)
ax.set_title("🎯 Accuracy des combinaisons de transformations (optimisé)", fontsize=14)
ax.xaxis.set_ticks_position('top')       # place les ticks en haut
ax.xaxis.set_label_position('top')       # place le label de l'axe en haut
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# === 7. Export CSV (optionnel) ===
results.to_csv("matrice_accuracies_optimized.csv")
print("✅ Résultats enregistrés dans 'matrice_accuracies_optimized.csv'")
