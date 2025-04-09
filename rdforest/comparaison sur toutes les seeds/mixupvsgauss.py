import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

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

# Standardisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)

# === Mixup augmentation ===
def mixup_augmentation(X, y, n_samples, alpha=0.2, seed=None, 
                       gaussian_std=0.01, jitter_factor=0.02, prob=0.5):
    np.random.seed(seed)
    X = X.to_numpy()
    y = y.to_numpy()
    
    idx1 = np.random.randint(0, len(X), size=n_samples)
    idx2 = np.random.randint(0, len(X), size=n_samples)
    
    lam = np.random.beta(alpha, alpha, size=n_samples).reshape(-1, 1)
    X_mix = lam * X[idx1] + (1 - lam) * X[idx2]
    
    # === Ajout de bruit gaussien et jittering avec probabilité ===
    if np.random.rand() < prob:
        noise = np.random.normal(0, gaussian_std, X_mix.shape)
        jitter = np.random.uniform(1 - jitter_factor, 1 + jitter_factor, X_mix.shape)
        X_mix = (X_mix + noise) * jitter
    
    y_mix = y[idx1]
    return pd.DataFrame(X_mix, columns=X_scaled.columns), pd.Series(y_mix, name="label_encoded")

# === Version bruit gaussien ===
def train_and_evaluate_noise(seed):
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=seed
    )
    df_train = pd.concat([X_train_base, y_train_base], axis=1)
    max_count = df_train["label_encoded"].value_counts().max()
    balanced_train = []
    for label in df_train["label_encoded"].unique():
        class_subset = df_train[df_train["label_encoded"] == label]
        n_to_generate = max_count - len(class_subset)
        if n_to_generate > 0:
            synthetic = class_subset.sample(n=n_to_generate, replace=True, random_state=seed)
            noise = np.random.normal(0, 0.05, synthetic.drop(columns=["label_encoded"]).shape)
            synthetic.iloc[:, :-1] += noise
            class_subset = pd.concat([class_subset, synthetic])
        balanced_train.append(class_subset)
    df_train_bal = pd.concat(balanced_train)
    df_train_bal = shuffle(df_train_bal, random_state=seed)
    X_train = df_train_bal.drop(columns=["label_encoded"])
    y_train = df_train_bal["label_encoded"]
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_base)
    return accuracy_score(y_test_base, y_pred)

# === Version Mixup ===
def train_and_evaluate_mixup(seed):
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=seed
    )
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
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_base)
    return accuracy_score(y_test_base, y_pred)

# === Comparaison sur 100 seeds ===
acc_noise = []
acc_mixup = []

for seed in range(100):
    acc_noise.append(train_and_evaluate_noise(seed))
    acc_mixup.append(train_and_evaluate_mixup(seed))

# Matrices
acc_noise_matrix = np.array(acc_noise).reshape(10, 10)
acc_mixup_matrix = np.array(acc_mixup).reshape(10, 10)
acc_diff_matrix = acc_mixup_matrix - acc_noise_matrix

# === Affichage heatmap des différences ===
plt.figure(figsize=(8, 6))
sns.heatmap(acc_diff_matrix, annot=True, fmt=".2f", center=0, cmap="coolwarm", cbar_kws={'label': 'Mixup - Gaussien'})
plt.title("Différence d'accuracy (Mixup – Bruit gaussien) sur 100 seeds")
plt.xlabel("Colonne (seed % 10)")
plt.ylabel("Ligne (seed // 10)")
plt.tight_layout()
plt.show()
