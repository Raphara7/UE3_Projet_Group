import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

# === Chargement et prétraitement des données de base (hors seed) ===
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

# === Fonction d'entraînement et d'évaluation ===
def train_and_evaluate(seed):
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

    acc = accuracy_score(y_test_base, y_pred)
    return acc

# === Calcul de l’accuracy pour 100 seeds ===
accuracies = []
for seed in range(100):
    acc = train_and_evaluate(seed)
    accuracies.append(acc)

# Reshape pour la heatmap (10x10)
acc_matrix = np.array(accuracies).reshape(10, 10)

# === Affichage heatmap ===
plt.figure(figsize=(8, 6))
sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Accuracy'})
plt.title("Accuracy du test final pour les 100 premières seeds (0 à 99)")
plt.xlabel("Colonne (seed % 10)")
plt.ylabel("Ligne (seed // 10)")
plt.tight_layout()
plt.show()
