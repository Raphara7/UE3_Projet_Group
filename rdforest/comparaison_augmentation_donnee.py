import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def plot_class_distribution(y_series, title):
    counts = y_series.value_counts().sort_index()
    counts.index = [f"Classe {i}" for i in counts.index]
    counts.plot(kind="bar", figsize=(8, 5), edgecolor="black")
    plt.title(title)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# === 1. Chargement des données
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])
X_raw = df.drop(columns=["class", "label_encoded"])
y_raw = df["label_encoded"]

# Affichage initial des classes
plot_class_distribution(y_raw, "Distribution des classes - Original")

# === 2. Standardisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# === 3. Équilibrage par bruit léger sur classes minoritaires
df_all = pd.concat([X_scaled, y_raw], axis=1)
max_count = df_all["label_encoded"].value_counts().max()
balanced_data = []

for label in df_all["label_encoded"].unique():
    class_subset = df_all[df_all["label_encoded"] == label]
    n_to_generate = max_count - len(class_subset)
    
    if n_to_generate > 0:
        synthetic_samples = class_subset.sample(n=n_to_generate, replace=True, random_state=42)
        noise = np.random.normal(0, 0.05, synthetic_samples.drop(columns=["label_encoded"]).shape)
        synthetic_samples.iloc[:, :-1] += noise
        class_subset = pd.concat([class_subset, synthetic_samples])
    
    balanced_data.append(class_subset)

df_balanced = pd.concat(balanced_data)
df_balanced = shuffle(df_balanced, random_state=42)

X1 = df_balanced.drop(columns=["label_encoded"])
y = df_balanced["label_encoded"]

# Affichage après équilibrage
plot_class_distribution(y, "Distribution des classes - Après équilibrage")

# === 4. Création du jeu bruité X2 à partir de X1
noise = np.random.normal(0, 0.05, X1.shape)
X1_noisy = X1 + noise
X2 = pd.concat([X1, X1_noisy])
y2 = pd.concat([y, y])

# Affichage après ajout de bruit à tout le jeu équilibré
plot_class_distribution(y2, "Distribution des classes - Après duplication avec bruit (X2)")

# === 5. Fonction de validation croisée
def evaluate_model(X, y, name):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scoring = {
        'Accuracy': make_scorer(accuracy_score),
        'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'Recall': make_scorer(recall_score, average='weighted'),
        'F1 Score': make_scorer(f1_score, average='weighted')
    }
    scores = {metric: cross_val_score(model, X, y, cv=5, scoring=scoring[metric]).mean()
              for metric in scoring}
    return pd.Series(scores, name=name)

# === 6. Évaluation des modèles
scores_1 = evaluate_model(X1, y, "Brutes équilibrées")
scores_2 = evaluate_model(X2, y2, "Équilibrées + bruit (doublées)")

metrics_df = pd.concat([scores_1, scores_2], axis=1)
print(metrics_df)

# === 7. Affichage des performances
colors = ["#4C72B0", "#55A868"]
metrics_df.T.plot(kind="bar", stacked=False, figsize=(10, 6), color=colors)
plt.title("Scores en validation croisée (5-fold) selon les types de données")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.legend(title="Métrique", loc="lower right")
plt.tight_layout()
plt.show()
