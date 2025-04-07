import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Chargement et préparation des données ===
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])
X_raw = df.drop(columns=["class", "label_encoded"])
y_raw = df["label_encoded"]

# Ajout de la dérivée 1re
X_deriv = X_raw.diff(axis=1).fillna(0)
X_combined = pd.concat([X_raw, X_deriv.add_suffix("_deriv")], axis=1)

# Standardisation
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_combined), columns=X_combined.columns)

# === 2. Split en train/test pour évaluation finale ===
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

# === 3. Équilibrage du train set ===
df_train = pd.concat([X_train_base, y_train_base], axis=1)
max_count = df_train["label_encoded"].value_counts().max()
balanced_train = []

for label in df_train["label_encoded"].unique():
    class_subset = df_train[df_train["label_encoded"] == label]
    n_to_generate = max_count - len(class_subset)

    if n_to_generate > 0:
        synthetic = class_subset.sample(n=n_to_generate, replace=True, random_state=42)
        noise = np.random.normal(0, 0.05, synthetic.drop(columns=["label_encoded"]).shape)
        synthetic.iloc[:, :-1] += noise
        class_subset = pd.concat([class_subset, synthetic])
    
    balanced_train.append(class_subset)

df_train_bal = pd.concat(balanced_train)
df_train_bal = shuffle(df_train_bal, random_state=42)

X_train = df_train_bal.drop(columns=["label_encoded"])
y_train = df_train_bal["label_encoded"]

# === 4. Fonction d’évaluation en cross-val ===
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

# === 5. Modèle sur labels réels
scores_real = evaluate_model(X_train, y_train, "Réels (cross-val)")

# === 6. Modèle sur labels aléatoires
y_random = shuffle(y_train, random_state=42)
scores_random = evaluate_model(X_train, y_random, "Aléatoires (cross-val)")

# === 7. Modèle sur jeu de test final
model_final = RandomForestClassifier(n_estimators=100, random_state=42)
model_final.fit(X_train, y_train)
y_pred = model_final.predict(X_test_base)

scores_test = pd.Series({
    'Accuracy': accuracy_score(y_test_base, y_pred),
    'Precision': precision_score(y_test_base, y_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_base, y_pred, average='weighted'),
    'F1 Score': f1_score(y_test_base, y_pred, average='weighted'),
}, name="Test final")

# === 8. Résultats globaux
metrics_df = pd.concat([scores_real, scores_random, scores_test], axis=1)
print("\n--- Résumé des performances ---\n")
print(metrics_df)

# === 9. Affichage des résultats
metrics_df.plot(kind="bar", figsize=(10, 6))
plt.title("Comparaison des performances (cross-val vs. test vs. aléatoire)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# === 10. Matrice de confusion sur test
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test_base, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_raw), yticklabels=np.unique(y_raw))
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion (test final)")
plt.tight_layout()
plt.show()
