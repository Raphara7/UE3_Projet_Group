import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Chargement des données
df = pd.read_csv("combined_data.csv")

# Encodage des classes si nécessaire
if "label_encoded" not in df.columns:
    df["label_encoded"] = LabelEncoder().fit_transform(df["class"])

# Comptage des classes
class_counts = df["class"].value_counts()

# Affichage
plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar", color="#4C72B0", edgecolor="black")
plt.title("Nombre d'exemples par classe")
plt.xlabel("Classe")
plt.ylabel("Nombre d'exemples")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
