import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les données depuis un chemin relatif
file_path = "./combined_data.csv"
data = pd.read_csv(file_path)

# Identifier les colonnes des spectres et la colonne de classe
# On suppose que toutes les colonnes sauf la dernière correspondent aux longueurs d'onde
spectre_columns_str = data.columns[:-1]  # noms de colonnes (souvent sous forme de chaînes)
class_column = data.columns[-1]           # la dernière colonne correspond à la classe

# Convertir les noms de colonnes en nombres pour l'axe des x
spectre_columns = [float(w) for w in spectre_columns_str]

# Définir la palette de couleurs pour chaque classe
classes = data[class_column].unique()       # Obtenir les classes uniques
colors = plt.cm.tab10(range(len(classes)))    # Générer des couleurs

# --- Tracé des spectres ---
plt.figure(figsize=(12, 6))
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]  # Filtrer les données pour chaque classe
    for idx, row in subset.iterrows():
        # Utiliser spectre_columns pour l'axe des x et les noms de colonnes pour accéder aux valeurs
        plt.plot(spectre_columns, row[spectre_columns_str], color=colors[i],
                 alpha=0.5, label=cls if idx == subset.index[0] else "")

# Ajouter une ligne verticale à 500 nm
plt.axvline(x=500, color='red', linestyle='--', label='500 nm')

# Personnaliser le graphique
plt.title("Visualisation des spectres par classe")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Intensité")
plt.legend(title="Classe")
plt.grid(True)
# Ajuster les graduations de l'axe des x (10 valeurs réparties uniformément)
ticks = np.linspace(min(spectre_columns), max(spectre_columns), 10)
plt.xticks(ticks=ticks, labels=[f"{tick:.1f}" for tick in ticks], rotation=45, fontsize=10)

plt.show(block=True)

# --- Calcul et tracé des dérivées et dérivées secondes ---
plt.figure(figsize=(12, 12))

# Dérivées (première dérivée)
plt.subplot(2, 1, 1)
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for idx, row in subset.iterrows():
        # Conversion des valeurs en float
        values = row[spectre_columns_str].values.astype(float)
        derivative = np.gradient(values)
        plt.plot(spectre_columns, derivative, color=colors[i],
                 alpha=0.5, label=cls if idx == subset.index[0] else "")
plt.axvline(x=500, color='red', linestyle='--', label='500 nm')
plt.title("Dérivées des spectres par classe")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Dérivée")
plt.legend(title="Classe")
plt.grid(True)
ticks = np.linspace(min(spectre_columns), max(spectre_columns), 10)
plt.xticks(ticks=ticks, labels=[f"{tick:.1f}" for tick in ticks], rotation=45, fontsize=10)

# Dérivées secondes (deuxième dérivée)
plt.subplot(2, 1, 2)
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for idx, row in subset.iterrows():
        values = row[spectre_columns_str].values.astype(float)
        second_derivative = np.gradient(np.gradient(values))
        plt.plot(spectre_columns, second_derivative, color=colors[i],
                 alpha=0.5, label=cls if idx == subset.index[0] else "")
plt.axvline(x=500, color='red', linestyle='--', label='500 nm')
plt.title("Dérivées secondes des spectres par classe")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Dérivée seconde")
plt.legend(title="Classe")
plt.grid(True)
ticks = np.linspace(min(spectre_columns), max(spectre_columns), 10)
plt.xticks(ticks=ticks, labels=[f"{tick:.1f}" for tick in ticks], rotation=45, fontsize=10)

plt.tight_layout()
plt.show()
