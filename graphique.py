#j'aimerais que tu me fasse un code pour faire un graphique pour visualiser mon sspectre avec une couluer par classe
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Import pour calculer les dérivées

# Charger les données
#peut tu ecrire en relative path, se sera plus facile pour un travail ne commun
file_path = "./combined_data.csv"  # Utilisation d'un chemin relatif
data = pd.read_csv(file_path)

# Identifier les colonnes des spectres et de la classe
spectre_columns = data.columns[:-1]  # Toutes les colonnes sauf la dernière
class_column = data.columns[-1]      # La dernière colonne correspond à la classe

# Tracer les spectres
plt.figure(figsize=(12, 6))
classes = data[class_column].unique()  # Obtenir les classes uniques
colors = plt.cm.tab10(range(len(classes)))  # Générer des couleurs pour chaque classe

for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]  # Filtrer les données par classe
    for _, row in subset.iterrows():
        plt.plot(spectre_columns, row[spectre_columns], color=colors[i], alpha=0.5, label=cls if _ == subset.index[0] else "")

# Ajouter une ligne verticale à 500 nm
plt.axvline(x=500, color='red', linestyle='--', label='500 nm')

# Ajouter des légendes et des titres
plt.title("Visualisation des spectres par classe")
plt.xlabel("Longueur d'onde")
plt.ylabel("Intensité")
plt.legend(title="Classe")
plt.grid(True)
plt.xticks(ticks=np.linspace(0, len(spectre_columns) - 1, 10, dtype=int),  # Limiter à 10 valeurs
           labels=np.array(spectre_columns)[np.linspace(0, len(spectre_columns) - 1, 10, dtype=int)], 
           rotation=45, fontsize=10)

# Afficher le graphique
plt.show(block=True)  # Ajout de block=True pour garantir l'affichage

#ajoute moi une représenattion graphique des dérivés et des dérivées seconde

# Calculer et tracer les dérivées et dérivées secondes
plt.figure(figsize=(12, 12))

# Dérivées
plt.subplot(2, 1, 1)
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for _, row in subset.iterrows():
        derivative = np.gradient(row[spectre_columns].values.astype(float))
        plt.plot(spectre_columns, derivative, color=colors[i], alpha=0.5, label=cls if _ == subset.index[0] else "")
# Ajouter une ligne verticale à 500 nm
plt.axvline(x=500, color='red', linestyle='--', label='500 nm')
plt.title("Dérivées des spectres par classe")
plt.xlabel("Longueur d'onde")
plt.ylabel("Dérivée")
plt.legend(title="Classe")
plt.grid(True)
plt.xticks(ticks=np.linspace(0, len(spectre_columns) - 1, 10, dtype=int),  # Limiter à 10 valeurs
           labels=np.array(spectre_columns)[np.linspace(0, len(spectre_columns) - 1, 10, dtype=int)], 
           rotation=45, fontsize=10)

# Dérivées secondes
plt.subplot(2, 1, 2)
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for _, row in subset.iterrows():
        second_derivative = np.gradient(np.gradient(row[spectre_columns].values.astype(float)))
        plt.plot(spectre_columns, second_derivative, color=colors[i], alpha=0.5, label=cls if _ == subset.index[0] else "")
# Ajouter une ligne verticale à 500 nm
plt.axvline(x=500, color='red', linestyle='--', label='500 nm')
plt.title("Dérivées secondes des spectres par classe")
plt.xlabel("Longueur d'onde")
plt.ylabel("Dérivée seconde")
plt.legend(title="Classe")
plt.grid(True)
plt.xticks(ticks=np.linspace(0, len(spectre_columns) - 1, 10, dtype=int),  # Limiter à 10 valeurs
           labels=np.array(spectre_columns)[np.linspace(0, len(spectre_columns) - 1, 10, dtype=int)], 
           rotation=45, fontsize=10)

plt.tight_layout()
plt.show()
