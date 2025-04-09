import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# Fonction pour tronquer un colormap (évite les couleurs trop claires)
def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# Charger les données (chemin relatif)
file_path = "./combined_data.csv"
data = pd.read_csv(file_path)

# Identifier les colonnes de spectre et la colonne de classe
spectre_columns = data.columns[:-1]
class_column = data.columns[-1]

# Conversion des longueurs d'onde en float (important pour axe x)
wavelengths = spectre_columns.astype(float)

# Liste des longueurs d'onde importantes et leurs valeurs d'importance
important_wavelengths = [680.07, 693.65, 699.09, 677.36, 688.22, 620.61, 626.0, 639.48, 685.5, 655.69,
                         636.78, 682.79, 674.65, 650.29, 696.37, 701.81, 489.77, 652.99, 803.1, 617.92]
Values_wavelenghts = [0.022255532241553168, 0.020965080901097045, 0.02090325537037073, 0.017536663122307596,
                      0.01644043993703079, 0.016402771108538383, 0.016398563897577503, 0.015907171521087164,
                      0.015633146995902195, 0.015378116754257798, 0.01455190189246996, 0.01452174439347766,
                      0.014341515654522453, 0.013853429889793583, 0.013013380904948852, 0.012580357503329764,
                      0.012282115911053139, 0.0122104278063796, 0.012030822244427643, 0.011810673484187352]

# Filtrer pour ne garder que les 10 premières longueurs d'onde importantes
important_wavelengths = important_wavelengths[:10]
Values_wavelenghts = Values_wavelenghts[:10]

# Colormap rouge tronqué pour éviter les couleurs trop claires
original_cmap = plt.cm.Reds
cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)
norm = Normalize(vmin=min(Values_wavelenghts), vmax=max(Values_wavelenghts))

# Couleurs pour les classes
classes = data[class_column].unique()
colors = plt.cm.tab10(range(len(classes)))

# =======================
# 1. Spectres par classe
# =======================
fig, ax = plt.subplots(figsize=(12, 6))
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for _, row in subset.iterrows():
        ax.plot(wavelengths, row[spectre_columns], color=colors[i], alpha=0.5, label=cls if _ == subset.index[0] else "")

# Lignes verticales colorées selon importance
for i, (wavelength, value) in enumerate(zip(important_wavelengths, Values_wavelenghts)):
    color = cmap(norm(value))
    ax.axvline(x=wavelength, color=color, linestyle='--', alpha=0.9, linewidth=2, label=f'{wavelength:.2f} nm' if i == 0 else "")

# Colorbar liée à l'importance
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label("Importance des longueurs d'onde")

ax.set_title("Spectres par classe")
ax.set_xlabel("Longueur d'onde (nm)")
ax.set_ylabel("Intensité")
ax.legend(title="Classe")
ax.grid(True)
fig.tight_layout()
plt.show()

# =========================
# 2. Dérivées et dérivées secondes
# =========================
plt.figure(figsize=(12, 12))

# ---- Dérivées ----
plt.subplot(2, 1, 1)
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for _, row in subset.iterrows():
        derivative = np.gradient(row[spectre_columns].values.astype(float), wavelengths)
        plt.plot(wavelengths, derivative, color=colors[i], alpha=0.5, label=cls if _ == subset.index[0] else "")
for i, (wavelength, value) in enumerate(zip(important_wavelengths, Values_wavelenghts)):
    color = cmap(norm(value))
    plt.axvline(x=wavelength, color=color, linestyle='--', alpha=0.9, linewidth=2, label=f'{wavelength:.2f} nm' if i == 0 else "")

plt.title("Dérivées des spectres")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Dérivée")
plt.legend(title="Classe")
plt.grid(True)

# ---- Dérivées secondes ----
plt.subplot(2, 1, 2)
for i, cls in enumerate(classes):
    subset = data[data[class_column] == cls]
    for _, row in subset.iterrows():
        second_derivative = np.gradient(np.gradient(row[spectre_columns].values.astype(float), wavelengths), wavelengths)
        plt.plot(wavelengths, second_derivative, color=colors[i], alpha=0.5, label=cls if _ == subset.index[0] else "")
for i, (wavelength, value) in enumerate(zip(important_wavelengths, Values_wavelenghts)):
    color = cmap(norm(value))
    plt.axvline(x=wavelength, color=color, linestyle='--', alpha=0.9, linewidth=2, label=f'{wavelength:.2f} nm' if i == 0 else "")

plt.title("Dérivées secondes des spectres")
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Dérivée seconde")
plt.legend(title="Classe")
plt.grid(True)

plt.tight_layout()
plt.show()