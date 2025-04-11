from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset
file_path = "./combined_data.csv"
df = pd.read_csv(file_path)

# Separate features (X) and target (y)
X = df.drop(columns=["class"]).values
y = df["class"].values

# Encode the target variable if necessary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)







# === 1. Chargement des données
df = pd.read_csv("combined_data.csv")
df["label_encoded"] = LabelEncoder().fit_transform(df["class"])
X_raw = df.drop(columns=["class", "label_encoded"])
y_raw = df["label_encoded"]

import matplotlib.pyplot as plt

def plot_class_distribution(y_series, title):
    """
    Plots the distribution of classes in the target variable.

    Parameters:
    - y_series: Pandas Series or array-like, the target variable.
    - title: str, the title of the plot.
    """
    counts = y_series.value_counts().sort_index()
    counts.index = [f"Class {i}" for i in counts.index]
    counts.plot(kind="bar", figsize=(8, 5), edgecolor="black")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
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



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Example test size proportions
splits = [0.2, 0.3, 0.4]

# Iterate through each test size
for split in splits:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=split, random_state=42, stratify=y2)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Define and train the neural network model
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, solver='adam', random_state=42, tol=1e-4)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test size: {split}, Accuracy: {accuracy:.2f}")
    #Why 42? The number 42 is an arbitrary choice and is often used as a reference to the book The Hitchhiker's Guide to the Galaxy, where 42 is "the answer to the ultimate question of life, the universe, and everything." It has become a convention in the programming and data science community.

# Split the dataset with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)









# Call the optimise_pls_cv function
n_components = 20  # Adjust based on your dataset
pls_model = optimise_pls_cv(X_train, y_train, n_components)

def optimise_pls_cv(X2, y2, n_comp, plot_components=True):
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''

    mse = []
    component = np.arange(1, n_comp + 1)

    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation
        y_cv = cross_val_predict(pls, X2, y2, cv=10)

        mse.append(mean_squared_error(y2, y_cv))

        comp = 100 * (i + 1) / n_comp
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin + 1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(left=0)

        plt.show()

    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=msemin + 1)

    # Fit to the entire dataset
    pls_opt.fit(X2, y2)
    y_c = pls_opt.predict(X2)

    # Cross-validation
    y_cv = cross_val_predict(pls_opt, X2, y2, cv=10)

    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y2, y_c)
    score_cv = r2_score(y2, y_cv)

    # Calculate mean squared error for calibration and cross-validation
    mse_c = mean_squared_error(y2, y_c)
    mse_cv = mean_squared_error(y2, y_cv)

    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)

    # Plot regression and figures of merit
    rangey = max(y2) - min(y2)
    rangex = max(y_c) - min(y_c)

    # Fit a line to the CV vs response
    z = np.polyfit(y2.flatten(), y_c.flatten(), 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_c, y2, c='red', edgecolors='k')
        # Plot the best fit line
        ax.plot(np.polyval(z, y2), y2, c='blue', linewidth=1)
        # Plot the ideal 1:1 line
        ax.plot(y2, y2, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): ' + str(score_cv))
        plt.xlabel('Predicted')
        plt.ylabel('Measured')

        plt.show()

    return pls_opt

# Example usage with your dataset
n_components = 20  # Adjust based on your dataset
pls_model = optimise_pls_cv(X_train, y_train, n_components)