# # Use Case

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ## Load and Preprocess Data
# 📂 Chemin d'accès vers le fichier CSV
file_path = ".\combined_data.csv"

# 🧪 Chargement des données
df = pd.read_csv(file_path)

# Charger les données avec des options pour gérer les valeurs manquantes
data = pd.read_csv(file_path, header=0, na_values=["", " ", "NA", "null"])


# Afficher un aperçu des données
print("Aperçu des données :")
print(data.head())



# Display the first few rows of the dataset
data.head()

# Split the data into features and labels
X = data.drop('class', axis=1)
y = data['class']


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



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Example test size proportions
splits = [0.2, 0.3, 0.4]

# Iterate through each test size
for split in splits:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=split, random_state=42, stratify=y)
    
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


# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Encode the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert features and labels to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ## Perceptron Model with scikit-learn
# Implement a Perceptron model using scikit-learn, train it on the wine dataset, and evaluate its performance.

# Implement a Perceptron model using scikit-learn
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Train the Perceptron model
perceptron.fit(X_train, y_train)

# Evaluate the Perceptron model
train_accuracy = perceptron.score(X_train, y_train)
test_accuracy = perceptron.score(X_test, y_test)

print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

# Predictions
y_pred_perceptron = perceptron.predict(X_test)

# Confusion Matrix for Dense Network
conf_matrix_perceptron = confusion_matrix(y_test, y_pred_perceptron)

print(conf_matrix_perceptron)

# ## Multilayer Perceptron with scikit-learn
# Implement a Multilayer Perceptron (MLP) using scikit-learn, train it on the wine dataset, and evaluate its performance.

# Implement a Multilayer Perceptron (MLP) using scikit-learn
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001,
                    solver='adam', random_state=42, tol=1e-4)

# Train the MLP model
mlp.fit(X_train, y_train)

# Evaluate the MLP model
train_accuracy_mlp = mlp.score(X_train, y_train)
test_accuracy_mlp = mlp.score(X_test, y_test)

print(f'MLP Training Accuracy: {train_accuracy_mlp:.2f}')
print(f'MLP Test Accuracy: {test_accuracy_mlp:.2f}')

# Predictions
y_pred_mlp = mlp.predict(X_test)

# Confusion Matrix for MPL
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)

print(conf_matrix_mlp)

# ## Dense Network with PyTorch
# Implement a Dense Network using PyTorch, train it on the wine dataset, and evaluate its performance.

# Define the Dense Network model using PyTorch without class definition
input_size = X_train.shape[1]
qqhidden_size = 100
output_size = len(y.unique())

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.Softmax(dim=1)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 300

# Train the Dense Network model
losses = []  # List to store loss values
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())  # Store the loss value
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the Dense Network model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    test_accuracy_dense = correct / total
    print(f'Dense Network Test Accuracy: {test_accuracy_dense:.2f}')

# Use the DataLoader to iterate on test data
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Passez les données à travers le modèle pour obtenir les probabilités
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion Matrix for Dense Network
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot of the Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Dense Network')
plt.show()        
        
# Individual class probabilities
print(outputs)

# Plot the training loss
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# ## Appendix : Class definition

# Define the Dense Network model using PyTorch
class DenseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenseNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# Set the parameters for the model
input_size = X_train.shape[1]
hidden_size = 100
output_size = len(y.unique())
learning_rate = 0.001
num_epochs = 300

# Initialize the model, loss function, and optimizer
model = DenseNetwork(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import torch
import torch.nn as nn
import torch.optim as optim

# Définir la classe DenseNetwork
class DenseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DenseNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        # Pas besoin d'ajouter nn.Softmax ici, car CrossEntropyLoss le gère automatiquement

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)  # Sorties brutes (logits)
        return x

# Initialiser les paramètres du modèle
input_size = X_train.shape[1]  # Nombre de caractéristiques
hidden_size = 100  # Nombre de neurones dans la couche cachée
output_size = len(y.unique())  # Nombre de classes
learning_rate = 0.001
num_epochs = 300

# Initialiser le modèle, la fonction de perte et l'optimiseur
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNetwork(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()  # Fonction de perte
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entraîner le modèle
losses = []  # Liste pour stocker les valeurs de perte
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        # Déplacer les données sur le bon appareil
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Passer les données à travers le modèle
        outputs = model(batch_X)
        
        # Calculer la perte
        loss = criterion(outputs, batch_y)
        
        # Rétropropagation
        loss.backward()
        
        # Mettre à jour les poids
        optimizer.step()
    
    # Stocker la perte pour chaque époque
    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Afficher la courbe de perte
import matplotlib.pyplot as plt
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Evaluate the Dense Network model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    test_accuracy_dense = correct / total
    print(f'Dense Network Test Accuracy: {test_accuracy_dense:.2f}')

# Plot the training loss
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
