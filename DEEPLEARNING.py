# Import necessary libraries
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

# Load the wine dataset from UCI repository
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
                'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
wine_data = pd.read_csv(url, header=None, names=column_names)

# Display the first few rows of the dataset
wine_data.head()
print(wine_data.head())

# Split the data into features and labels
X = wine_data.drop('Class', axis=1)
y = wine_data['Class']
# Adjust class labels to be in the range [0, C-1]
y = y - 1
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Create PyTorch datasets and dataloaders
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = data.TensorDataset(X_test_tensor, y_test_tensor)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Define the Dense Network model using PyTorch without class definition
input_size = X_train.shape[1]
hidden_size = 100
output_size = len(y.unique())

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)#,
    #nn.Softmax(dim=1) # Softmax is not needed in the final layer for CrossEntropyLoss
    # as it applies softmax internally
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

        
