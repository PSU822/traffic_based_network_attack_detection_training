import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from train.models import *

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(y_batch.cpu())
    return accuracy(torch.cat(all_labels), torch.cat(all_preds))

# ===============================================================================================

# path = os.path.join(ROOT_DIR, 'datasets', 'training-set.csv')
path = os.path.join(ROOT_DIR, 'datasets', 'training-data-preprocess.csv')
data = pd.read_csv(path)

# data = data.drop(columns=['id'])
categorical_cols = ['proto', 'service', 'state']
categorical_encoders = {}


for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    categorical_encoders[col] = le

label_encoder = LabelEncoder()
data['attack_cat'] = label_encoder.fit_transform(data['attack_cat'])

X = data.drop(columns=['attack_cat']).values
y = data['attack_cat'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test,dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")

criterion = nn.CrossEntropyLoss()

cnn_model = CNN(input_size=X_train.shape[1], num_classes=len(np.unique(y_train))).to(device)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

train_losses = []
test_accs = []

for epoch in range(50):
    train_loss = train(cnn_model, train_loader, optimizer, criterion, device)
    test_acc = test(cnn_model, test_loader, device)
    
    train_losses.append(train_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}, CNN Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
model_path = os.path.join(CURRENT_DIR, 'cnn_model.pth')
torch.save(cnn_model.state_dict(), model_path)

save_dict = {
    'label_encoder': label_encoder,
    'scaler': scaler,
    'categorical_encoders': categorical_encoders,
}

prep_path = os.path.join(ROOT_DIR, 'train','preprocessing.pkl' )
with open(prep_path, 'wb') as f:
    pickle.dump(save_dict, f)
    
plt.figure(figsize=(12, 4))
    
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accs)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()