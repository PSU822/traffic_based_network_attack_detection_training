import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

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
from train.makeGraph import *

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def train(model, A_hat, train_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output, attention_weights = model(X_batch, A_hat)  
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def test(model, A_hat, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output, _ = model(X_batch, A_hat)
            _, predicted = torch.max(output, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(y_batch)
    
    test_acc = accuracy(
        torch.cat(all_labels), 
        torch.cat(all_preds)
    )
    return test_acc

def visualize_attention(model, edge_index, sample_X, feature_names, device):
    model.eval()
    with torch.no_grad():
        sample_X = sample_X.to(device)
        _, attention_weights = model(sample_X, edge_index)
        
        edge_idx, attn = attention_weights
        attn_mean = attn.mean(dim=1)
        
        print("\n" + "="*70)
        print("가장 중요한 피처 관계 TOP 20")
        print("="*70)
        
        top_k = min(20, len(attn_mean))
        top_indices = torch.topk(attn_mean, top_k).indices
        
        for rank, idx in enumerate(top_indices, 1):
            src = edge_idx[0][idx].item()
            dst = edge_idx[1][idx].item()
            avg_attn = attn_mean[idx].item()
            
            src_name = feature_names[src] if src < len(feature_names) else f"F{src}"
            dst_name = feature_names[dst] if dst < len(feature_names) else f"F{dst}"
            
            print(f"{rank:2d}. {src_name:20s} → {dst_name:20s}: {avg_attn:.4f}")
        
        print("="*70 + "\n")

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

feature_names = [col for col in data.columns
                 if col != 'attack_cat']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test,dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device : {device}")
'''
gcn_data_train = CreateSampleGraph(X=X_train, Y=y_train, featureNames=feature_names, attackFeatures=attack_features)
gcn_data_test = CreateSampleGraph(X=X_test, Y=y_test, featureNames=feature_names, attackFeatures=attack_features)
'''
A_hat = CreateFeatureGraph(feature_names, attack_features).to(device)
print("피처 그래프 생성 완료")

print(f"A_hat shape: {A_hat.shape}")
print(f"A_hat sum: {A_hat.sum()}")
print(f"A_hat 샘플:\n{A_hat[:5, :5]}")

classes = np.unique(y_train)
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)

class_weights_tensor = torch.tensor(
    class_weights, 
    dtype=torch.float32
).to(device)

for i, cls in enumerate(label_encoder.classes_):
    weight = class_weights_tensor[i].item()
    count = (y_train == i).sum()
    print(f"{cls}: weight={weight:.2f}, count={count}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print("가중치 적용 완료")

gnn_model = GAN(num_feature=len(feature_names), hidden_size=128, num_classes=len(np.unique(y_train))).to(device)
optimizer = optim.Adam(gnn_model.parameters(), lr=0.0001)

A_hat_matrix = CreateFeatureGraph(feature_names, attack_features)
print(f"A_hat shape: {A_hat_matrix.shape}")

edge_list = []
num_features = A_hat_matrix.shape[0]

for i in range(num_features):
    for j in range(num_features):
        if A_hat_matrix[i, j] > 0:
            edge_list.append([i, j])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
print(f"edge_index shape: {edge_index.shape}")  


train_losses = []
test_accs = []

for epoch in range(50):
    train_loss = train(gnn_model, edge_index, train_loader, optimizer, criterion, device)
    test_acc = test(gnn_model, edge_index, test_loader, device)
    
    train_losses.append(train_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}, GAN Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
    
model_path = os.path.join(CURRENT_DIR, 'gnn_model.pth')
torch.save(gnn_model.state_dict(), model_path)

save_dict = {
    'label_encoder': label_encoder,
    'scaler': scaler,
    'categorical_encoders': categorical_encoders,
}
sample = next(iter(test_loader))[0][:1]
visualize_attention(gnn_model,edge_index,sample,feature_names ,device)

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