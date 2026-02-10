import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

# HyperParams
DROPOUT = 0.4

# v1
attack_features_ = {
    'Generic': ['ct_dst_sport_ltm', 'sload', 'sbytes'],
    'Exploits': ['dttl', 'ct_srv_dst', 'ackdat'],
    'Fuzzers': ['sttl', 'ct_state_ttl', 'dload'],
    
    'Reconnaissance': ['ct_srv_dst', 'sttl', 'ct_dst_src_ltm'],
    'DoS': ['sload', 'rate', 'spkts'],
    'Analysis': ['proto', 'dmean'],
    'Backdoor': ['service', 'dur', 'smean'],
    'Shellcode': ['proto', 'service', 'ct_src_ltm'],
    'Worms': ['smean', 'sbytes', 'sinpkt']
}
# v1.5
attack_features__ = {
    # 통계 기반 (그대로 유지, 1개만 추가)
    'Generic': ['ct_dst_sport_ltm', 'sload', 'sbytes', 'ct_dst_src_ltm'], 
    'Exploits': ['dttl', 'ct_srv_dst', 'ackdat', 'tcprtt'], 
    'Fuzzers': ['sttl', 'ct_state_ttl', 'dload', 'dmean'],  
    'Reconnaissance': ['ct_srv_dst', 'sttl', 'ct_dst_src_ltm', 'smean'], 
    
    # 소수 클래스
    'DoS': ['sload', 'rate', 'spkts', 'ct_srv_dst'], 
    'Analysis': ['proto', 'dmean', 'dbytes', 'service'],
    'Backdoor': ['service', 'dur', 'smean', 'proto'],  
    'Shellcode': ['proto', 'service', 'ct_src_ltm', 'ct_dst_src_ltm'], 
    'Worms': ['smean', 'sbytes', 'sinpkt', 'sload']  
}

# v2
attack_features = {
    'Generic': ['ct_dst_sport_ltm', 'sload', 'sbytes', 'response_body_len'], 
    'Exploits': ['dttl', 'ct_srv_dst', 'ackdat', 'response_body_len'], 
    'Fuzzers': ['sttl', 'ct_state_ttl', 'dload', 'dmean'],  
    'Reconnaissance': ['ct_srv_dst', 'sttl', 'ct_dst_src_ltm', 'is_sm_ips_ports'], 
    
    'DoS': ['sload', 'rate', 'spkts', 'swin'], 
    'Analysis': ['proto', 'dmean', 'dbytes', 'response_body_len'],
    'Backdoor': ['service', 'dur', 'smean', 'is_sm_ips_ports'],  
    'Shellcode': ['proto', 'service', 'ct_src_ltm', 'stcpb'], 
    'Worms': ['smean', 'sbytes', 'sinpkt', 'dwin']  
}


class GCN(nn.Module):
    def __init__(self, num_feature, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.feature_embad = nn.Parameter(torch.randn(num_feature, hidden_size))
        
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x, A_hat):
        h = self.feature_embad
        h = torch.matmul(A_hat, h)
        h = F.relu(h)
        
        sample_repr = torch.matmul(x, h) 
        
        out = F.relu(self.fc1(sample_repr))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
class GAN(nn.Module):
    def __init__(self, num_feature, hidden_size, num_classes):
        super(GAN, self).__init__()
        self.feature_embad = nn.Parameter(torch.randn(num_feature, hidden_size))
        
        
        self.gat1 = GATConv(
            hidden_size, 
            hidden_size,
            heads=4
        )
        
        self.fc1 = nn.Linear(hidden_size * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, x, edge_index):
        h = self.feature_embad
        h, attention_weights = self.gat1(h, edge_index, return_attention_weights=True)
        h = F.elu(h)
        
        sample_repr = torch.matmul(x, h) 
        
        out = F.relu(self.fc1(sample_repr))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, attention_weights
    
class CNN(nn.Module):
    def __init__(self, input_size, num_classes, dropout= 0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        self.fc1 = nn.Linear(128 * (input_size // 4), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x