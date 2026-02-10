import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

def CreateSampleGraph(X, Y, featureNames, attackFeatures, k=10):
    

    n = len(X)
    edgeList = []
    
    for attack, features in attackFeatures.items():
        print(f"\n{attack}: {features}")
        
        featureIndices = [featureNames.index(f) for f in features]
        X_subset = X[:, featureIndices]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(X_scaled)
        distances, indices = nbrs.kneighbors(X_scaled)
        
        for i in range(n):
            for j in indices[i][1:]:
                edgeList.append([i, j])
        
    
    edgeList = list(set(map(tuple, edgeList)))
    edgeList = [list(e) for e in edgeList]
    
    edgeIndex = torch.tensor(edgeList, dtype=torch.long).t()
    
    graph = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edgeIndex,
        y=torch.tensor(Y, dtype=torch.long)
    )
    
    print(f"노드: {graph.num_nodes:,}개")
    print(f"엣지: {graph.num_edges:,}개")
    
    return graph


def CreateFeatureGraph(featureName, attackFeature):
    F = len(featureName)
    nameToIdx = {n: i for i, n in enumerate(featureName)}
    
    A = np.zeros((F,F), dtype=float)
    for attack, feats in attackFeature.items():
        for i, fa in enumerate(feats):
            for j, fb in enumerate(feats):
                if i != j and fa in nameToIdx and fb in nameToIdx:
                    ia = nameToIdx[fa]
                    ib = nameToIdx[fb]
                    A[ia, ib] = 1.0
    A = A + np.eye(F)
    
    deg = A.sum(axis=1)
    degInvSqrt = np.power(deg + 1e-10, -0.5)
    DInvSqrt = np.diag(degInvSqrt)
    A_hat = DInvSqrt @ A @ DInvSqrt
    
    return torch.tensor(A_hat, dtype=torch.float32)