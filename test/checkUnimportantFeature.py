import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

path = 'datasets/training-set.csv'
data = pd.read_csv(path)

print(f"원본 shape: {data.shape}\n")

# id 제거
data = data.drop(columns=['id'])

# Categorical 인코딩
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# 레이블 분리
y = data['attack_cat'].values
X = data.drop(columns=['attack_cat'])

feature_names = X.columns.tolist()
print(f"전체 피처 개수: {len(feature_names)}")
print(f"피처 목록: {feature_names}\n")

X = X.values

rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    max_depth=10
)

le_label = LabelEncoder()
y_encoded = le_label.fit_transform(y)

rf.fit(X, y_encoded)

importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("=== Top 20 중요한 피처 ===")
print(importances_df.head(20).to_string(index=False))

print("\n=== Bottom 10 ===")
print(importances_df.tail(10).to_string(index=False))


plt.figure(figsize=(12, 8))


plt.subplot(1, 2, 1)
top_20 = importances_df.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Important Features')
plt.gca().invert_yaxis()


plt.subplot(1, 2, 2)
bottom_15 = importances_df.tail(15)
plt.barh(range(len(bottom_15)), bottom_15['importance'])
plt.yticks(range(len(bottom_15)), bottom_15['feature'])
plt.xlabel('Importance')
plt.title('Bottom 15 Features (Candidates to Drop)')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\n=== 제거 추천 피처 ===")
low_importance = importances_df[importances_df['importance'] < 0.01]
print(low_importance.to_string(index=False))
print(f"\n총 {len(low_importance)}개 피처 제거 고려")
