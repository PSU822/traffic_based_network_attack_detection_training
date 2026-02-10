'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# ========== 데이터 로드 ==========
path = 'datasets/training-set.csv'
data = pd.read_csv(path)

print(f"원본 shape: {data.shape}\n")

# ========== 전처리 ==========
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

# 피처명 저장 (중요!)
feature_names = X.columns.tolist()
print(f"전체 피처 개수: {len(feature_names)}")
print(f"피처 목록: {feature_names}\n")

X = X.values

# ========== Random Forest로 Feature Importance 분석 ==========
print("=== Feature Importance 분석 중, 냥냥! ===\n")

rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    max_depth=10
)

# Label도 인코딩
le_label = LabelEncoder()
y_encoded = le_label.fit_transform(y)

rf.fit(X, y_encoded)

# ========== 결과 정리 ==========
importances_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("=== Top 20 중요한 피처, 냥! ===")
print(importances_df.head(20).to_string(index=False))

print("\n=== Bottom 10 덜 중요한 피처, 냥! ===")
print(importances_df.tail(10).to_string(index=False))

# ========== 시각화 ==========
plt.figure(figsize=(12, 8))

# 상위 20개
plt.subplot(1, 2, 1)
top_20 = importances_df.head(20)
plt.barh(range(len(top_20)), top_20['importance'])
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Important Features')
plt.gca().invert_yaxis()

# 하위 15개
plt.subplot(1, 2, 2)
bottom_15 = importances_df.tail(15)
plt.barh(range(len(bottom_15)), bottom_15['importance'])
plt.yticks(range(len(bottom_15)), bottom_15['feature'])
plt.xlabel('Importance')
plt.title('Bottom 15 Features (Candidates to Drop)')
plt.gca().invert_yaxis()

plt.tight_layout()

# ========== 제거 후보 제안 ==========
print("\n=== 제거 추천 피처 (importance < 0.01), 냥냥! ===")
low_importance = importances_df[importances_df['importance'] < 0.01]
print(low_importance.to_string(index=False))
print(f"\n총 {len(low_importance)}개 피처 제거 고려, 냥!")

print("\n분석 완료, 냥냥! 🎉")


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ========== 데이터 로드 및 전처리 ==========
path = 'datasets/training-data-preprocess.csv'
data = pd.read_csv(path)

# id 제거, categorical 인코딩
# data = data.drop(columns=['id'])
categorical_cols = ['proto', 'service', 'state']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# 피처명 (제거된 10개 빼고)
drop_features = [
    'is_ftp_login', 'ct_ftp_cmd', 'dwin', 'dtcpb', 'stcpb',
    'response_body_len', 'ct_flw_http_mthd', 'trans_depth',
    'is_sm_ips_ports', 'swin']

feature_names = [col for col in data.columns 
                 if col not in ['attack_cat']]

print(f"사용 피처 개수: {len(feature_names)}")
print(f"피처 목록: {feature_names}\n")

# ========== 공격별 샘플 수 확인 ==========
print("=" * 60)
print("=== 공격 유형별 샘플 분포, 냥! ===")
print("=" * 60)

attack_counts = data['attack_cat'].value_counts()
print(attack_counts)
print(f"\n전체: {len(data)}개")
print(f"공격 유형: {len(attack_counts)}개\n")

# ========== 전체 공격 유형 분석 ==========
all_attacks = attack_counts.index.tolist()
normal_count = attack_counts['Normal']

results = {}

for attack in all_attacks:
    if attack == 'Normal':
        continue
        
    print(f"\n{'='*60}")
    print(f"=== {attack} 공격 분석, 냥! ===")
    print(f"{'='*60}\n")
    
    # 해당 공격 vs Normal (이진 분류)
    attack_mask = data['attack_cat'] == attack
    normal_mask = data['attack_cat'] == 'Normal'
    
    binary_data = data[attack_mask | normal_mask].copy()
    
    attack_count = attack_mask.sum()
    print(f"{attack} 샘플 수: {attack_count} ({attack_count/len(data)*100:.2f}%)")
    print(f"Normal 샘플 수: {normal_count}")
    print(f"비율: 1:{normal_count/attack_count:.1f}\n")
    
    # X, y 분리 (제거할 피처 빼고)
    X_cols = [col for col in binary_data.columns 
              if col not in ['attack_cat']]
    X_binary = binary_data[X_cols].values
    y_binary = (binary_data['attack_cat'] == attack).astype(int)
    
    # Random Forest 학습
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=10
    )
    rf.fit(X_binary, y_binary)
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 결과 저장
    results[attack] = {
        'count': attack_count,
        'ratio': normal_count/attack_count,
        'top_features': importances.head(10),
        'all_importances': importances
    }
    
    print(f"=== {attack} 탐지 핵심 Top 10 피처, 냥! ===")
    print(importances.head(10).to_string(index=False))

# ========== 전체 요약 ==========
print("\n" + "=" * 60)
print("=== 전체 공격 유형 요약, 냥냥! ===")
print("=" * 60)

summary = pd.DataFrame({
    'Attack': list(results.keys()),
    'Count': [results[a]['count'] for a in results.keys()],
    'Ratio_to_Normal': [f"1:{results[a]['ratio']:.1f}" for a in results.keys()],
    'Top_Feature': [results[a]['top_features'].iloc[0]['feature'] for a in results.keys()],
    'Top_Importance': [f"{results[a]['top_features'].iloc[0]['importance']:.3f}" for a in results.keys()]
})

print(summary.to_string(index=False))

# ========== 시각화 ==========
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for idx, attack in enumerate(results.keys()):
    if idx >= 9:
        break
    
    top_10 = results[attack]['top_features'].head(10)
    
    axes[idx].barh(range(len(top_10)), top_10['importance'])
    axes[idx].set_yticks(range(len(top_10)))
    axes[idx].set_yticklabels(top_10['feature'], fontsize=8)
    axes[idx].set_xlabel('Importance', fontsize=9)
    axes[idx].set_title(f'{attack} (n={results[attack]["count"]})', fontsize=10)
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.show()

print("\n🎉 전체 공격 유형 분석 완료, 냥냥!")

'''
import pandas as pd

# Test set 로드
test_data = pd.read_csv('datasets/training-data-preprocess.csv')

# Proto 분포 확인
proto_counts = test_data['proto'].value_counts()

print("=== Test Set Proto 분포 ===")
print(proto_counts)
print(f"\n총 프로토콜 종류: {len(proto_counts)}개")
print(f"총 샘플 수: {len(test_data)}개")

# 각 프로토콜 비율
print("\n=== 비율 ===")
for proto, count in proto_counts.items():
    percentage = count / len(test_data) * 100
    print(f"{proto}: {count}개 ({percentage:.2f}%)")

# Clean에 없던 프로토콜
clean_data = pd.read_csv('datasets/training-data-preprocess.csv')
clean_protos = set(clean_data['proto'].unique())
test_protos = set(test_data['proto'].unique())

missing_protos = test_protos - clean_protos

print(f"\n=== Clean에 없는 프로토콜 ===")
print(f"종류: {missing_protos}")

# 각각의 샘플 수
print("\n상세:")
for proto in missing_protos:
    count = len(test_data[test_data['proto'] == proto])
    percentage = count / len(test_data) * 100
    print(f"{proto}: {count}개 ({percentage:.2f}%)")

total_missing = len(test_data[test_data['proto'].isin(missing_protos)])
print(f"\n총 영향받는 샘플: {total_missing}개 ({total_missing/len(test_data)*100:.2f}%)")

'''