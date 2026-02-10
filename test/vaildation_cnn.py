import pickle
import torch
import pandas as pd
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from train.models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

preprocess_path = os.path.join(ROOT_DIR, 'train', 'preprocessing.pkl')
with open(preprocess_path, 'rb') as f:
    prep = pickle.load(f)

label_encoder = prep['label_encoder']
scaler = prep['scaler']
cat_encoders = prep['categorical_encoders']

attack_classes = [
    'Normal', 'DoS', 'Generic', 'Exploits', 'Fuzzers',
    'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'
]

num_classes = len(attack_classes)

dataset_path = os.path.join(ROOT_DIR, 'datasets', 'test-set.csv')
test_data = pd.read_csv(dataset_path)
test_ids = test_data['id'].values

drop_features = [
    'id',
    'is_ftp_login','ct_ftp_cmd', 'ct_flw_http_mthd', 'trans_depth'
]

test_data = test_data.drop(columns=drop_features)

for col in ['proto', 'service', 'state']:
    known = set(cat_encoders[col].classes_)
    unknown_mask = ~test_data[col].isin(known)
    
    if unknown_mask.any():
        unknown_vals = test_data.loc[unknown_mask, col].unique()
        print(f"{col}에 없던 값: {unknown_vals}")
        
        test_data.loc[unknown_mask, col] = cat_encoders[col].classes_[0]
    
    test_data[col] = cat_encoders[col].transform(test_data[col])

print("Categorical 인코딩 완료")

X_test = test_data.values
print(f"피처 개수: {X_test.shape[1]}")

X_test_scaled = scaler.transform(X_test)

model = CNN(input_size=X_test_scaled.shape[1], num_classes=num_classes)
model_path = os.path.join(ROOT_DIR, 'train', 'cnn_model.pth')

model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)

predictions = predictions.cpu().numpy()


predicted_labels = label_encoder.inverse_transform(predictions)

submission = pd.DataFrame({
    'id': test_ids,
    'attack_cat': predicted_labels
})
submission_path = os.path.join(ROOT_DIR)
submission.to_csv('submission_cnn.csv', index=False)
print("\nsubmission.csv 생성 완료!")
print(f"\n예측 분포:")
print(pd.Series(predicted_labels).value_counts())
print(f"\n샘플 확인:")
print(submission.head(20))