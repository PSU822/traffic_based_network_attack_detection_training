# 필요없는 컬럼을 제외한 csv를 추출

import pandas as pd
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# 데이터 읽기

path = os.path.join(ROOT_DIR, 'datasets','training-set.csv')
data = pd.read_csv(path)

dropFeatures = ['is_ftp_login','ct_ftp_cmd', 'ct_flw_http_mthd', 'trans_depth']
data = data.drop(columns=dropFeatures)

data = data.drop(columns='id')

feature_cols = [col for col in data.columns 
                if col not in ['id', 'attack_cat']]

label_counts = data.groupby(feature_cols)['attack_cat'].nunique()

conflict_features = label_counts[label_counts > 1].index

conflict_mask = data.set_index(feature_cols).index.isin(conflict_features)
clean_data = data[~conflict_mask]

print(f"제거: {conflict_mask.sum()}개")
print(f"남음: {len(clean_data)}개")
print(f"손실률: {conflict_mask.sum()/len(data)*100:.2f}%")

clean_path = os.path.join(ROOT_DIR, 'datasets','training-data-preprocess.csv')
clean_data.to_csv(clean_path, index=False)