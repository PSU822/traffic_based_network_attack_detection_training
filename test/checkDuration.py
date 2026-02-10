import pandas as pd

data = pd.read_csv('datasets/training-set.csv')

feature_cols = [col for col in data.columns 
                if col not in ['id','attack_cat']]

duplicated = data[data.duplicated(subset=feature_cols, keep=False)]

print(f"전체: {len(data)}개")
print(f"중복 피처 가진 샘플: {len(duplicated)}개")
print(f"비율: {len(duplicated)/len(data)*100:.2f}%")

conflict = duplicated.groupby(feature_cols)['attack_cat'].nunique()
conflict = conflict[conflict > 1]

print(f"\n충돌하는 피처 조합: {len(conflict)}개")
print(f"영향받는 샘플: {conflict.sum()}개")
