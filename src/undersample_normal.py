import pandas as pd
from sklearn.model_selection import train_test_split

# 讀取 CSV 文件
df = pd.read_csv('../data/train_annotations_caries.csv')

# 分離出類別0與類別1的資料
df_class0 = df[df['category_id'] == 0]
df_class1 = df[df['category_id'] == 1]

# 計算類別1的樣本數
n_class1 = len(df_class1)

# 設定類別0所需的樣本數為類別1的 2 倍
desired_class0 = 5 * n_class1

# 從類別0中隨機抽取所需的樣本
df_class0_sampled = df_class0.sample(n=desired_class0, random_state=42)

# 合併 undersampling 後的類別0與所有類別1資料
df_balanced = pd.concat([df_class0_sampled, df_class1], axis=0)

# 可以選擇打亂資料順序
df_balanced = df_balanced.sample(frac=1, random_state=42)

# 切分訓練集和驗證集（90% 訓練, 10% 驗證）
# train_df, val_df = train_test_split(
#     df_balanced, test_size=0.1, random_state=42, stratify=df_balanced['category_id'])

# 輸出到新的 CSV 文件
df_balanced.to_csv(
    '../data/train_annotations_caries_undersample_l_5.csv', index=False)
# val_df.to_csv(
#     '../data/val_annotations_periapical_undersample_l_2.csv', index=False)

# 輸出分割後的資料數量
print(f"訓練集數量: {len(df_balanced)}")
# print(f"驗證集數量: {len(val_df)}")
print(f"訓練集類別分布:\n{df_balanced['category_id'].value_counts()}")
# print(f"驗證集類別分布:\n{val_df['category_id'].value_counts()}")
