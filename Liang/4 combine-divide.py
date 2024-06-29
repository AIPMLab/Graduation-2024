from sklearn.model_selection import train_test_split
import pandas as pd

# 读取训练集和测试集的 CSV 文件
train_data = pd.read_csv(
    'D:/radiomic clinic/updated_train-s1 -add.csv')
test_data = pd.read_csv('D:/radiomic clinic/updated_test-s1.csv')

# 合并训练集和测试集
combined_df = pd.concat([train_data, test_data], ignore_index=True)
# combined_df.to_csv('D:/radiomic clinic/combined-df.csv')
# 拆分数据集，test_size=0.37
new_train_df, new_test_df = train_test_split(
    combined_df, test_size=0.37, random_state=42)

# 保存新的训练集和测试集为CSV文件
new_train_df.to_csv(
    'D:/radiomic clinic/new_updated_train-s1 -add.csv', index=False)
new_test_df.to_csv(
    'D:/radiomic clinic/new_updated_test-s1.csv', index=False)
