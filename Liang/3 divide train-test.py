
from sklearn.model_selection import train_test_split
import pandas as pd

dataPath = 'D:/radiomic1/glioma/tcsv/Total_OMICS_t2.csv'
data = pd.read_csv(dataPath)
# 拆分数据集
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

# 保存训练集和测试集为CSV文件
train_df.to_csv(
    'D:/radiomic1/glioma/tcsv/train_t2.csv', index=False)
test_df.to_csv(
    'D:/radiomic1/glioma/tcsv/test_t2.csv', index=False)
