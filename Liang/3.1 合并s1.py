import pandas as pd

# 加载CSV文件
train_dataset = pd.read_csv('D:/radiomic1/glioma/csv4/test_t2_s1.csv')
selected_survival_data = pd.read_csv(
    'D:/radiomic clinic/selected_survival_data.csv')

# 将selected_survival_data的case_submitter_id设置为索引
selected_survival_data.set_index('case_submitter_id', inplace=True)

# 在train_dataset中通过索引连接selected_survival_data的os和os.time列
train_dataset = train_dataset.join(
    selected_survival_data[['OS', 'OS.time']], on='index')

# 去掉label列
train_dataset = train_dataset.drop(columns=['label'])

# 输出结果查看或保存新的CSV文件
print(train_dataset.head())
train_dataset.to_csv(
    'D:/radiomic clinic/updated_test-s1.csv', index=False)
print("finish")
