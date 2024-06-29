import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

gbm_data = pd.read_csv('D:/radiomic1/glioma/tcsv/GBM_t2_s.csv')
lgg_data = pd.read_csv('D:/radiomic1/glioma/tcsv/LGG_t2_s.csv')

gbm_data.insert(1, 'label', 1)  # 插入标签
lgg_data.insert(1, 'label', 0)  # 插入标签

# 因为有些特征是字符串，直接删掉
cols = [x for i, x in enumerate(
    gbm_data.columns) if type(gbm_data.iat[1, i]) == str]
cols.remove('index')
gbm_data = gbm_data.drop(cols, axis=1)
cols = [x for i, x in enumerate(
    lgg_data.columns) if type(lgg_data.iat[1, i]) == str]
cols.remove('index')
lgg_data = lgg_data.drop(cols, axis=1)

# 再合并成一个新的csv文件。
total_data = pd.concat([gbm_data, lgg_data])
total_data.to_csv(
    'D:/radiomic1/glioma/tcsv/Total_OMICS_t2.csv', index=False)

# 简单查看数据的分布
fig, ax = plt.subplots()
sns.set()
ax = sns.countplot(x='label', hue='label', data=total_data)
plt.show()
print(total_data['label'].value_counts())
