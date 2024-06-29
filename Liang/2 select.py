import pandas as pd

# 读取 Total_OMICS_t2.csv 文件和 survival_data.csv 文件
total_omics = pd.read_csv("D:/radiomic1/glioma/csv2/Total_OMICS_t2.csv")
survival_data = pd.read_csv("D:/radiomic clinic/survival_data.csv")

# 使用 Total_OMICS_t2.csv 中的 index 列筛选 survival_data.csv 的 case_submitter_id
selected_case_submitter_ids = total_omics['index']

# 根据筛选出的 case_submitter_id 从 survival_data 中提取相应的行
selected_survival_data = survival_data[survival_data['case_submitter_id'].isin(
    selected_case_submitter_ids)]

# 将筛选出的数据保存为 CSV 文件
selected_survival_data.to_csv(
    "D:/radiomic clinic/selected_survival_data.csv", index=False)
