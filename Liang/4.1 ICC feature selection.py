import pandas as pd

# Read train.csv file
train_df = pd.read_csv('D:/radiomic1/glioma/csv4/test_t2_s1.csv')

# Read train_t2_icc.csv file
icc_df = pd.read_csv('D:/radiomic1/glioma/csv4/train_t2_icc_change.csv')

# Filter selected features based on ICC values greater than 0.8
selected_feature_names = icc_df.loc[icc_df['icc'] >= 0.8, 'FeatureName']

# Keep only selected features along with label from train.csv
train_icc_selection_df = train_df[list(selected_feature_names)+['label']]

# Save the DataFrame with label as train_t2_icc_selection.csv
train_icc_selection_df.to_csv(
    'D:/radiomic1/glioma/csv4/test_t2_s1_icc_selection.csv', index=False)
