import sys
import pandas as pd
import os
import random
import shutil
import numpy as np
import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

kinds = ['GBM', 'LGG']
# 这个是特征处理配置文件，具体可以参考pyradiomics官网
para_path = 'D:/radiomic1/glioma/yaml/MR_1mm.yaml'

extractor = featureextractor.RadiomicsFeatureExtractor(para_path)
dir = 'D:/radiomic1/glioma/data/Imagetcga_1/'

for kind in kinds:
    print("{}:开始提取特征".format(kind))
    features_dict = dict()
    df = pd.DataFrame()
    path = dir + kind
    # 使用配置文件初始化特征抽取器
    for index, folder in enumerate(os.listdir(path)):
        for f in os.listdir(os.path.join(path, folder)):
            if 't2' in f:
                ori_path = os.path.join(path, folder, f)
                break
        lab_path = ori_path.replace('t2', 'seg')
        features = extractor.execute(ori_path, lab_path)  # 抽取特征
        # 新增一列用来保存病例文件夹名字
        features_dict['index'] = folder
        for key, value in features.items():  # 输出特征
            features_dict[key] = value
        df = df._append(pd.DataFrame.from_dict(
            features_dict.values()).T, ignore_index=True)
        print(index)
    df.columns = features_dict.keys()
    df.to_csv('D:/radiomic1/glioma/xcsv/' +
              '{}_t2_s.csv'.format(kind), index=0)
    print('Done')
print("完成")
