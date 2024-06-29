# 数据读取
library(dplyr)
library(data.table)
clin_data <- fread("D:\\radiomic clinic\\clinical.cart.2024-04-22\\clinical.tsv", data.table = F)

# 去重
use_clin <- clin_data %>%
    dplyr::select(case_submitter_id, vital_status, days_to_death, days_to_last_follow_up) %>%
    dplyr::filter(!duplicated(case_submitter_id)) ## 去除重复

# 数据合并
sur_data <- use_clin %>%
    dplyr::mutate(OS.time = case_when(
        vital_status == "Alive" ~ days_to_last_follow_up,
        vital_status == "Dead" ~ days_to_death
    )) %>%
    dplyr::mutate(OS = case_when(
        vital_status == "Alive" ~ 0,
        vital_status == "Dead" ~ 1
    ))

# 保存筛选的列
sur_data1 <- sur_data %>%
    select(case_submitter_id, OS, OS.time) %>%
    na.omit()

# 将结果保存为 CSV 文件
write.csv(sur_data1, file = "D:\\radiomic clinic\\survival_data.csv", row.names = FALSE)
