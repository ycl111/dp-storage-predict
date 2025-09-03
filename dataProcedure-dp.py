# 由于Backblaze数据集按照每天上百万的磁盘数据保存的，我们目前需要使用lstm做预测 需要一段时间相同磁盘的smart数据变化趋势
# 以磁盘序列号.csv文件名重新整理按照时间变化一年周期内的smart数据变化情况
import pandas as pd

import os

# 每个日期文件最多只有一条同一个序列号磁盘的smart信息记录
# 取2020_balance.csv failure为1的磁盘序列号 然后遍历根据序列号取得每天的数据
data_frame = pd.read_csv("./2020_balance.csv")
# 选取特定型号硬盘
data_frame = data_frame[data_frame['model'] == 'ST12000NM0007']
failure_series_numbers = data_frame[data_frame["failure"]==1]["serial_number"].values

# 每100个序列号保存一次
for j in range(len(failure_series_numbers)//100+1):
    if j < (len(failure_series_numbers)//100):
        serial_numbers = failure_series_numbers[j*100:(j+1)*100]
    else:
        serial_numbers = failure_series_numbers[j * 100:]
    # 以磁盘序列号为索引 查找数据集一年内所有此序列号的记录 并保存为此序列号.csv文件格式 放在data目录下
    data = {}
    for sr_num in serial_numbers:
    # 初始化100个查找到的sr_num保存的空列表
        data[sr_num] = []

    # 全年文件查找
    for i in range(4):
        dirname = "./data_Q{}_2020/".format(i+1)
        filenames = os.listdir(dirname)
        for file in filenames:
            if file.endswith(".csv"):
                data_tmp = pd.read_csv(os.path.join(dirname,file))
                # 选取特定型号硬盘
                data_tmp = data_tmp[data_tmp['model'] == 'ST12000NM0007']
                for sr_num in serial_numbers:
                    if data_tmp["serial_number"].isin([sr_num]).any():
                        data[sr_num].append(data_tmp[data_tmp["serial_number"]==sr_num])
    # list转dataframe 然后保存为csv文件
    for sr_num,data_list in data.items():
        df = pd.concat(data_list,ignore_index=True)
        # 按照日期排序
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by="date")
        df.to_csv("./data/{}.csv".format(sr_num),index=False,header=True)


