# 准备数据 从Backblaze公司数据集下载2020年一季度到四季度的数据
# 由于正常数据和异常数据量差异太大，所以提取全年所有异常数据，然后选取同等数量的正常数据
# 以一季度数据处理为例
# 由于failed故障样本占总样本比例太小，容易造成机器学习偏向正常样本，所以把故障样本集中处理，再找出数量相同的正常样本进行训练
# 遍历数据目录文件夹里面的所有日期的数据文件
import os
import pandas as pd
import gc
# 不同季度文件目录不一样 修改path值
for i in range(4):
    path = "./data_Q{}_2020/".format(i+1)
    files = os.listdir(path)
    # 过滤所有文件末尾是csv的文件
    files_csv=[f for f in files if f.endswith(".csv")]
    # 由于数据读取后放入列表中需要占用内存 所以为了避免内存不够用 只是一个季度一个季度处理数据
    # 创建一个空的dataframe列表 并逐个读取文件加入到df列表中
    dfs = []
    for file in files_csv:
        df = pd.read_csv(os.path.join(path,file))
        dfs.append(df)
    # df列表合并
    df_merged = pd.concat(dfs,ignore_index=True)
    # 取得异常数据
    failed = df_merged[df_merged["failure"]==1]
    # 保存异常数据 q1文件名为q1_failed.csv q2文件名为q2_failed.csv q3文件名为q3_failed.csv q4文件名为q4_failed.csv
    failed.to_csv("q{}_failed.csv".format(i+1),header=True,index=False)

    # 由于加载数据较多 需要手动回收内存
    del failed
    del df_merged
    gc.collect()

# 合并q1-q4 failed异常数据进行合并
dfs = []
for i in range(4):
    data = pd.read_csv("q{}_failed.csv".format(i+1))
    dfs.append(data)
df_merged_failed = pd.concat(dfs,ignore_index=True)
# 1495个failed 再取得1495个正常样本 平衡训练样本
data = pd.read_csv("./data_Q4_2020/2020-12-08.csv")
valid = data[data["failure"]==0]
df_merged_valid = valid[:1495]
# 将异常数据和同等数量的正常数据进行合并保存
data_all = pd.concat([df_merged_valid,df_merged_failed],ignore_index=True)
data_all.to_csv("2020_balance.csv",header=True,index=False)


# for i in range(4):
#     path = "./data_Q{}_2020/".format(i+1)
#     files = os.listdir(path)
#     # 过滤所有文件末尾是csv的文件
#     files_csv=[f for f in files if f.endswith(".csv")]
#     # 由于数据读取后放入列表中需要占用内存 所以为了避免内存不够用 只是一个季度一个季度处理数据
#     # 创建一个空的dataframe列表 并逐个读取文件加入到df列表中
#     dfs_failure = []
#
#     for file in files_csv:
#         df = pd.read_csv(os.path.join(path,file))
#         # 取得异常数据
#         failed = df[df["failure"] == 1]
#         if len(failed)>10:
#             dfs_failure.append({
#                 "filename": file,
#                 "failurecount": len(failed)
#             })
#     for failurefile in dfs_failure:
#         print(failurefile)
