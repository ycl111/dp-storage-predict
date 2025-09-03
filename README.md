# dp-storage-predict

#### 介绍
使用深度学习lstm预测未来几天数据 使用机器学习随机森林算法 预测磁盘健康状态  

#### 软件架构
软件架构说明
数据集选择 blackblaze的2020年四个季度的公用数据集 需自行下载解压为 data_Q1_2020 data_Q2_2020 data_Q3_2020 data_Q4_2020 四个目录

predict_data 目录存储了磁盘序列号.csv文件 每个文件是用于预测的数据 以磁盘序列号为文件名，需要至少window_size -1天数据 此处是49天数据 超出的数据只取最后49天数据

model 目录用于保存磁盘健康度和smart数值预测的模型

log 目录用于保存tensorboard生成的图表 用于分析深度学习模型的优劣

dataProcedure-storageHealth.py 用于磁盘健康度训练需要的数据 数据处理逻辑 用于平衡正常样本和异常样本数量

dataProcedure-dp.py 用于处理smart值预测lstm训练所需数据 数据处理逻辑 搜索公有数据集中以磁盘序列号为目标查找文件 按照时间顺序合并到磁盘序列号.csv文件中 存储到目录data中 手动把其中一部分文件复制到train_data目录中用于训练使用 另一部分文件复制到predict_data中用于验证使用

storageHealthModel.py 用于磁盘健康度检测 准确率 查准率 查全率 F1-Score的数值 并保存为./model/storage_health_model_{}.pth 用于预测磁盘未来几天状态的模型使用

model.py smart数值预测lstm模型结构

dpStorageLearn.py 用于模型训练 模型预测方法 用于磁盘smart数值预测和磁盘健康度检测结合使用 通过深度学习lstm模型预测未来几天数据 然后根据磁盘健康度检测模型预测出来的数据判断磁盘是否正常

requirements.txt 用于pip安装环境使用

应用环境 
python 3.6 

pip 21.2

#### 使用说明
# 准备工作

下载公有数据集 blackblaze 2020年四个季度数据集压缩文件 并解压缩  data_Q1_2020 data_Q2_2020 data_Q3_2020 data_Q4_2020 四个目录

# 安装依赖
pip install -r requirements.txt
# 磁盘健康度检测数据处理
python dataProcedure-storageHealth.py
# 磁盘健康度检测模型训练，评估，保存
python storageHealthModel.py
# smart数值预测数据处理
python dataProcedure-dp.py
# smart数值预测模型训练 评估 和保存 根据预测结果结合磁盘健康度检测模型判断磁盘状态，与已知状态进行对比 得到准确率，查全率 f1
python dpStorageLearn.py

结果
磁盘健康度检测 精确率 准确率 查全率 f1
测试集模型评估结果：
精确率为：0.9966582007752974
查准率为:0.8333333333333334
查全率为：0.7971014492753623
F1-Score为：0.8148148148148148

smart数值预测结合磁盘健康度检测的结果 精确率 准确率 查全率 f1
模型评估结果：
精确率为：0.1944268077601411
查准率为:0.002707423580786026
查全率为：1.0
F1-Score为：0.005400226461109659

