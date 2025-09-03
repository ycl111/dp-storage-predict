import os
import time

import numpy as np
import pandas as pd
import warnings

import torch
from sklearn.preprocessing import StandardScaler
from model import *
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# data目录下已按照时间顺序 以磁盘序列号.csv为文件名保存了100个文件 处理方法参考data_procedure.py
# 数据准备
from torch.utils.tensorboard import SummaryWriter

scaler = joblib.load(os.path.join("./model","scaler.pkl"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_prepare(data_dir, window_size=50, train_test_split_percent=0.8):
    # 下列smart列与磁盘性能关系密切 作为预测数据特征列
    # SMART 5: 重映射扇区计数
    # SMART 9: 通电时间累计
    # SMART 187: 无法校正的错误
    # SMART 188: 指令超时计数
    # SMART 193: 磁头加载/卸载计数
    # SMART 194: 温度
    # SMART 197: 等待被映射的扇区数
    # SMART 198: 报告给操作系统的无法通过硬件ECC校正的错误
    # SMART 241: 逻辑块寻址模式写入总数
    # SMART 242: 逻辑块寻址模式读取总数

    # 选取属性作为特征值SMART 5, 9, 187, 188, 193, 194, 197, 198, 241, 242
    features_specified = []
    features = [5, 9, 187, 188, 193, 194, 197, 198, 241, 242]
    for feature in features:
        features_specified.append("smart_{0}_raw".format(feature))
    # 数据准备
    filenames = os.listdir(data_dir)
    # 特征集
    data_x = []
    # 目标值 10列数据
    data_y = []

    for file in filenames:
        if file.endswith(".csv"):

            data = pd.read_csv(os.path.join(data_dir, file))
            data = data[data['model'] == 'ST12000NM0007']
            if (len(data.values)==0):
                continue
            data_feature = data[features_specified]
            # 有空值存在先填充空值
            data_feature = data_feature.fillna(0)
            data_feature = data_feature.values
            # 归一化
            data_feature = scaler.transform(data_feature)
            for i in range(len(data_feature) - window_size + 1):
                data_x.append(data_feature[i:i + window_size-1].tolist())
                # 目标值 10列数据 用于预测
                data_y.append(data_feature[i + window_size-1:i + window_size].tolist())
    # list转成tensor
    data_x = torch.tensor(data_x, dtype=torch.float)
    data_y = torch.tensor(data_y, dtype=torch.float)
    # 训练集和测试集划分
    data_x_train = data_x[:int(len(data_x)*train_test_split_percent)]
    data_x_test = data_x[int(len(data_x) * train_test_split_percent):]
    data_y_train = data_y[:int(len(data_y) * train_test_split_percent)]
    data_y_test = data_y[int(len(data_y) * train_test_split_percent):]


    return data_x_train,data_y_train,data_x_test,data_y_test


# 按照batchsize进行数据分组
def data_batch_loader(data_x, data_y, batchsize=64, lastdrop=False):
    len_data = len(data_x)
    data_x_all = []
    data_y_all = []
    data_x_batch = []
    data_y_batch = []
    for i in range(int(len_data / batchsize)):
        data_x_batch.append(
            data_x[i * batchsize:(i + 1) * batchsize].tolist())  # data_x和data_y tensor类型转list 加入到分组后的list
        data_y_batch.append(data_y[i * batchsize:(i + 1) * batchsize].tolist())
    # 转换list为tensor
    data_x_batch = torch.tensor(data_x_batch)
    data_y_batch = torch.tensor(data_y_batch)
    data_x_all.append(data_x_batch)
    data_y_all.append(data_y_batch)
    if ((len_data % batchsize) != 0) and (not lastdrop):  # 如果lastdrop为false表示最后剩余数据保留
        data_x_last = []
        data_y_last = []
        data_x_last.append(data_x[len_data - (len_data % batchsize):].tolist())
        data_y_last.append(data_y[len_data - (len_data % batchsize):].tolist())
        # list转tensor
        data_x_last = torch.tensor(data_x_last)
        data_y_last = torch.tensor(data_y_last)
        data_x_all.append(data_x_last)
        data_y_all.append(data_y_last)
    return data_x_all, data_y_all


# 递归调用模型生成多日预测数据
def mutiplePredict(data_x, lstmmodels, predict_length):
    # data_x (1,49,10)
    # lstmmodels是10个模型的模型名的列表

    # cur_frame  (1,49,10)
    cur_frame = data_x
    cur_frame = cur_frame.to(device)
    predict = []
    # 循环获取预测的三行值
    for _ in range(predict_length):
        # 每个模型预测一列数值 然后组合成列表
        predict_y = []
        for j in range(10):
            lstmmodels[j].to(device)
            predict_y_tmp = lstmmodels[j](cur_frame)
            predict_y.append(np.array(predict_y_tmp.tolist()))
        predict_y = np.concatenate(predict_y)
        # 调整predict_y的形状(10,1)为(1,10)
        predict_y = predict_y.reshape((-1, 10))
        predict.append(predict_y)
        # cur_frame列表右移一位 然后末尾加入上一步的预测结果
        cur_frame = cur_frame[:, 1:, :]
        # torch.unsqueeze 扩充维度 从(1,10) 更改为(1,1,10)
        predict_y = torch.tensor(predict_y.tolist())
        predict_y = torch.unsqueeze(predict_y, dim=1).to(device)
        # torch.cat 按照行把predict_y 添加到末尾 进行下一步预测 形成递归 dim表示在第二个维度合并
        cur_frame = torch.cat((cur_frame, predict_y), dim=1).to(device)

    predict = torch.tensor(np.concatenate(predict).tolist())  # (predict_length,10)

    return predict


# 磁盘健康模型，根据smart预测磁盘健康度
def predict_storage_health(data_x, storageHealthModelpath):
    # 加载模型
    optimizer = torch.load(storageHealthModelpath)
    # data_predict 预测磁盘是否健康 0是健康 1是有问题 (3,10)
    # 使用磁盘健康模型进行预测
    predict_all =[]
    data_x = data_x.to(device)
    for i in range(len(data_x)):
        data_predict = optimizer.predict(data_x[i])
        data_predict = torch.tensor([1 if prd > 0.5 else 0 for prd in data_predict.tolist()])
        predict_all.append(data_predict)

    return predict_all


def predict_data(predict_data_dir, predict_length, lstmmodels, storageHealthModelpath, window_size=50):
    # 下列smart列与磁盘性能关系密切 作为预测数据特征列
    # SMART 5: 重映射扇区计数
    # SMART 9: 通电时间累计
    # SMART 187: 无法校正的错误
    # SMART 188: 指令超时计数
    # SMART 193: 磁头加载/卸载计数
    # SMART 194: 温度
    # SMART 197: 等待被映射的扇区数
    # SMART 198: 报告给操作系统的无法通过硬件ECC校正的错误
    # SMART 241: 逻辑块寻址模式写入总数
    # SMART 242: 逻辑块寻址模式读取总数

    # 选取属性作为特征值SMART 5, 9, 187, 188, 193, 194, 197, 198, 241, 242
    features_specified = []
    features = [5, 9, 187, 188, 193, 194, 197, 198, 241, 242]
    for feature in features:
        features_specified.append("smart_{0}_raw".format(feature))
    # 数据准备
    # 特征值列表
    data_x = []
    # 目标值列表
    data_y = []
    filenames = os.listdir(predict_data_dir)
    for file in filenames:
        if file.endswith(".csv"):
            filename = os.path.join(predict_data_dir, file)
            # 判断文件内的长度是否大于window_size+predict_length-1
            if len(pd.read_csv(filename).values) < (window_size + predict_length - 1):
                continue
            data = pd.read_csv(filename)
            # 选取特定型号硬盘
            data = data[data['model'] == 'ST12000NM0007']
            if (len(data.values)==0):
                continue
            data_x_feature = data[features_specified]
            data_y_label = data["failure"].values
            # 有空值存在先填充空值
            data_x_feature = data_x_feature.fillna(0)

            # 循环data 直到倒数第predict_length 以便做模型评估 准确率 召回率
            data_x_feature = data_x_feature.values

            for i in range(len(data_x_feature) - predict_length - window_size + 2):
                # 目标值是特征值后predict_length行 即为之后与预测对比值
                data_y.append(data_y_label[i + window_size - 1:i + window_size - 1 + predict_length])
                data_x_window = data_x_feature[i:i + window_size - 1]
                # 归一化
                data_x_window = scaler.transform(data_x_window)
                # 扩容维度
                data_x_window = torch.unsqueeze(torch.Tensor(data_x_window.tolist()), dim=0)  # (1,49,10)
                # 时序预测  (predict_length,10)
                predict = mutiplePredict(data_x_window, lstmmodels, predict_length)
                # 硬盘健康度预测
                storage_health_status = predict_storage_health(predict, storageHealthModelpath)
                data_x.append(np.array(storage_health_status))
    data_x = np.stack(data_x)
    data_y = np.stack(data_y)
    return data_x, data_y


# 模型训练和保存
def train_eval_model(data_x_train_batch, data_y_train_batch, data_x_test_batch, data_y_test_batch, predict_data_dir,
                     predict_length):
    # 初始化10个模型实例 损失函数和优化器 用于预测10个列的预测值 训练使用
    lstmmodels = []
    optimizer = []
    loss_fn = []
    for i in range(10):
        lstmmodels.append(LSTMModel())
        # 添加gpu支持
        lstmmodels[i].to(device)
        # 损失函数
        loss_fn.append(nn.MSELoss(reduction="mean"))
        loss_fn[i].to(device)

        # 优化器
        lr = 0.01  # 学习率
        optimizer.append(torch.optim.Adam(lstmmodels[i].parameters(), lr=lr))

    # 设置训练的一些参数
    # 总的训练次数
    total_train_step = 0
    # 总的测试次数
    total_test_step = 0
    # 训练轮数
    epoch = 20

    # 添加tensorboard
    writer = SummaryWriter("./logs")
    start_time = time.time()
    for epoch_step in range(epoch):
        print("第{}次训练开始:\n".format(epoch_step + 1))

        # 循环取出batch数据和last数据
        for i in range(len(data_x_train_batch)):
            for j in range(len(data_x_train_batch[i])):
                # 获取训练数据x (batchsize,49.10)
                data_x_train = data_x_train_batch[i][j]
                # 获取训练数据目标值
                data_y_train = data_y_train_batch[i][j]
                data_x_train = data_x_train.to(device)
                data_y_train = data_y_train.to(device)
                # 调整data_y_train形状 (batchsize,1,10) 调整为(batchsize,10)
                data_y_train = torch.reshape(data_y_train, (-1, 10))
                total_train_loss = 0
                # 循环预测出每一列的数值 一共10列数值
                for k in range(10):
                    lstmmodels[k].train()
                    # 预测
                    data_y_predict = lstmmodels[k](data_x_train)

                    # 取出第k列的数据和预测值计算损失 形状为(batchsize,1)
                    data_y_train_col = data_y_train[:, k:k + 1]
                    # 损失函数
                    loss = loss_fn[k](data_y_predict, data_y_train_col)
                    # 每一列计算损失合并起来
                    total_train_loss += loss.item()
                    # 优化器优化模型
                    # 梯度清零
                    optimizer[k].zero_grad()
                    # 反向传播
                    loss.backward()
                    # 梯度更新
                    optimizer[k].step()

                total_train_step += 1
                if total_train_step % 100 == 0:
                    end_time = time.time()
                    print(end_time - start_time)
                    print("第{}次训练，损失值为{}\n".format(total_train_step, total_train_loss))
                    writer.add_scalar("train_loss", loss.item(), total_train_step)

        with torch.no_grad():
            total_test_loss = 0
            # 循环取出batch数据和last数据
            for i in range(len(data_x_test_batch)):
                for j in range(len(data_x_test_batch[i])):
                    data_x_test = data_x_test_batch[i][j]
                    data_y_test = data_y_test_batch[i][j]
                    data_x_test = data_x_test.to(device)
                    data_y_test = data_y_test.to(device)
                    # 调整data_y_test形状 (batchsize,1,10) 调整为(batchsize,10)
                    data_y_test = torch.reshape(data_y_test, (-1, 10))
                    for k in range(10):
                        lstmmodels[k].eval()
                        data_y_predict = lstmmodels[k](data_x_test)
                        # 取出第k列的数据和预测值计算损失 形状为(batchsize,1)
                        data_y_test_col = data_y_test[:, k:k + 1]
                        # 损失函数
                        loss = loss_fn[k](data_y_predict, data_y_test_col)
                        total_test_loss += loss.item()

            total_test_step += 1
            print("整体测试集的loss为:{}".format(total_test_loss))
            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            # 模型评估 评估预测的磁盘健康度和实际磁盘健康度的准确率 召回率 F1-score

            # 保存每次训练后的10个模型
            for i in range(10):
                torch.save(lstmmodels[i],
                           os.path.join("./model/", "lstmModel_{}_{}.pth".format(total_test_step, i + 1)))

            modelpath = os.path.join("model", "storage_health_model_20.pth")  # 预测磁盘是否正常的机器学习模型

            data_x, data_y = predict_data(predict_data_dir, predict_length, lstmmodels, modelpath, window_size=50)
            predict_label_all = data_x.flatten()
            label_all = data_y.flatten()

            # 模型评估结果
            print("第{}轮模型评估结果：".format(epoch_step + 1))
            accuracy = accuracy_score(label_all, predict_label_all)
            print("精确率为：{}".format(accuracy))
            precision = precision_score(label_all, predict_label_all)
            print("查准率为:{}".format(precision))
            recall = recall_score(label_all, predict_label_all)
            print("查全率为：{}".format(recall))
            f1 = f1_score(label_all, predict_label_all)
            print("F1-Score为：{}".format(f1))

    writer.close()


def trainModel(train_data_dir, predict_data_dir, predict_length):
    window_size = 50
    train_test_split_percent = 0.8  # 表示训练集 测试集按照0.8 0.2比例进行划分
    data_x_train, data_y_train, data_x_test, data_y_test = data_prepare(train_data_dir,
                                                                        window_size,
                                                                        train_test_split_percent)
    batchsize = 64
    data_x_train_batch, data_y_train_batch = data_batch_loader(data_x_train, data_y_train, batchsize, lastdrop=True)
    data_x_test_batch, data_y_test_batch = data_batch_loader(data_x_test, data_y_test, batchsize, lastdrop=True)

    # 模型的训练和保存
    train_eval_model(data_x_train_batch, data_y_train_batch, data_x_test_batch, data_y_test_batch, predict_data_dir,
                     predict_length)


if __name__ == "__main__":
    trainModel("train_data", "predict_data", 3)

