import os

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import joblib

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StorageModel(nn.Module):
    def __init__(self):
        super(StorageModel, self).__init__()
        self.seqmodel = nn.Sequential(
            nn.Linear(10, 320),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(320, 320),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(320, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seqmodel(x)


# 定义二分类损失函数focal loss
class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gama=2, alpha=0.75, reduction='mean'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gama = gama
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target, weight):
        BCE_loss = F.binary_cross_entropy(logits, target, weight=weight, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gama * BCE_loss

        if self.reduction:
            return torch.mean(F_loss)
        else:
            return F_loss


def prepare_data():
    # 选取属性作为特征值SMART 5, 9, 187, 188, 193, 194, 197, 198, 241, 242
    features_specified = []
    features = [5, 9, 187, 188, 193, 194, 197, 198, 241, 242]
    for feature in features:
        features_specified.append("smart_{0}_raw".format(feature))
    features_specified.append("failure")
    data = pd.read_csv("./2020_balance.csv", index_col=0)  # 获取数据
    # data_failure = data[data["failure"] == 1]
    # print(data_failure["model"].value_counts())
    # 选取特定型号硬盘
    data = data[data['model'] == 'ST12000NM0007']
    data = data[features_specified]
    data = data.fillna(0)
    data_failure = data[data["failure"] == 1].values
    data_train_failure = data_failure[:int(len(data_failure) * 0.8)]
    data_test_failure = data_failure[int(len(data_failure) * 0.8):]
    data = pd.read_csv("./data_Q1_2020/2020-01-06.csv", index_col=0)
    data = data[data['model'] == 'ST12000NM0007']
    data = data[features_specified]
    data = data.fillna(0)
    # 负样本数量是正样本数量的10倍
    data_success = data[data["failure"] == 0].values
    # 计算loss中的权重tensor
    success_weight = (len(data_failure) + len(data_success)) / len(data_success)
    failure_weight = (len(data_failure) + len(data_success)) / len(data_failure)
    weight = failure_weight // success_weight + 1
    data_train_success = data_success[:int(len(data_success) * 0.8)]
    data_test_success = data_success[int(len(data_success) * 0.8):]
    data_train = np.concatenate([data_train_success, data_train_failure])
    np.random.shuffle(data_train)

    len_data_train = len(data_train)

    data_test = np.concatenate([data_test_success, data_test_failure])
    np.random.shuffle(data_test)

    data_all = np.concatenate([data_train, data_test])
    #  # 归一化 保存standart scaler的参数 应用到dpStorageLearn 做归一化使用 保持一致
    scaler = StandardScaler()
    data_all[:, :-1] = scaler.fit_transform(data_all[:, :-1])
    # 保存归一化参数 用于下一步预测后判断磁盘健康状态归一化使用
    joblib.dump(scaler,os.path.join("./model","scaler.pkl"))
    data_train = data_all[:len_data_train]
    data_test = data_all[len_data_train:]

    data_train = torch.tensor(data_train.tolist(), dtype=torch.float)
    data_test = torch.tensor(data_test.tolist(), dtype=torch.float)

    return data_train, data_test, weight


def data_batchloader(data_train, data_test, batchsize):
    train_batch = []
    test_batch = []

    for i in range(int(len(data_train) // batchsize)):
        train_batch.append(data_train[i * batchsize:(i + 1) * batchsize])
    for i in range(int(len(data_test) // batchsize)):
        test_batch.append(data_test[i * batchsize:(i + 1) * batchsize])

    if len(data_train) % batchsize != 0:
        train_batch.append(data_train[(0 - len(data_train) % batchsize):])
    if len(data_test) % batchsize != 0:
        test_batch.append(data_test[(0 - len(data_test) % batchsize):])

    return train_batch, test_batch


def trainModel(train_batch, test_batch, weight):
    storage_model = StorageModel()
    storage_model.to(device)
    lr = 0.01
    optimizer = torch.optim.Adam(storage_model.parameters(), lr=lr)
    fn_loss = BCEFocalLosswithLogits()
    fn_loss.to(device)
    epoch = 20
    train_step = 0
    # 添加tensorboard
    writer = SummaryWriter("./logs")
    for step in range(epoch):
        # 开始训练
        storage_model.train()
        for X in train_batch:
            # 定义损失权重
            weight_all = []
            for i in range(len(X)):
                weight_all.append([weight])
            weight_all = torch.tensor(weight_all)
            weight_all = weight_all.to(device)
            # fn_loss.weight = weight_all
            data_x = X[:, :-1]
            data_y = X[:, -1:]
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            predict_y = storage_model(data_x)

            loss = fn_loss(predict_y, data_y, weight_all)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
            if train_step % 1000 == 0:
                print("train step {},loss is {}".format(train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), train_step)
        # 模型保存
        torch.save(storage_model,
                   os.path.join("./model/", "storage_health_model_{}.pth".format(step + 1)))

        # 模型验证
        with torch.no_grad():
            storage_model.eval()
            predict_all = []
            data_y_all = []
            for X in test_batch:
                data_x = X[:, :-1]
                data_y = X[:, -1:]
                data_x = data_x.to(device)
                data_y_all.append(data_y)
                predict_y = storage_model(data_x)
                predict_all.append(predict_y)
            predict_all = torch.cat(predict_all)
            predict_all = torch.flatten(predict_all)
            predict_all = torch.tensor([1 if prd > 0.5 else 0 for prd in predict_all.tolist()])
            data_y_all = torch.cat(data_y_all)
            data_y_all = torch.flatten(data_y_all)
            print("测试集模型评估结果：")
            accuracy = accuracy_score(data_y_all, predict_all)
            print("精确率为：{}".format(accuracy))
            writer.add_scalar("accuracy", accuracy, step + 1)
            precision = precision_score(data_y_all, predict_all)
            print("查准率为:{}".format(precision))
            recall = recall_score(data_y_all, predict_all)
            writer.add_scalar("precision", precision, step + 1)
            print("查全率为：{}".format(recall))
            writer.add_scalar("recall", recall, step + 1)
            f1 = f1_score(data_y_all, predict_all)
            print("F1-Score为：{}".format(f1))
            writer.add_scalar("f1", f1, step + 1)
    writer.close()


if __name__ == "__main__":
    data_train, data_test, weight = prepare_data()
    train_batch, test_batch = data_batchloader(data_train, data_test, 64)
    trainModel(train_batch, test_batch, weight)

    # isolation_forest(data_train, data_test, weight)
