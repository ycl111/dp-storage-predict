# 创建模型 多任务学习模型
# 为了增加预测准确率 减少损失
from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        # 每一个列单独做一个序列 预测 features = [5, 9, 187, 188, 193, 194, 197, 198, 241, 242]
        # lstm层 输入(batchsize,49,10) 输出(batchsize,49,100)
        self.lstm1 = nn.LSTM(input_size=10, hidden_size=100, num_layers=1, batch_first=True, bidirectional=False)
        # droupout层 0.2
        self.droupout1 = nn.Dropout(p=0.2)
        # lstm层 输入(batchsize,49,100) 输出(batchsize,49,100)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True, bidirectional=False)
        # lstm层 输入(batchsize,49,100) 输出最后一行 (batchsize,1,100)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True, bidirectional=False)
        # droupout层 0.2
        self.droupout2 = nn.Dropout(p=0.2)
        # 全连接层 输入100维度 输出1维度
        self.dense = nn.Linear(100, 1)

    def forward(self, input):
        # 输入input (batchsize,49,10)  输出(batchsize,1)
        lstm_output1, _ = self.lstm1(input)
        droupout1 = self.droupout1(lstm_output1)
        lstm_output2, _ = self.lstm2(droupout1)
        lstm_output3, _ = self.lstm3(lstm_output2)
        # 取最后一行为输出结果
        lastoutput = lstm_output3[:, -1, :]
        droupout2 = self.droupout2(lastoutput)
        dense_output = self.dense(droupout2)
        return dense_output
