import torch.nn as nn
import torch

class Model(nn.Module):
    # RCNN
    def __init__(self):
        super(Model, self).__init__()
        # 设置双向的LSTM
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=256,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.5)
        self.relu = nn.ReLU()
        # max pooling的窗口大小
        self.maxpool = nn.MaxPool1d(1)

        # self.rnn_hidden = 256  self.hidden_size = 768
        self.fc = nn.Linear(256 * 2 + 768, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = torch.cat((x, out), 2)
        # out = out.view(out.shape[0], -1)
        out = self.relu(out)
        out = out.permute(0, 2, 1)
        # # 在seq_len长度上做最大池化  out(8,1280)
        out = self.maxpool(out).squeeze()
        # self.save_out = out
        out = self.fc(out)
        # 这个softmax函数的目的就是再多分类的时候将所有概率归一化处理，但是本课题是二分类的问题，因此只需要比较两个的大小即可
        # 在后面单条语句要输出概率的时候，就可以做Softmax函数处理
        # out = self.softmax(out)
        return out