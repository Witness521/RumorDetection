# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN_Att'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/valid.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # 类别
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]
        # 词表
        self.vocab_path = dataset + '/data/vocab.pkl'
        # self.vocab_path = dataset + '/data/vocab.txt'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        # self.batch_size = 32                                           # mini-batch大小
        self.batch_size = 1
        self.pad_size = 45                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        # self.hidden_size = 32
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64

'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            # 使用搜狗的预训练词向量来对每一个字进行编码，并转为词向量
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        # 初始化一个可学习的权重矩阵w
        #  nn.Parameter 将一个不可训练的类型Tensor转换成可以训练的类型parameter,并将这个parameter绑定到这个module里面
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)
        # 存储out矩阵
        self.save_out = 0

    def forward(self, x):
        # x, _ = x  原
        x = x[0]
        # [batch_size, seq_len, embedding]=[128, 32, 300]
        # 将传入的微博评论使用搜狗的预训练字向量进行
        emb = self.embedding(x)
        # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]    num_direction：若双向LSTM则num_direction = 2
        # H, _ = self.lstm(emb) 原
        H = self.lstm(emb)[0]
        # [128, 32, 256]
        M = self.tanh1(H)
        ##################### 注意力机制 #####################
        # 注意力机制就是对lstm每刻的隐层进行加权平均。比如句长为4，首先算出4个时刻的归一化分值：[0.1, 0.3, 0.4, 0.2]，然后h终极 = 0.1h + 0.3h + 0.4h + 0.2h
        # M = torch.tanh(torch.matmul(H, self.u))
        # 对LSTM的输出进行非线性激活后与w进行矩阵相乘，并经行softmax归一化，得到每时刻的分值：
        # M与w作乘法，提出128得到[128, 32, 1]， 然后按列做归一化
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)  # [128, 32]
        alpha = alpha.unsqueeze(-1)  # [128, 32, 1]
        # 将LSTM的每一时刻的隐层状态H乘对应的分值后求和
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)   # [128, 256]
        self.save_out = out

        # end ##################### 注意力机制 #####################

        # [128, 64]
        out = self.fc1(out)
        # [128, 2(class类别数)]
        out = self.fc(out)
        return out