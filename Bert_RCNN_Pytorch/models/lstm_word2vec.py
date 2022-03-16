# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding='embedding_SougouNews.npz'):
        # 模型名称
        self.model_name = 'lstm_word2vec'

        # class_list类别
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        # 使用CPU或GPU进行训练
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 词表
        self.vocab_path = dataset + '/data/vocab.pkl'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量

        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一

        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000
        # num_classes类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 5
        # mini-batch大小
        self.batch_size = 80
        # 每句话处理成的长度(短填长切)
        self.pad_size = 100
        # lr学习率
        self.learning_rate = 5e-3
        self.dropout = 0.3
        # 输入的大小
        self.input_size = 300

        # 词嵌入的长度
        self.hidden_size = 768
        # 隐藏层节点数
        self.rnn_hidden = 128
        self.num_layers = 2

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # word2vec
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        # LSTM
        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.rnn_hidden,
                            num_layers=config.num_layers,
                            bidirectional=False)

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.rnn_hidden * config.pad_size, config.num_classes)

    def forward(self, x):
        x_len = x[1].cpu()
        x = x[0]  # 输入的句子
        emb = self.embedding(x)
        # emb = pack_padded_sequence(emb, x_len, batch_first=True, enforce_sorted=False)
        emb = self.lstm(emb)[0]
        # emb = pad_packed_sequence(emb, batch_first=True, total_length=x_len)
        emb = emb.view(emb.shape[0], -1)
        emb = self.dropout(emb)
        emb = self.fc(emb)  # 句子最后时刻的 hidden state
        return emb

