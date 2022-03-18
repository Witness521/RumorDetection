import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding='embedding_SougouNews.npz'):
        # 模型名称
        self.model_name = 'TextCNN'

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
        self.num_classes = 2
        # epoch数
        self.num_epochs = 5
        # mini-batch大小
        self.batch_size = 80
        # 每句话处理成的长度(短填长切)
        self.pad_size = 140
        # lr学习率
        self.learning_rate = 5e-3
        self.dropout = 0.1

        self.kernel_dim = 100  # 卷积产生的通道数
        self.kernel_sizes = (3, 4, 5)



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # word2vec
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=config.kernel_dim, kernel_size=(K, 300)) for K in config.kernel_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(len(config.kernel_sizes) * config.kernel_dim, 2)

    def forward(self, x):
        x = x[0]  # 输入的句子
        inputs = self.embedding(x).unsqueeze(1)  # (Batch_size, 1, pad_size, 每个词embedding的长度)   unsqueeze(num) 在指定位置(num)对数据进行扩充  squeeze(num)在指定位置(num)对数据进行压缩
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]  # 变换卷积核进行卷积
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs]  # 在i.size(2)维上做max-pooling
        concated = torch.cat(inputs, 1)  # 按行进行拼接
        concated = self.dropout(concated)  # (N,len(Ks)*Co)
        out = self.fc(concated)
        return out  # F.log_softmax(out,1)