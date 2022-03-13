# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        # 模型名称
        self.model_name = 'bert'
        # 训练集
        self.train_path = dataset + '/data/train.txt'
        # 验证集
        self.dev_path = dataset + '/data/valid.txt'
        # 测试集
        self.test_path = dataset + '/data/test.txt'
        # class_list类别
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 5
        # mini-batch大小
        # self.batch_size = 64    # 训练使用
        self.batch_size = 2       # 正式使用模型
        # 每句话处理成的长度(短填长切)
        self.pad_size = 80
        # 学习率lr
        self.learning_rate = 5e-5
        # self.learning_rate = 0.001

        # self.bert_path = './bert_pretrain'
        # 暂时换成绝对路径，whole_model中路径读取不到
        self.bert_path = 'F:\\graduationDesign\\improve\\LSTM_Attention\\bert_pretrain'

        # 读bert文件
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # 词嵌入的长度
        self.hidden_size = 768
        # 卷积核尺寸
        self.filter_sizes = (2, 3, 4)
        # 卷积核数量(channels数)
        self.num_filters = 256
        self.dropout = 0.2
        # 隐藏层节点数
        self.rnn_hidden = 256
        self.num_layers = 2
        # 存储out矩阵
        self.save_out = 0

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # 从预训练模型中读取Bert模型
        self.bert = BertModel.from_pretrained(config.bert_path)
        # 设置梯度为True
        for param in self.bert.parameters():
            param.requires_grad = True
        # 设置双向的LSTM
        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.rnn_hidden,
                            num_layers=config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)

        # max pooling的窗口大小
        self.maxpool = nn.MaxPool1d(config.pad_size)

        # self.rnn_hidden = 256  self.hidden_size = 768
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # encoder_out(8,80,768)   text_cls(8,768)
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # out(8, 80, 512)
        out, _ = self.lstm(encoder_out)
        # 将两个Tensor按照第三维(厚度)的方向进行拼接   out(8, 80, 1280)
        out = torch.cat((encoder_out, out), 2)
        # 无变化
        out = F.relu(out)
        # out(8,1280,80)
        out = out.permute(0, 2, 1)
        # 在seq_len长度上做最大池化  out(8,1280)
        out = self.maxpool(out).squeeze()
        self.save_out = out
        out = self.fc(out)
        # 这个softmax函数的目的就是再多分类的时候将所有概率归一化处理，但是本课题是二分类的问题，因此只需要比较两个的大小即可
        # 在后面单条语句要输出概率的时候，就可以做Softmax函数处理
        # out = self.softmax(out)
        return out
