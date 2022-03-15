# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import BuildDataSet, build_iterator, get_time_dif
import models.lstm_word2vec as lstm_word2vec
import models.TextCNN as TextCNN

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'dataSet'  # 数据集

    # TextCNN
    config = TextCNN.Config(dataset)
    # 设置随机数的种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("加载数据...")
    # 构建数据集(形式：(token_id:list, 类别的标签:0/1, 长度:int, mask:list) )
    train_data, valid_data, test_data = BuildDataSet(config).build_dataset()
    # 返回一个DataSetIterator
    train_iter = build_iterator(train_data, config)
    valid_iter = build_iterator(valid_data, config)
    test_iter = build_iterator(test_data, config)

    # 模型训练部分
    # 将模型加载到GPU上
    # model = bert.Model(config).to(config.device)
    model = TextCNN.Model(config).to(config.device)
    train(config, model, train_iter, valid_iter, test_iter)
