# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, kFold_train
from importlib import import_module
import argparse
from utils import BuildDataSet, build_iterator, get_time_dif
import models.lstm_word2vec as lstm_word2vec
import models.TextCNN as TextCNN
import models.bert_RCNN as bertRCNN

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'dataSet'  # 数据集

    # lstm_word2vec
    config = lstm_word2vec.Config(dataset)
    # 将模型加载到GPU上
    model = lstm_word2vec.Model(config).to(config.device)
    # 设置随机数的种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


    isKFold = input("请选择是否使用K折验证(Y/N)")
    if isKFold in ['Y', 'y']:  # 使用K折验证
        print("加载数据...")
        # 测试集的平均acc和loss
        test_average_acc = 0
        test_average_loss = 0
        # 记录第几个Fold
        fold = 1
        # 构建数据集
        fold_dataset = BuildDataSet(config).build_kFold_dataset()   # [[train1, test1], [train2, test2]...]
        for train_data, test_data in fold_dataset:
            print('**********Fold ' + str(fold) + '**********')
            train_iter = build_iterator(train_data, config)
            test_iter = build_iterator(test_data, config)
            # k折划分训练
            test_acc, test_loss = kFold_train(config, model, train_iter, test_iter)
            test_average_acc += test_acc
            test_average_loss += test_loss
            # 重新装载model
            model = lstm_word2vec.Model(config).to(config.device)
            fold += 1
            print(end='\n\n')
        print("5-Fold Test Acc:{0:>7.2%}".format(test_average_acc / 5))
        print("5-Fold Test Loss:{0:>5.2}".format(test_average_loss / 5))

    elif isKFold in ['N', 'n']:  # 不使用K折验证
        print("加载数据...")
        # 构建数据集(形式：(token_id:list, 类别的标签:0/1, 长度:int, mask:list) )
        train_data, valid_data, test_data = BuildDataSet(config).build_dataset()
        # 返回一个DataSetIterator
        train_iter = build_iterator(train_data, config)
        valid_iter = build_iterator(valid_data, config)
        test_iter = build_iterator(test_data, config)

        # 模型训练部分
        train(config, model, train_iter, valid_iter, test_iter)
    else:
        raise Exception("请输入Y或者N:")



