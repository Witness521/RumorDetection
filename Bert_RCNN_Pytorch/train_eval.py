# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import precision_recall_fscore_support


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


'''正常划分训练集 验证集 测试集的训练方法'''
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    # Bert的encoder有12层，
    # 学习率不衰减的参数(集合)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # 分层权重衰减
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Adam优化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         # t_total的预热部分
                         warmup=0.05,
                         # 学习的训练步骤总数
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    # 最佳损失 初始设置为最大值
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    # 将模型设置为训练模式
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            # 将模型的参数梯度设置为0
            model.zero_grad()
            # 计算交叉熵损失
            loss = F.cross_entropy(outputs, labels)
            # 反向传播，计算当前梯度
            loss.backward()
            # 根据梯度更新网络参数
            optimizer.step()
            if total_batch % 25 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 如果验证集的loss下降，则标注*
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                # 如果验证集的loss没有下降，则不标注*
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Valid Loss: {3:>5.2},  Valid Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


'''K折验证的方式训练'''
def kFold_train(config, model, train_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    # Bert的encoder有12层，
    # 学习率不衰减的参数(集合)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # 分层权重衰减
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Adam优化器
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         # t_total的预热部分
                         warmup=0.05,
                         # 学习的训练步骤总数
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 记录进行到多少batch
    # 最佳损失
    best_loss = float('inf')
    # 将模型设置为训练模式
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            # 将模型的参数梯度设置为0
            model.zero_grad()
            # 计算交叉熵损失
            loss = F.cross_entropy(outputs, labels)
            # 反向传播，计算当前梯度
            loss.backward()
            # 根据梯度更新网络参数
            optimizer.step()
            if total_batch % 25 == 0:
                # 每25轮输出在训练集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), config.save_path)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%},  Time: {3}'
                print(msg.format(total_batch, loss.item(), train_acc, time_dif))
                model.train()
            total_batch += 1
    # 对模型进行测试
    test_acc, test_loss, pre, recall, f1, sup = test(config, model, test_iter)
    return test_acc, test_loss, pre, recall, f1, sup


'''测试模型'''
def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    # 测试状态
    # 框架会自动把BN和Dropout固定住
    model.eval()
    start_time = time.time()
    test_acc, test_loss, pre, recall, f1, sup, test_report = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    time_dif = get_time_dif(start_time)
    print("用时:", time_dif)
    return test_acc, test_loss, pre, recall, f1, sup


'''对验证集进行评估'''
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 默认不计算梯度
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # 输出output
            # print(outputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        pre, recall, f1, sup = precision_recall_fscore_support(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # 返回的数据依次为 准确度acc, loss, pre, recall, F1, support, repost(直接打印)
        return acc, loss_total / len(data_iter), pre, recall, f1, sup, report
    return acc, loss_total / len(data_iter)
