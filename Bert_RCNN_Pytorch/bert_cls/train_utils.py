import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

class TrainUtils():

    '''正常8:1:1划分数据集的训练方法'''
    def train(self, config, model, train_iter, valid_iter, test_iter):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            for i, (trains, labels) in enumerate(train_iter):
                outputs = model(trains)
                # 将模型的参数梯度设置为0
                model.zero_grad()
                # 将labels从(60, 1)变成(60,)
                labels = torch.squeeze(labels, 1)
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
                    valid_acc, valid_loss = self.evaluate(config, model, valid_iter)
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%},  Valid Loss: {3:>5.2},  Valid Acc: {4:>6.2%} {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc, improve))
                    model.train()
                total_batch += 1
        # 对模型进行测试
        self.test(config, model, test_iter)

    '''5折验证的方法进行训练'''
    def kFold_train(self, config, model, train_iter, test_iter):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            for (trains, labels) in train_iter:
                outputs = model(trains)
                # 将模型的参数梯度设置为0
                model.zero_grad()
                # 将labels从(60, 1)变成(60,)
                labels = torch.squeeze(labels, 1)
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
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%} {3}'
                    print(msg.format(total_batch, loss.item(), train_acc, improve))
                    model.train()
                total_batch += 1
        # 对模型进行测试
        test_acc, test_loss, pre, recall, f1, sup = self.test(config, model, test_iter)
        return test_acc, test_loss, pre, recall, f1, sup

    '''对验证集进行评估'''
    def evaluate(self, config, model, valid_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        # 默认不计算梯度
        with torch.no_grad():
            for (texts, labels) in valid_iter:
                outputs = model(texts)
                # 将labels从(60, 1)变成(60,)
                labels = torch.squeeze(labels, 1)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
        # 正确的比例
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            pre, recall, f1, sup = precision_recall_fscore_support(labels_all, predict_all)
            report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
            # 返回的数据依次为 准确度acc, loss, pre, recall, F1, support, repost(直接打印)
            return acc, loss_total / len(valid_iter), pre, recall, f1, sup, report
        return acc, loss_total / len(valid_iter)

    '''测试模型'''
    def test(self, config, model, test_iter):
        model.load_state_dict(torch.load(config.save_path))
        # 测试状态
        # 框架会自动把BN和Dropout固定住
        model.eval()
        test_acc, test_loss, pre, recall, f1, sup, test_report = self.evaluate(config, model, test_iter, True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        return test_acc, test_loss, pre, recall, f1, sup

'''
    自定义dataset(封装(data, label))
    实现torch.utils下的Dataset接口，为实现DataLoader
'''
class TrainDataSet(Dataset):
    def __init__(self, data_label_list):
        self.data_label_list = data_label_list
        self.len = len(data_label_list)

    def __getitem__(self, index):
        return self.data_label_list[index][0], self.data_label_list[index][1]

    def __len__(self):
        return self.len