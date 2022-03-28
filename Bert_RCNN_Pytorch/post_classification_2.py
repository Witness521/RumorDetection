import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        return self.fc(x)


# 使用提取出的特征进行二分类
class Classification_2():
    def __init__(self):
        self.tensorLocation = './dataSet/saved_tensor/post_embedding.pt'
        # 模型训练结果保存
        self.save_path = './dataSet/saved_dict/post_cls.ckpt'
        self.class_list = ['non-rumor', 'rumor']
        self.post_label_list = []  # list中存储(tensor, label)
        self.post_label_list_batch = []  # list中存储(多个tensor, 多个label)

        # 训练过程中的参数
        self.learning_rate = 4e-4
        self.num_epochs = 7
        self.batch_size = 50

    '''
        对数据进行加载标记
        返回训练集 验证集 测试集 形式：(tensor, label)
    '''
    def deal_data(self):
        tensor = torch.load(self.tensorLocation)
        for i in range(len(tensor)):
            if i <= 1531:  # 从0-1531都是rumor
                self.post_label_list.append((tensor[i], torch.tensor([1])))
            else:
                self.post_label_list.append((tensor[i], torch.tensor([0])))
        # 把post_label_list进行打乱
        random.shuffle(self.post_label_list)
        # 按照8:1:1的比例划分训练集 验证集 测试集
        post_label_len = len(self.post_label_list)
        train_data = self.post_label_list[0:math.floor(0.8 * post_label_len)]
        valid_data = self.post_label_list[math.floor(0.8 * post_label_len):math.floor(0.9 * post_label_len)]
        test_data = self.post_label_list[math.floor(0.9 * post_label_len):]
        return train_data, valid_data, test_data

    # 按照self.batch_size的大小划分数据集
    def data_to_batch(self, dataset):
        for i in range(len(dataset)):
            if i % self.batch_size == 0:
                saved_post_emb = dataset[i][0].unsqueeze(0)
                saved_label = dataset[i][1].unsqueeze(0)
            elif (i+1) % self.batch_size == 0:
                saved_post_emb = torch.cat((saved_post_emb, dataset[i][0].unsqueeze(0)), dim=0)
                saved_label = torch.cat((saved_label, dataset[i][1].unsqueeze(0)), dim=1)
                saved_label = saved_label.squeeze(0)
                self.post_label_list_batch.append((saved_post_emb, saved_label))
            else:
                saved_post_emb = torch.cat((saved_post_emb, dataset[i][0].unsqueeze(0)), dim=0)
                saved_label = torch.cat((saved_label, dataset[i][1].unsqueeze(0)), dim=1)

            

    def train(self, model, train_data, valid_data, test_data):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            for (trains, labels) in self.post_label_list_batch:
                outputs = model(trains)
                # 将模型的参数梯度设置为0
                model.zero_grad()
                # outputs = torch.unsqueeze(outputs, 0)
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
                    valid_acc, valid_loss = self.evaluate(model, valid_data)
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(model.state_dict(), self.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%},  Valid Loss: {3:>5.2},  Valid Acc: {4:>6.2%} {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc, improve))
                    model.train()
                total_batch += 1
        # 对模型进行测试
        self.test(model, test_data)

    '''对验证集进行评估'''
    def evaluate(self, model, valid_data, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        # 默认不计算梯度
        with torch.no_grad():
            for (texts, labels) in valid_data:
                outputs = model(texts)
                # 输出output
                outputs = torch.unsqueeze(outputs, 0)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=self.class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(valid_data), report, confusion
        return acc, loss_total / len(valid_data)

    '''测试模型'''
    def test(self, model, test_data):
        model.load_state_dict(torch.load(self.save_path))
        # 测试状态
        # 框架会自动把BN和Dropout固定住
        model.eval()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(model, test_data, True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)


if __name__ == '__main__':
    classification = Classification_2()
    train_data, valid_data, test_data = classification.deal_data()
    classification.data_to_batch(train_data)
    # print(classification.post_label_list_batch[0][0].shape)
    model = Model()
    classification.train(model, train_data, valid_data, test_data)

