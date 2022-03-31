import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')
# 设置随机种子
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 使用提取出的特征进行二分类
class Classification_2():
    def __init__(self):
        self.tensorLocation = './dataSet/saved_tensor/post_embedding.pt'
        # 模型训练结果保存
        self.save_path = './dataSet/saved_dict/post_cls.ckpt'
        self.class_list = ['Real', 'Fake']
        self.post_label_list = []  # list中存储(tensor, label)
        self.post_label_list_batch = []  # list中存储(多个tensor, 多个label)

        # 训练过程中的参数
        self.learning_rate = 4e-4
        self.num_epochs = 6
        self.batch_size = 50

    '''对数据进行加载标记'''
    def load_data(self):
        tensor = torch.load(self.tensorLocation)
        for i in range(len(tensor)):
            if i <= 1531:  # 从0-1531都是rumor
                self.post_label_list.append((tensor[i], torch.tensor([1])))
            else:
                self.post_label_list.append((tensor[i], torch.tensor([0])))
        # 把post_label_list进行打乱
        random.shuffle(self.post_label_list)


    '''返回训练集 验证集 测试集 形式：(tensor, label)'''
    def divide_data(self):
        # 按照8:1:1的比例划分训练集 验证集 测试集
        post_label_len = len(self.post_label_list)
        train_data = self.post_label_list[0:math.floor(0.8 * post_label_len)]
        valid_data = self.post_label_list[math.floor(0.8 * post_label_len):math.floor(0.9 * post_label_len)]
        test_data = self.post_label_list[math.floor(0.9 * post_label_len):]
        return train_data, valid_data, test_data

    '''按照self.batch_size的大小划分数据集'''
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

    '''正常8:1:1划分数据集的训练方法'''
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

    '''5折交叉验证'''
    def build_kFold_dataSet(self):
        kf = KFold(n_splits=5, shuffle=False)
        fold = 1
        # 测试集的平均acc和loss
        test_average_acc = 0
        test_average_loss = 0
        test_all_pre = 0
        test_all_recall = 0
        test_all_f1 = 0
        for train_index, test_index in kf.split(self.post_label_list):
            print('**********Fold ' + str(fold) + '**********')
            # 每次循环都重新装载model
            model = Model()
            # 根据第i折的索引获取data
            train_data = np.array(self.post_label_list)[train_index]
            test_data = np.array(self.post_label_list)[test_index]
            self.data_to_batch(train_data)
            # k折
            test_acc, test_loss, pre, recall, f1, sup = self.kFold_train(model, test_data)
            # 将数据累加以计算平均值
            test_average_acc += test_acc
            test_average_loss += test_loss
            test_all_pre += pre
            test_all_recall += recall
            test_all_f1 += f1
            ####
            fold += 1
            print(end='\n\n')
        # 打印K折的平均准确度和损失
        print("5-Fold Test Acc:{0:>7.2%}".format(test_average_acc / 5))
        print("5-Fold Test Loss:{0:>5.2}".format(test_average_loss / 5))
        print('Fake: pre:{0:>6.2%},rec:{1:>6.2%},f1:{2:>6.2%}'.format((test_all_pre / 5)[1], (test_all_recall / 5)[1], (test_all_f1 / 5)[1]))
        print('Real: pre:{0:>6.2%},rec:{1:>6.2%},f1:{2:>6.2%}'.format((test_all_pre / 5)[0], (test_all_recall / 5)[0], (test_all_f1 / 5)[0]))
        print('Total: pre:{0:>6.2%},rec:{1:>6.2%},f1:{2:>6.2%}'.format(np.mean(test_all_pre / 5), np.mean(test_all_recall / 5), np.mean(test_all_f1 / 5)))

    '''5折验证的方法进行训练'''
    def kFold_train(self, model, test_data):
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
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(model.state_dict(), self.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%} {3}'
                    print(msg.format(total_batch, loss.item(), train_acc, improve))
                    model.train()
                total_batch += 1
        # 对模型进行测试
        test_acc, test_loss, pre, recall, f1, sup = self.test(model, test_data)
        return test_acc, test_loss, pre, recall, f1, sup

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
        # 正确的比例
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            pre, recall, f1, sup = precision_recall_fscore_support(labels_all, predict_all)
            report = metrics.classification_report(labels_all, predict_all, target_names=self.class_list, digits=4)
            # 返回的数据依次为 准确度acc, loss, pre, recall, F1, support, repost(直接打印)
            return acc, loss_total / len(valid_data), pre, recall, f1, sup, report
        return acc, loss_total / len(valid_data)

    '''测试模型'''
    def test(self, model, test_data):
        model.load_state_dict(torch.load(self.save_path))
        # 测试状态
        # 框架会自动把BN和Dropout固定住
        model.eval()
        test_acc, test_loss, pre, recall, f1, sup, test_report = self.evaluate(model, test_data, True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        return test_acc, test_loss, pre, recall, f1, sup

    def execute(self):
        self.load_data()
        model = Model()
        isKFold = input("请选择是否使用K折验证(Y/N)")
        if isKFold in ['Y', 'y']:  # 使用K折验证
            self.build_kFold_dataSet()
        elif isKFold in ['N', 'n']:  # 不使用K折验证
            train_data, valid_data, test_data = self.divide_data()
            # 将train_data变成batch
            self.data_to_batch(train_data)
            self.train(model, train_data, valid_data, test_data)

if __name__ == '__main__':
    classification = Classification_2()
    classification.execute()
