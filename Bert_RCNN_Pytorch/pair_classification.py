import math
import numpy as np
import random
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

# 设置随机种子
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        return self.fc(x)

class Pair_classification():
    def __init__(self):
        self.tensorLocation = './dataSet/saved_tensor/post_review/'
        # 模型训练结果保存
        self.save_path = './dataSet/saved_dict/pair_cls.ckpt'
        self.class_list = ['non-rumor', 'rumor']
        self.pair_tensor = []
        self.pair_label_list_batch = []  # list中存储(多个tensor, 多个label)

        # 训练过程中的参数
        self.learning_rate = 8e-4
        self.num_epochs = 10
        self.batch_size = 50

    # 将所有的tensor进行竖向平均，然后把所有平均后的tensor进行存储，(pair_tensor, label)
    def deal_the_data(self):
        for i in tqdm(range(3371)):
            tensor_all = torch.load(self.tensorLocation + str(i) + '.pt')
            tensor_average = torch.mean(tensor_all, dim=0)
            if i <= 1531:
                self.pair_tensor.append((tensor_average, torch.tensor([1])))
            else:
                self.pair_tensor.append((tensor_average, torch.tensor([0])))
        # 把post_label_list进行打乱
        random.shuffle(self.pair_tensor)
        # 按照8:1:1的比例划分训练集 验证集 测试集
        pair_label_len = len(self.pair_tensor)
        train_data = self.pair_tensor[0:math.floor(0.8 * pair_label_len)]
        valid_data = self.pair_tensor[math.floor(0.8 * pair_label_len):math.floor(0.9 * pair_label_len)]
        test_data = self.pair_tensor[math.floor(0.9 * pair_label_len):]
        return train_data, valid_data, test_data

    # 按照self.batch_size的大小划分数据集
    def data_to_batch(self, dataset):
        for i in range(len(dataset)):
            if i % self.batch_size == 0:
                saved_post_emb = dataset[i][0].unsqueeze(0)
                saved_label = dataset[i][1].unsqueeze(0)
            elif (i + 1) % self.batch_size == 0:
                saved_post_emb = torch.cat((saved_post_emb, dataset[i][0].unsqueeze(0)), dim=0)
                saved_label = torch.cat((saved_label, dataset[i][1].unsqueeze(0)), dim=1)
                saved_label = saved_label.squeeze(0)
                self.pair_label_list_batch.append((saved_post_emb, saved_label))
            else:
                saved_post_emb = torch.cat((saved_post_emb, dataset[i][0].unsqueeze(0)), dim=0)
                saved_label = torch.cat((saved_label, dataset[i][1].unsqueeze(0)), dim=1)

    def train(self, model, valid_data, test_data):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        for epoch in range(self.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.num_epochs))
            for (trains, labels) in self.pair_label_list_batch:
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
    classification = Pair_classification()
    train_data, valid_data, test_data = classification.deal_the_data()
    classification.data_to_batch(train_data)
    model = Model()
    classification.train(model, valid_data, test_data)
