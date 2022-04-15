import math
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
import torch.nn.functional as F
from mlp_model import Model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

# 设置随机种子
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

class Config():
    """配置参数"""
    def __init__(self):
        # tensor储存的位置
        self.tensorLocation = '../dataSet/saved_tensor/post_review/'
        # 模型训练结果保存
        self.save_path = '../dataSet/saved_dict/pair_cls.ckpt'
        self.class_list = ['Real', 'Fake']
        # 训练过程中的参数
        self.learning_rate = 5e-4
        self.num_epochs = 3
        self.batch_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # 声明列表存储(固定名称)
        self.post_label_list = []


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_channels=768, out_channels=512, heads=4, concat=False)
        self.conv2 = GATConv(in_channels=512, out_channels=256, heads=4, concat=False)
        self.fc1 = nn.Linear(256, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # 将x按照batch进行竖向平均
        x = scatter_mean(x, data.batch, dim=0)
        x = self.fc1(x)
        return x


class GAT_Cls():
    def __init__(self, config):
        self.config = config
        self.loss_function = nn.CrossEntropyLoss()

    '''
        对数据进行加载并标记
        对tensor进行存储，(pair_tensor, label)
    '''
    def load_data(self):
        for i in tqdm(range(3371)):
            tensor_all = torch.load(self.config.tensorLocation + str(i) + '.pt')
            if i <= 1531:
                self.config.post_label_list.append((tensor_all, torch.tensor([1], dtype=torch.long)))
            else:
                self.config.post_label_list.append((tensor_all, torch.tensor([0], dtype=torch.long)))
        # 把post_label_list进行打乱
        random.shuffle(self.config.post_label_list)

    '''返回训练集 验证集 测试集 形式：(tensor, label)'''
    def divide_data(self):
        # 按照8:1:1的比例划分训练集 验证集 测试集
        pair_label_len = len(self.config.post_label_list)
        train_data = self.config.post_label_list[0:math.floor(0.8 * pair_label_len)]
        valid_data = self.config.post_label_list[math.floor(0.8 * pair_label_len):math.floor(0.9 * pair_label_len)]
        test_data = self.config.post_label_list[math.floor(0.9 * pair_label_len):]
        return train_data, valid_data, test_data

    '''返回pyg格式的DataLoader'''
    def build_data(self, dataset):
        # data_list
        data_list = []
        # 构建边的list
        src_id = []
        target_id = []
        for (node_features, label) in dataset:
            # 构建edge_index
            for i in range(node_features.shape[0]):
                src_id.extend([i] * node_features.shape[0])
                target_id.extend(list(range(node_features.shape[0])))
            edge_index = torch.tensor([src_id, target_id], dtype=torch.long)
            data = Data(x=node_features, edge_index=edge_index, y=label)
            data_list.append(data)
            # 清空这两个构建边的list
            src_id.clear()
            target_id.clear()
        return DataLoader(data_list, batch_size=self.config.batch_size, shuffle=False)

    # 构建dataLoader
    def build_dataLoader(self):
        self.load_data()
        train_data, valid_data, test_data = self.divide_data()
        train_dataloader = self.build_data(train_data)
        valid_dataloader = self.build_data(valid_data)
        test_dataloader = self.build_data(test_data)
        return train_dataloader, valid_dataloader, test_dataloader

    # 训练方法
    def train(self, train_dataloader, valid_dataloader, test_dataloader):
        model = Net()
        # 模型放置GPU
        model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        # 每隔多少个batch测试验证集
        interval_batch = 300
        # 将训练集的loss相加
        train_total_loss = 0
        train_true = []
        train_pred = []
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            for i, data in enumerate(train_dataloader):
                model.train()
                model.zero_grad()
                # 数据放置在GPU
                data.to(self.config.device)
                preds = model(data)
                loss = self.loss_function(preds, data.y)
                # 累加train的loss
                train_total_loss += loss.item()
                train_true.extend(data.y.cpu().numpy())
                train_pred.extend(torch.max(preds.data, 1)[1].cpu().numpy())
                # 反向传播
                loss.backward()
                optimizer.step()
                if total_batch % interval_batch == 0 and total_batch != 0:
                    # 训练集准确率
                    train_acc = metrics.accuracy_score(train_true, train_pred)
                    valid_acc, valid_loss = self.evaluate(model, valid_dataloader)
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(model.state_dict(), self.config.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%},  Valid Loss: {3:>5.2},  Valid Acc: {4:>6.2%} {5}'
                    print(msg.format(total_batch, train_total_loss / interval_batch, train_acc, valid_loss, valid_acc, improve))
                    model.train()
                total_batch += 1
        # 对模型进行测试
        self.test(model, test_dataloader)

    # 评估方法
    def evaluate(self, model, valid_dataLoader, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        # 默认不计算梯度
        with torch.no_grad():
            for data in valid_dataLoader:
                # 数据放置在gpu
                data.to(self.config.device)
                # pred labels
                pred = model(data)
                labels = data.y
                loss = F.cross_entropy(pred, labels)
                loss_total += loss.item()
                # pred labels变成numpy
                predic = torch.max(pred.data, 1)[1].cpu().numpy()
                labels = labels.cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
        # 正确的比例
        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            pre, recall, f1, sup = precision_recall_fscore_support(labels_all, predict_all)
            report = metrics.classification_report(labels_all, predict_all, target_names=self.config.class_list, digits=4)
            # 返回的数据依次为 准确度acc, loss, pre, recall, F1, support, repost(直接打印)
            return acc, loss_total / len(valid_dataLoader), pre, recall, f1, sup, report
        return acc, loss_total / len(valid_dataLoader)

    '''测试模型'''
    def test(self, model, test_dataLoader):
        model.load_state_dict(torch.load(self.config.save_path))
        # 测试状态
        # 框架会自动把BN和Dropout固定住
        model.eval()
        test_acc, test_loss, pre, recall, f1, sup, test_report = self.evaluate(model, test_dataLoader, True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        return test_acc, test_loss, pre, recall, f1, sup

if __name__ == '__main__':
    cls = GAT_Cls(Config())
    train_dataloader, valid_dataloader, test_dataloader = cls.build_dataLoader()
    cls.train(train_dataloader, valid_dataloader, test_dataloader)

