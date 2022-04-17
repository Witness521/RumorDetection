import math
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from mlp_model import Model
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
import torch.nn.functional as F
from sklearn.model_selection import KFold
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
        self.learning_rate = 3e-4
        self.num_epochs = 9
        self.batch_size = 25
        self.net_tensor_num = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        # 声明列表存储(固定名称)
        self.post_label_list = []


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_channels=768, out_channels=768, heads=12, concat=False)
        self.conv2 = GATConv(in_channels=768, out_channels=768, heads=12, concat=False)
        self.mlp = Model()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # 将x按照batch进行竖向平均
        x = scatter_mean(x, data.batch, dim=0)
        x = self.mlp(x)
        return x


class GAT_Cls():
    def __init__(self, config):
        self.config = config
        self.loss_function = nn.CrossEntropyLoss()

    '''
        对数据进行加载并标记，只取前6行
        对tensor进行存储，(pair_tensor, label)
    '''
    def load_data(self):
        for i in tqdm(range(3371)):
            tensor_all = torch.load(self.config.tensorLocation + str(i) + '.pt')
            # 只取tensor的前6行，不足6行取整个tensor
            if tensor_all.shape[0] >= self.config.net_tensor_num:
                tensor_all = tensor_all[:self.config.net_tensor_num]
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

    '''构建dataLoader'''
    def build_dataLoader(self):
        self.load_data()
        train_data, valid_data, test_data = self.divide_data()
        train_dataloader = self.build_data(train_data)
        valid_dataloader = self.build_data(valid_data)
        test_dataloader = self.build_data(test_data)
        return train_dataloader, valid_dataloader, test_dataloader

    '''8:1:1 训练方法'''
    def train(self, train_dataloader, valid_dataloader, test_dataloader):
        model = Net()
        # 模型放置GPU
        model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            for i, data in enumerate(train_dataloader):
                model.train()
                model.zero_grad()
                # 数据放置在GPU
                data.to(self.config.device)
                preds = model(data)
                loss = self.loss_function(preds, data.y)
                # 反向传播
                loss.backward()
                optimizer.step()
                if total_batch % 25 == 0 and total_batch != 0:
                    # 训练集准确率
                    true = data.y.cpu()
                    preds = torch.max(preds.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, preds)
                    valid_acc, valid_loss = self.evaluate(model, valid_dataloader)
                    if valid_loss < best_loss:
                        best_loss = valid_loss
                        torch.save(model.state_dict(), self.config.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%},  Valid Loss: {3:>5.2},  Valid Acc: {4:>6.2%} {5}'
                    print(msg.format(total_batch, loss.item(), train_acc, valid_loss, valid_acc, improve))
                total_batch += 1
        # 对模型进行测试
        self.test(model, test_dataloader)

    '''5折交叉验证########################################################################################'''
    def build_kFold_dataSet(self):
        self.load_data()
        kf = KFold(n_splits=5, shuffle=False)
        fold = 1
        # 测试集的平均acc和loss
        test_average_acc = 0
        test_average_loss = 0
        test_all_pre = 0
        test_all_recall = 0
        test_all_f1 = 0
        for train_index, test_index in kf.split(self.config.post_label_list):
            print('**********Fold ' + str(fold) + '**********')
            # 根据第i折的索引获取data
            train_data = np.array(self.config.post_label_list)[train_index]
            test_data = np.array(self.config.post_label_list)[test_index]
            train_dataLoader = self.build_data(train_data)
            test_dataLoader = self.build_data(test_data)
            # k折
            test_acc, test_loss, pre, recall, f1, sup = self.kFold_train(train_dataLoader, test_dataLoader)
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
        print('Fake: pre:{0:>6.2%},rec:{1:>6.2%},f1:{2:>6.2%}'.format((test_all_pre / 5)[1], (test_all_recall / 5)[1],
                                                                      (test_all_f1 / 5)[1]))
        print('Real: pre:{0:>6.2%},rec:{1:>6.2%},f1:{2:>6.2%}'.format((test_all_pre / 5)[0], (test_all_recall / 5)[0],
                                                                      (test_all_f1 / 5)[0]))
        print('Total: pre:{0:>6.2%},rec:{1:>6.2%},f1:{2:>6.2%}'.format(np.mean(test_all_pre / 5), np.mean(test_all_recall / 5),
                                                                       np.mean(test_all_f1 / 5)))

    '''5折验证的方法进行训练'''
    def kFold_train(self, train_dataset, test_dataset):
        # 每次重新装载model
        model = Net()
        # 模型放置GPU
        model.to(self.config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        total_batch = 0  # 记录进行到多少batch
        # 最佳损失
        best_loss = float('inf')
        for epoch in range(self.config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.num_epochs))
            for data in train_dataset:
                model.train()
                # 将模型的参数梯度设置为0
                model.zero_grad()
                data.to(self.config.device)
                preds = model(data)
                # 计算交叉熵损失
                loss = self.loss_function(preds, data.y)
                # 反向传播，计算当前梯度
                loss.backward()
                # 根据梯度更新网络参数
                optimizer.step()
                if total_batch % 50 == 0:
                    # 训练集准确率
                    true = data.y.cpu()
                    preds = torch.max(preds.data, 1)[1].cpu()
                    # 训练集准确率
                    train_acc = metrics.accuracy_score(true, preds)
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(model.state_dict(), self.config.save_path)
                        improve = '*'
                    else:
                        improve = ''
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>7.2%} {3}'
                    print(msg.format(total_batch, loss.item(), train_acc, improve))
                total_batch += 1
        # 对模型进行测试
        test_acc, test_loss, pre, recall, f1, sup = self.test(model, test_dataset)
        return test_acc, test_loss, pre, recall, f1, sup
    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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

    '''执行'''
    def execute(self):
        isKFold = input("请选择是否使用K折验证(Y/N)")
        if isKFold in ['Y', 'y']:  # 使用K折验证
            self.build_kFold_dataSet()
        elif isKFold in ['N', 'n']:  # 不使用K折验证
            train_dataloader, valid_dataloader, test_dataloader = self.build_dataLoader()
            self.train(train_dataloader, valid_dataloader, test_dataloader)

if __name__ == '__main__':
    cls = GAT_Cls(Config())
    cls.execute()