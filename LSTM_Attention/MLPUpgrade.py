import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from LSTM_Attention.models import TextRNN_Att
from sklearn import metrics
from tqdm import tqdm

# 定义TraindataSet
class TraindataSet(Dataset):
    def __init__(self, train_features, train_labels):
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class MLP():
    def __init__(self, dataset):
        self.TextRNN_Att_config = TextRNN_Att.Config(dataset, embedding='embedding_SougouNews.npz')

        # 初始化定义，在load_data方法中赋值
        self.x_length = 0
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

        # M是样本数量，input_size是输入层大小 hidden_size是隐含层大小，output_size是输出层大小
        self.M, self.input_size, self.hidden_size, self.output_size = self.x_length * 0.9, 1467, 150, 2

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

        # 定义损失函数
        # loss_fn = nn.MSELoss(reduction='sum')
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')

        ## 设置超参数 ##
        self.learning_rate = 1e-4
        self.weight_decay = 0.1
        self.batch_size = 256
        # EPOCH = 300
        self.EPOCH = 60

        # 设置一个最大的acc，如果比acc大则对模型进行保存
        self.biggest_accuracy = 1


    '''
        读取数据并赋值
    '''
    def load_data(self):
        x = torch.load('dataSet/saved_dict/save_text_review.pth').to(torch.float32)
        y = torch.load('dataSet/saved_dict/rumor_result.pth', ).to(torch.float32)
        # x和y的长度
        x_length = x.size()[0]
        # 为M赋值
        self.M = x_length * 0.9
        # 按照9:1的比例划分训练集和测试集
        self.x_train = x[:int(0.9 * x_length)]
        self.x_test = x[int(0.9 * x_length):x_length]
        self.y_train = y[:int(0.9 * x_length)]
        self.y_test = y[int(0.9 * x_length):x_length]


    '''
        对验证集进行验证
        返回验证集的准确率
    '''
    def valid_data(self, valid_features, valid_iter):
        self.model.eval()
        valid_correct = 0
        valid_total_loss = 0
        with torch.no_grad():
            for X, y in valid_iter:
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y).item()
                valid_total_loss += loss
                y_pred = y_hat.max(1, keepdim=True)[1]
                singleValLabel = y.max(1, keepdim=True)[1]
                # 计算预测正确的总数
                valid_correct += y_pred.eq(singleValLabel.view_as(y_pred)).sum().item()
        # 计算平均的loss
        valid_loss = valid_total_loss / len(valid_features)
        valid_accuracy = 100. * valid_correct / len(valid_features)
        print('验证集: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            valid_loss, valid_correct, len(valid_features), valid_accuracy))
        return valid_accuracy


    '''
        训练方法
    '''
    def train(self):
        # 定义训练集
        trainDataset = TraindataSet(self.x_train, self.y_train)
        train_iter = DataLoader(trainDataset, self.batch_size, shuffle=True)

        # 定义验证集
        validDataset = TraindataSet(self.x_test, self.y_test)
        valid_iter = DataLoader(validDataset, self.batch_size, shuffle=True)
        # 使用Adam优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # 记录10epoch总的准确度
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        # 记录Epoch和训练集的准确度来画图
        train_epoch_list = []
        train_accuracy_list = []
        # 记录验证集valid的准确率
        valid_accuracy_list = []
        valid_epoch_list = []
        # 全局变量
        # global biggest_accuracy

        # 开始训练
        for epoch in tqdm(range(1, self.EPOCH+1)):
            self.model.train()
            correct = 0  # 记录分类正确的个数
            i = 0  # 记录
            for X, y in train_iter:  # 分批训练
                i += 1
                # 向前传播
                y_pred = self.model(X)
                # 计算损失
                loss = self.loss_fn(y_pred, y)
                # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
                optimizer.zero_grad()
                # 计算梯度
                loss.backward()
                # 更新梯度
                optimizer.step()
                # 得到每个epoch的 loss 和 accuracy
                pred = y_pred.max(1, keepdim=True)[1]
                singleLabel = y.max(1, keepdim=True)[1]
                correct += pred.eq(singleLabel.view_as(pred)).sum().item()

            # 计算每一个epoch的训练集的准确率
            train_accuracy = 100.0 * correct / trainDataset.len
            print('Epoch: {}, Loss: {:.5f}, 训练集准确度: {}/{} ({:.3f}%)'.format(
                epoch, loss.item(), correct, trainDataset.len, train_accuracy))
            # 每十个EPOCH计算一下训练集平均的准确率
            train_accuracy_sum += train_accuracy  # 记录10个epoch总的accuracy
            if epoch % 10 == 0:
                # 每十个epoch记录训练集的平均准确率
                train_accuracy_list.append(train_accuracy_sum / 10)
                train_epoch_list.append(epoch)
                train_accuracy_sum = 0
            ####################################
            # 计算每一个epoch的测试集的准确率
            valid_accuracy_sum += self.valid_data(self.x_test, valid_iter)
            # 每10个epoch记录验证集上的准确度
            if epoch % 10 == 0:
                valid_average = valid_accuracy_sum / 10
                valid_accuracy_list.append(valid_average)
                valid_epoch_list.append(epoch)
                valid_accuracy_sum = 0
                # 如果准确度高于acc最大值则对模型进行保存
                if valid_average > self.biggest_accuracy:
                    torch.save(self.model.state_dict(), 'dataSet/saved_dict/MLP_update.ckpt')
                    self.biggest_accuracy = valid_average

        # 解决中文无法显示的问题
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 绘制train_acc的图像
        plt.plot(train_epoch_list, train_accuracy_list, color='red', label='训练集的准确度')
        plt.plot(valid_epoch_list, valid_accuracy_list, color='blue', label='验证集的准确度')
        plt.title('train&valid acc')
        plt.legend()  # 显示图例
        plt.xlabel('Epoch')
        plt.ylabel('accuracy %')
        plt.show()




    '''测试'''
    def test(self):
        self.model.load_state_dict(torch.load('dataSet/saved_dict/MLP_update.ckpt'))
        self.model.eval()
        y_pred = self.model(self.x_test)
        loss = self.loss_fn(y_pred, self.y_test)
        print(loss)
        # y_pred = nn.Softmax(y_pred)
        # print(y_pred)
        # 对数据进行处理
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        print(y_pred[0])
        print(self.y_test[0])
        for i in range(y_pred.size()[0]):
            pred = y_pred[i]
            yTest = self.y_test[i]
            # 增加一个维度
            pred = pred.unsqueeze(0)
            yTest = yTest.unsqueeze(0)
            predict = torch.max(pred.data, 1)[1].cpu().numpy()
            label = torch.max(yTest.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)
        # print(labels_all)
        # print(predict_all)

        # 输出测试的结果(详细)
        test_acc, test_loss, test_report, test_confusion = self.test_evaluate(predict_all, labels_all, loss)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)


    def test_evaluate(self, predict_all, labels_all, loss):
        acc = metrics.accuracy_score(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=self.TextRNN_Att_config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss.detach().numpy().tolist() / len(predict_all), report, confusion


    '''
        k折划分，第i折作为验证集
        返回X_train, y_train, X_valid, y_valid
    '''
    def get_k_fold_data(self, k, i, X, y):  ### 此过程主要是步骤（1）
        # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
        assert k > 1  # 如果k<=1则直接报错
        fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

        X_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
            # idx 为每组valid，对行进行切割
            X_part, y_part = X[idx, :], y[idx]
            # 第i折作valid
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
                y_train = torch.cat((y_train, y_part), dim=0)
        return X_train, y_train, X_valid, y_valid


    '''
        K折交叉验证
    '''
    def k_fold(self, k, X_train, y_train):
        train_loss_sum, valid_loss_sum = 0, 0
        train_acc_sum, valid_acc_sum = 0, 0

        for i in range(k):
            data = self.get_k_fold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据
            # model_ = model()  # 实例化模型
            # 每份数据进行训练,体现步骤三
            self.train()
        # train_ls, valid_ls =
        #     print('*' * 25, '第', i + 1, '折', '*' * 25)
        #     print('train_loss:%.6f' % train_ls[-1][0], 'train_acc:%.4f\n' % valid_ls[-1][1], \
        #           'valid loss:%.6f' % valid_ls[-1][0], 'valid_acc:%.4f' % valid_ls[-1][1])
        #     train_loss_sum += train_ls[-1][0]
        #     valid_loss_sum += valid_ls[-1][0]
        #     train_acc_sum += train_ls[-1][1]
        #     valid_acc_sum += valid_ls[-1][1]
        # print('#' * 10, '最终k折交叉验证结果', '#' * 10)
        # ####体现步骤四#####
        # print('train_loss_sum:%.4f' % (train_loss_sum / k), 'train_acc_sum:%.4f\n' % (train_acc_sum / k), \
        #       'valid_loss_sum:%.4f' % (valid_loss_sum / k), 'valid_acc_sum:%.4f' % (valid_acc_sum / k))


if __name__ == '__main__':
    mlp = MLP('dataSet')
    mlp.load_data()
    mlp.k_fold(3, mlp.x_train, mlp.y_train)
    mlp.test()
