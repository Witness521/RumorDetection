import math
import numpy as np
import random
import torch
from mlp_model import Model
from train_utils import TrainUtils
from torch.utils.data import DataLoader
from train_utils import TrainDataSet
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')
# 设置随机种子
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

class Config():
    """配置参数"""
    def __init__(self):
        # tensor储存的位置
        self.tensorLocation = '../dataSet/saved_tensor/post_embedding.pt'
        # 模型训练结果保存
        self.save_path = '../dataSet/saved_dict/post_cls.ckpt'
        self.class_list = ['Real', 'Fake']
        # 训练过程中的参数
        self.learning_rate = 3e-4
        self.num_epochs = 9
        self.batch_size = 128
        # 声明列表存储(名称固定)
        self.post_label_list = []  # list中存储(tensor, label)


# 使用提取出的特征进行二分类
class Classification():
    def __init__(self, config):
        self.config = config
        self.trainUtils = TrainUtils()

    '''对数据进行加载标记'''
    def load_data(self):
        tensor = torch.load(self.config.tensorLocation)
        for i in range(len(tensor)):
            if i <= 1531:  # 从0-1531都是rumor
                self.config.post_label_list.append((tensor[i], torch.tensor([1])))
            else:
                self.config.post_label_list.append((tensor[i], torch.tensor([0])))
        # 把post_label_list进行打乱
        random.shuffle(self.config.post_label_list)

    '''返回训练集 验证集 测试集 形式：(tensor, label)'''
    def divide_data(self):
        # 按照8:1:1的比例划分训练集 验证集 测试集
        post_label_len = len(self.config.post_label_list)
        train_data = self.config.post_label_list[0:math.floor(0.8 * post_label_len)]
        valid_data = self.config.post_label_list[math.floor(0.8 * post_label_len):math.floor(0.9 * post_label_len)]
        test_data = self.config.post_label_list[math.floor(0.9 * post_label_len):]
        # 构建训练集 验证集 测试集的迭代器
        trainDataset = TrainDataSet(train_data)
        train_iter = DataLoader(trainDataset, self.config.batch_size, shuffle=False)
        validDataset = TrainDataSet(valid_data)
        valid_iter = DataLoader(validDataset, self.config.batch_size, shuffle=False)
        testDataset = TrainDataSet(test_data)
        test_iter = DataLoader(testDataset, self.config.batch_size, shuffle=False)
        return train_iter, valid_iter, test_iter

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
        for train_index, test_index in kf.split(self.config.post_label_list):
            print('**********Fold ' + str(fold) + '**********')
            # 每次循环都重新装载model
            model = Model()
            # 根据第i折的索引获取data
            train_data = np.array(self.config.post_label_list)[train_index]
            test_data = np.array(self.config.post_label_list)[test_index]
            trainDataset = TrainDataSet(train_data)
            train_iter = DataLoader(trainDataset, self.config.batch_size, shuffle=False)
            testDataset = TrainDataSet(test_data)
            test_iter = DataLoader(testDataset, self.config.batch_size, shuffle=False)
            # k折
            test_acc, test_loss, pre, recall, f1, sup = self.trainUtils.kFold_train(self.config, model, train_iter, test_iter)
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

    def execute(self):
        self.load_data()
        model = Model()
        isKFold = input("请选择是否使用K折验证(Y/N)")
        if isKFold in ['Y', 'y']:  # 使用K折验证
            self.build_kFold_dataSet()
        elif isKFold in ['N', 'n']:  # 不使用K折验证
            train_iter, valid_iter, test_iter = self.divide_data()
            self.trainUtils.train(self.config, model, train_iter, valid_iter, test_iter)

if __name__ == '__main__':
    classification = Classification(Config())
    classification.execute()