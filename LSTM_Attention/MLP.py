import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from LSTM_Attention.models import TextRNN_Att
from sklearn import metrics
from tqdm import tqdm

dataset = 'dataSet'
config = TextRNN_Att.Config(dataset, embedding='embedding_SougouNews.npz')

# 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
# 我们不需要求这几个 Variable 的导数
# x = Variable(torch.randn(M, input_size))
# y = Variable(torch.randn(M, output_size))
x = torch.load('dataSet/saved_dict/save_text_review.pth').to(torch.float32)
y = torch.load('dataSet/saved_dict/rumor_result.pth', ).to(torch.float32)
# x和y的长度
x_length = x.size()[0]
# 按照9:1的比例划分训练集和测试集
x_train = x[:int(0.9 * x_length)]
x_test = x[int(0.9 * x_length):x_length]
y_train = y[:int(0.9 * x_length)]
y_test = y[int(0.9 * x_length):x_length]

# M是样本数量，input_size是输入层大小
# hidden_size是隐含层大小，output_size是输出层大小
# M, input_size, hidden_size, output_size = 200, 1000, 100, 2
M, input_size, hidden_size, output_size = x_length * 0.9, 1467, 150, 2

# 使用 nn 包的 Sequential 来快速构建模型，Sequential可以看成一个组件的容器。
# 它涵盖神经网络中的很多层，并将这些层组合在一起构成一个模型.
# 之后，我们输入的数据会按照这个Sequential的流程进行数据的传输，最后一层就是输出层。
# 默认会帮我们进行参数初始化
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
)

# 定义损失函数
# loss_fn = nn.MSELoss(reduction='sum')
loss_fn = nn.CrossEntropyLoss(reduction='sum')

## 设置超参数 ##
learning_rate = 1e-4
weight_decay = 0.1
EPOCH = 15000

'''
    训练方法
'''
def train():
    # 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
    # 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    miniLoss = 10000
    # 记录Epoch和loss以画图
    epochList = []
    lossList = []

    ## 开始训练 ##
    for t in tqdm(range(EPOCH)):
        # 向前传播
        y_pred = model(x_train)
        # 计算损失
        loss = loss_fn(y_pred, y_train)
        # 显示损失
        if (t+1) % 50 == 0:
            # 处理损失loss
            loss_float = loss.detach().numpy().tolist()  # 转化为float
            loss_float = loss_float / (x_length * 0.9)   # 求平均值
            # 添加到画图的list中
            epochList.append(t+1)
            lossList.append(loss_float)
            if loss_float < miniLoss:
                # 保存模型
                torch.save(model.state_dict(), 'dataSet/saved_dict/MLP.ckpt')
                miniLoss = loss_float
        # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()
    plt.plot(epochList, lossList, 'r--')
    plt.title('train-loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def test():
    model.load_state_dict(torch.load('dataSet/saved_dict/MLP.ckpt'))
    model.eval()
    y_pred = model(x_test)
    loss = loss_fn(y_pred, y_test)
    print(loss)
    # y_pred = nn.Softmax(y_pred)
    # print(y_pred)
    # 对数据进行处理
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    print(y_pred[0])
    print(y_test[0])
    for i in range(y_pred.size()[0]):
        pred = y_pred[i]
        yTest = y_test[i]
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
    test_acc, test_loss, test_report, test_confusion = test_evaluate(predict_all, labels_all, loss)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def test_evaluate(predict_all, labels_all, loss):
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    return acc, loss.detach().numpy().tolist() / len(predict_all), report, confusion


if __name__ == '__main__':
    train()
    test()

