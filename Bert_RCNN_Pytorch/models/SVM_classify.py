import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import *

class SVM_classify:
    def __init__(self):
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        # 记录acc
        self.average_acc = 0
        # 数据集的位置
        self.dataLocation = '../dataSet/data/Chinese_Rumor_dataset_clean.xls'

    '''正常读取数据集'''
    def readData(self):
        # 读取数据集
        df = pd.read_excel(self.dataLocation)
        # 划分数据集
        # 训练集占总体的80%
        train_data = df.sample(frac=0.8, replace=False, random_state=0, axis=0)
        # 测试集占20%
        test_data = df[~df.index.isin(train_data.index)]

        self.x_train = train_data['post']
        self.x_test = test_data['post']

        self.y_train = train_data['label']
        self.y_test = test_data['label']

    '''k折交叉验证'''
    def kFold_readData(self):
        # 记录第几个Fold
        fold = 1
        # 读取数据集
        df = pd.read_excel(self.dataLocation)
        kf = KFold(n_splits=5, shuffle=True)  # , random_state=42
        # 切割成五个数据集
        for train_index, test_index in kf.split(df):
            print('**********Fold ' + str(fold) + '**********')
            # 根据index获取对应的data(记录一下dataframe格式的数据用iloc切片)
            train_data, test_data = df.iloc[train_index.tolist(), :], df.iloc[test_index.tolist(), :]
            # 对train_data和test_data进行shuffle操作
            train_data = train_data.sample(frac=1, random_state=0)
            test_data = test_data.sample(frac=1, random_state=0)
            self.x_train = train_data['post']
            self.x_test = test_data['post']
            self.y_train = train_data['label']
            self.y_test = test_data['label']
            # 调用SVM方法
            self.svmClassify()
            fold += 1
        print('Test Average Acc:{0:>6.2%}'.format(self.average_acc / 5))


    # 处理样本
    def svmClassify(self):
        # 默认不去停用词的向量化
        # 使用词频的方式，使用矩阵来表示这句话 参考： https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vec = CountVectorizer()
        # 计算各个词语出现的次数
        x_count_train = count_vec.fit_transform(self.x_train)
        # print(x_count_train.toarray())
        x_count_test = count_vec.transform(self.x_test)

        # 去除停用词
        # count_stop_vec = CountVectorizer(analyzer='word', stop_words='english')
        # x_count_stop_train = count_stop_vec.fit_transform(self.x_train)
        # x_count_stop_test = count_stop_vec.transform(self.x_test)

        ## 模型训练
        svc = SVC()
        svc.fit(x_count_train, self.y_train)
        y_predict = svc.predict(x_count_test)
        # 准确率
        acc = svc.score(x_count_test, self.y_test)
        self.average_acc += acc
        print('test acc:{0:>6.2%}'.format(acc))
        target_name = ['non-rumor', 'rumor']
        print(classification_report(self.y_test, y_predict, target_names=target_name))

        # ## TF−IDF处理后在训练
        # ## 默认配置不去除停用词
        # tfid_vec = TfidfVectorizer()
        # x_tfid_train = tfid_vec.fit_transform(self.x_train)
        # x_tfid_test = tfid_vec.transform(self.x_test)
        #
        # ## 模型训练
        # mnb_tfid = SVC()
        # mnb_tfid.fit(x_tfid_train, self.y_train)
        # mnb_tfid_y_predict = mnb_tfid.predict(x_tfid_test)
        # mnb_tfid.score(x_tfid_test, self.y_test)

if __name__ == '__main__':
    svmC = SVM_classify()
    isKFold = input("请选择是否使用K折验证(Y/N)")
    if isKFold in ['Y', 'y']:  # 使用K折验证
        svmC.kFold_readData()
    elif isKFold in ['N', 'n']:  # 不使用K折验证
        svmC.readData()
        svmC.svmClassify()
    else:
        raise Exception("请输入Y或者N:")