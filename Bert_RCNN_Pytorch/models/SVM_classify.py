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

    def readData(self):
        # 读取数据集
        df = pd.read_excel('../dataSet/data/Chinese_Rumor_dataset_clean.xls')
        # 划分数据集
        # 训练集占总体的80%
        train_data = df.sample(frac=0.8, replace=False, random_state=0, axis=0)
        # 测试集占20%
        test_data = df[~df.index.isin(train_data.index)]

        self.x_train = train_data['post']
        self.x_test = test_data['post']

        self.y_train = train_data['label']
        self.y_test = test_data['label']

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
        print('test acc:', end=' ')
        print(svc.score(x_count_test, self.y_test))
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
    svmC.readData()
    svmC.svmClassify()