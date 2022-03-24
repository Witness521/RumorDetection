import ast
import math

import torch
import pandas as pd
from transformers import BertConfig, AutoTokenizer, BertModel
from utils import BuildDataSet, build_iterator
from tqdm import tqdm


# review_embedding的config
class Config(object):
    def __init__(self):
        self.model_name = 'bert_embedding'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.bert_path = './bert_pretrain'
        self.dataLocation = 'dataSet/data/Chinese_Rumor_dataset_clean.xls'
        # 词嵌入的长度
        self.pad_size = 140

class Model():
    def __init__(self, config):
        self.config = config
        # 在加载类的时候加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)
        self.bertModel = BertModel.from_pretrained(config.bert_path, config=self.bert_config)


    '''读取、划分num份数据集 并将划分之后的数据集以list的方式返回'''
    def read_dataset(self, num):
        df = pd.read_excel(self.config.dataLocation)
        df_len = df.shape[0]
        # 划分之后每一个df的长度
        each_epoch_len = math.floor(df_len / num)
        # 存储划分之后的df_division
        df_list = []
        for index in tqdm(range(num)):
            if index < num - 1:
                df_division = df[each_epoch_len * index:each_epoch_len * (index + 1)]
            else:
                df_division = df[each_epoch_len * index:]
            df_list.append(df_division)
        return df_list

    # def try1(self):

def count_utils():
    df = pd.read_excel('dataSet/data/Chinese_Rumor_dataset_clean.xls')
    review_len = []
    for indexs in tqdm(df.index):
        reviews = df.loc[indexs].values[2]
        review_list = ast.literal_eval(reviews)
        # 向review_len中写入每一条评论的长度
        for review in review_list:
            review_len.append(len(review))
    review_len.sort()
    print(len(review_len))



if __name__ == '__main__':
    df = pd.read_excel('dataSet/data/Chinese_Rumor_dataset_clean.xls')
    reviews = df.loc[0].values[2]
    review_list = ast.literal_eval(reviews)
    print(type(review_list))
    print(review_list[0])