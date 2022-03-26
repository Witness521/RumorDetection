import ast
import math
import queue
import torch.nn as nn
import torch
import numpy
import pandas as pd
from transformers import BertConfig, AutoTokenizer, BertModel
from tqdm import tqdm


# review_embedding的config
class Config(object):
    def __init__(self):
        self.model_name = 'bert_embedding'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.bert_path = './bert_pretrain'
        self.dataLocation = 'dataSet/data/Chinese_Rumor_dataset_clean.xls'
        # self.dataLocation = 'dataSet/data/reduced_data.xlsx'
        # 词嵌入的长度
        self.pad_size = 30  # 90%位置的长度

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # review的list
        self.total_review_list = []
        # id的list
        self.id_list = []
        self.config = config
        # 队列存储每一条post的[cls]对应的输出(768维)
        self.save_review_cls_list = queue.Queue()
        # 在加载类的时候加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.bert_config = BertConfig.from_pretrained(config.bert_path)
        self.bertModel = BertModel.from_pretrained(config.bert_path, config=self.bert_config)


    '''
        读取、划分数据集, 每份数据集的长度为dataset_len
        并将划分之后的数据集以list的方式返回
    '''
    def read_dataset(self, dataset_len):
        df = pd.read_excel(self.config.dataLocation)
        # 总数据集的长度
        df_len = df.shape[0]
        # 根据每份数据集的长度(dataset_len)计算出一共有多少个数据集df(df_division_num)
        df_division_num = math.floor(df_len / dataset_len)
        # 判断是否能整除
        flag = 0
        if df_len / dataset_len - df_division_num:
            flag = 1
        # 存储划分之后的df_division
        df_list = []
        for index in tqdm(range(df_division_num + flag)):
            if index < df_division_num - 1 + flag:
                df_division = df[dataset_len * index:dataset_len * (index + 1)]
            else:
                df_division = df[dataset_len * index:]
            df_list.append(df_division)
        return df_list


    '''将获取到的df按照一对一的方式分别存入两个列表'''
    def save_review_id_to_list(self, df):
        for indexs in df.index:
            reviews = df.loc[indexs].values[2]
            review_list = ast.literal_eval(reviews)
            id = df.loc[indexs].values[0]
            # 向列表中一对一写入review_list和id(id<=2600就是谣言, id>2600就不是谣言)
            self.total_review_list.append(review_list)
            self.id_list.append(id)

    '''
        对review进行embedding
        存储每个review的[cls]对应的那个768维的tensor到self.save_post_cls_list
    '''
    def review_embedding(self, review_list):
        # tokenizer
        inputs = self.tokenizer(review_list,
                                padding=True,  # 不够长的进行补齐
                                truncation=True,  # 超长的进行截断
                                max_length=self.config.pad_size,
                                return_tensors="pt")  # 返回pytorch tensor
        # 将inputs放置在GPU上
        inputs = inputs.to(self.config.device)
        # 将content输入到模型中，获取embedding后的输出
        outputs = self.bertModel(**inputs)[0]
        # 将embedding之后的(140,768)中的第一个tensor储存
        for review_tensor in outputs:
            # 只存储每个post的[cls]对应的那个768维的tensor(转换成numpy进行存储!!!)
            content_numpy = review_tensor[0].unsqueeze(0).cpu().detach().numpy()  # 将(768)变成(1, 768)
            self.save_review_cls_list.put(content_numpy)
        del outputs


    '''
       执行模型中对post进行embedding的操作，对外提供的接口
       num代表是整个df_list拆分为几个小的列表进行执行
    '''
    def execute(self, dataset_len):
        df_list = self.read_dataset(dataset_len)
        count = 0
        for df in tqdm(df_list):
            self.save_review_id_to_list(df)
            print(len(self.total_review_list))
            for review_list in self.total_review_list:
                # 此处会由于review_list过大导致内存溢出，因此对于长度大于100的review_list进行切割
                if len(review_list) > 100:
                    division_num = math.floor(len(review_list) / 100)
                    # 判断是否整除
                    flag = 0
                    if len(review_list) / 100 - division_num:
                        flag = 1
                    for index in range(division_num + flag):
                        if index == 0:
                            division_review_list = review_list[index * 100:(index + 1) * 100]
                            # 对每一条post的多条评论进行embedding, review_list代表的是一条post的评论list
                            self.review_embedding(division_review_list)
                            # 将queue中的numpy取出，拼成一个大的array
                            review_emb = self.save_review_cls_list.get()
                        elif index < division_num - 1 + flag:
                            division_review_list = review_list[index * 100:(index + 1) * 100]
                            # 对每一条post的多条评论进行embedding, review_list代表的是一条post的评论list
                            self.review_embedding(division_review_list)
                        else:
                            division_review_list = review_list[100 * index:]
                            # 对每一条post的多条评论进行embedding, review_list代表的是一条post的评论list
                            self.review_embedding(division_review_list)

                        # 当队列不为空的时候
                        while not self.save_review_cls_list.empty():
                            review_emb = numpy.concatenate((review_emb, self.save_review_cls_list.get()), axis=0)  # 竖向拼接

                else:
                    self.review_embedding(review_list)
                    # 将queue中的numpy取出，拼成一个大的array
                    review_emb = self.save_review_cls_list.get()
                    # 当队列不为空的时候
                    while not self.save_review_cls_list.empty():
                        review_emb = numpy.concatenate((review_emb, self.save_review_cls_list.get()), axis=0)  # 竖向拼接

                # 最后将post_emb转成tensor进行存储
                print(review_emb.shape)
                review_emb = torch.from_numpy(review_emb)
                # 按照excel中post的顺序，从0开始编码存储
                torch.save(review_emb, './dataSet/saved_tensor/review/' + str(count) + '.pt')
                count += 1
            # 将review_list和id_list进行清空
            self.total_review_list.clear()
            self.id_list.clear()



# 计算出所有的review按照长度升序排列，90%位置的长度
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
    print(len(review_len))  # review的总条数 606418
    print(review_len[math.floor(len(review_len) * 0.9)])  # 90%位置的长度为30



if __name__ == '__main__':
    config = Config()
    Model(config).to(config.device).execute(20)