import math
import numpy
import torch
import queue
import torch.nn as nn
import pandas as pd
from transformers import BertConfig, AutoTokenizer, BertModel
from tqdm import tqdm


# post_embedding的config
class Config(object):
    def __init__(self):
        self.model_name = 'bert_embedding'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.bert_path = './bert_pretrain'
        self.dataLocation = 'dataSet/data/Chinese_Rumor_dataset_clean.xls'
        # self.dataLocation = 'dataSet/data/reduced_data.xlsx'
        # 词嵌入的长度
        self.pad_size = 140


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # content的list
        self.content_list = []
        # id的list(验证读取post的顺序)
        self.id_list = []
        self.config = config
        # 队列存储每一条post的[cls]对应的输出(768维)
        self.save_post_cls_list = queue.Queue()
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
    def save_content_id_to_list(self, df):
        for indexs in df.index:
            content = df.loc[indexs].values[1]
            if type(content) != str:
                continue
            id = df.loc[indexs].values[0]
            # 向列表中一对一写入content和id(id<=2600就是谣言, id>2600就不是谣言)
            self.content_list.append(content)
            self.id_list.append(id)

    '''
        对post进行embedding
        存储每个post的[cls]对应的那个768维的tensor到self.save_post_cls_list
    '''
    def post_embedding(self):
        # tokenizer
        inputs = self.tokenizer(self.content_list,
                               padding=True,  # 不够长的进行补齐
                               truncation=True,  # 超长的进行截断
                               max_length=self.config.pad_size,
                               return_tensors="pt")  # 返回pytorch tensor
        # 将inputs放置在GPU上
        inputs = inputs.to(self.config.device)
        # 将content输入到模型中，获取embedding后的输出
        outputs = self.bertModel(**inputs)[0]
        # 将embedding之后的(140,768)中的第一个tensor储存
        for content_tensor in outputs:
            # 只存储每个post的[cls]对应的那个768维的tensor(转换成numpy进行存储!!!)
            content_numpy = content_tensor[0].unsqueeze(0).cpu().detach().numpy()  # 将(768)变成(1, 768)
            self.save_post_cls_list.put(content_numpy)
        del outputs
        # 将content_list和id_list进行清空
        self.content_list.clear()
        # self.id_list.clear()


    '''
        执行模型中对post进行embedding的操作，对外提供的接口
        num代表是整个df_list拆分为几个小的列表进行执行
    '''
    def execute(self, dataset_len):
        df_list = self.read_dataset(dataset_len)
        for df in tqdm(df_list):
            self.save_content_id_to_list(df)
            self.post_embedding()
        # 将queue中的numpy取出，拼成一个大的array
        post_emb = self.save_post_cls_list.get()
        # 当队列不为空的时候
        while not self.save_post_cls_list.empty():
            post_emb = numpy.concatenate((post_emb, self.save_post_cls_list.get()), axis=0)  # 竖向拼接
        print(post_emb.shape)

        # 最后将post_emb转成tensor进行存储
        post_emb = torch.from_numpy(post_emb)
        torch.save(post_emb, './dataSet/saved_tensor/post_embedding.pt')
        # 存储self.id_list验证post的顺序
        numpy.save('dataSet/saved_tensor/id_sequence.npy', numpy.array(self.id_list))


if __name__ == '__main__':
    config = Config()
    Model(config).to(config.device).execute(20)