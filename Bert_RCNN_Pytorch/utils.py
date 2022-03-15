# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import KFold

from Bert_RCNN_Pytorch.models import TextCNN

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

class BuildDataSet:
    def __init__(self, config):
        self.tokenizer = lambda x: [y for y in x]  # char-level
        self.vocab = pkl.load(open(config.vocab_path, 'rb'))
        self.config = config
        # 数据集的位置
        self.dataLocation = 'dataSet/data/Chinese_Rumor_dataset_clean.xls'

    '''
        对数据集进行加工
    '''
    def load_dataset(self, df, pad_size=32):
        contents = []
        for indexs in tqdm(df.index):
            content = df.loc[indexs].values[1]
            if type(content) != str:
                continue
            label = df.loc[indexs].values[3]
            token = self.tokenizer(content)
            seq_len = len(token)
            words_line = []
            # 如果不够pad_size的长度
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0, seq_len), ([...], 1, seq_len), ...]
    '''
        正常读取数据集
    '''
    def build_dataset(self):
        print(f"Vocab size: {len(self.vocab)}")
        # 读取数据集
        df = pd.read_excel(self.dataLocation)
        # 划分数据集
        # 训练集占总体的80%
        train_data = df.sample(frac=0.8, replace=False, random_state=0, axis=0)
        # test是除了train数据集之外的
        test_data = df[~df.index.isin(train_data.index)]
        # valid是从train中取1/8(也就是总数据集的0.1)
        valid_data = train_data.sample(frac=1 / 8, replace=False, random_state=0, axis=0)
        # 从train数据集中去掉valid数据集
        train_data = train_data[~train_data.index.isin(valid_data)]
        # 对数据进行处理
        train = self.load_dataset(train_data, self.config.pad_size)
        valid = self.load_dataset(valid_data, self.config.pad_size)
        test = self.load_dataset(test_data, self.config.pad_size)
        return train, valid, test

    '''k折交叉验证'''
    def build_kFold_dataset(self):
        print(f"Vocab size: {len(self.vocab)}")
        # 读取数据集
        df = pd.read_excel(self.dataLocation)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(df):
            train_data, test_data = df[train_index], df[test_index]



class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    # train_dir = "dataSet/data/train.txt"
    # vocab_dir = "dataSet/data/vocab.pkl"
    # pretrain_dir = "dataSet/data/sgns.sogou.char"
    # emb_dim = 300
    # filename_trimmed_dir = "dataSet/data/embedding_SougouNews"
    # if os.path.exists(vocab_dir):
    #     word_to_id = pkl.load(open(vocab_dir, 'rb'))
    # else:
    #     # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    #     tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    #     word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #     pkl.dump(word_to_id, open(vocab_dir, 'wb'))
    #
    # embeddings = np.random.rand(len(word_to_id), emb_dim)
    # f = open(pretrain_dir, "r", encoding='UTF-8')
    # for i, line in enumerate(f.readlines()):
    #     # if i == 0:  # 若第一行是标题，则跳过
    #     #     continue
    #     lin = line.strip().split(" ")
    #     if lin[0] in word_to_id:
    #         idx = word_to_id[lin[0]]
    #         emb = [float(x) for x in lin[1:301]]
    #         embeddings[idx] = np.asarray(emb, dtype='float32')
    # f.close()
    # np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    config = TextCNN.Config('dataSet')
    data_set = BuildDataSet(config)
    data_set.build_kFold_dataset()
