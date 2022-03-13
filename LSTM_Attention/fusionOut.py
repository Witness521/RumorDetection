'''
 目的是将评论的Out1和文本的Out2进行融合操作
 首先将评论的Out1进行降维操作（池化等操作）
 然后再将文本的Out2和Out1进行拼接的操作
'''
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from LSTM_Attention.models import bert_RCNN

from LSTM_Attention.models import TextRNN_Att
from LSTM_Attention.readModel import readModelLSTMAtt

class FusionOut():
    # init中的方法只有在创建实例对象的时候才被调用
    def __init__(self, dataset_path):
        self.dataset = dataset_path
        print(self.dataset, '---------------------------------', 'fusionout')
        # TextRNN_Att的config
        self.TextRNN_Att_config = TextRNN_Att.Config(self.dataset, embedding='embedding_SougouNews.npz')
        # bert_RCNN的config
        self.bert_RCNN_config = bert_RCNN.Config(self.dataset)

    baseTextPath = 'F:\\graduationDesign\\dataSet\\Chinese_Rumor_Dataset-master\\CED_Dataset\\'
    blogTextPath = baseTextPath + 'original-microblog'  # 待读取文件的文件夹绝对地址
    blogNonRumorReviewPath = baseTextPath + 'non-rumor-repost'
    blogRumorReviewPath = baseTextPath + 'rumor-repost'
    blogTextFiles = os.listdir(blogTextPath)  # 微博文本的名称列表
    random.shuffle(blogTextFiles)  # 将名称列表随机打乱
    blogNonRumorReviewFiles = os.listdir(blogNonRumorReviewPath)  # 微博非谣言评论的名称列表
    blogRumorReviewFiles = os.listdir(blogRumorReviewPath)  # 微博谣言评论的名称列表

    PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


    '''读取json文件'''
    def load_json(self, path, file):
        with open(path + '\\' + file, 'r', encoding='UTF-8') as load_f:
            load_dict = json.load(load_f)
            return load_dict


    '''
    读取一条微博下评论的操作
    将多条评论用池化的方法来进行降维，降维到一行多列的形式
    '''
    def read_review(self, path, fileName):
        data = self.load_json(path, fileName)
        # 存储一条微博的相关评论的Out
        save_review = []
        # 设置一个是否有评论的标志
        flag = False
        # 读取Json数组(评论)
        for i in range(len(data)):
            if len(data[i]['text']) > 20:
                # 标志设置为True
                flag = True
                # 获取out之后变成list
                save_review.append(readModelLSTMAtt(self.TextRNN_Att_config, review=data[i]['text']).detach().numpy().tolist()[0])
        if flag == False:
            save_review.append([0 for i in range(256)])
        # list转成ndarray
        save_review = np.array([save_review], dtype=np.float64)
        # numpy转torch.Tensor
        save_review_tensor = torch.from_numpy(save_review)
        # 最大池化(卷积核为len(save_review[0]) * 70)
        m = nn.MaxPool2d(kernel_size=(len(save_review[0]), 70), stride=1)
        output = m(save_review_tensor)
        return output.detach().numpy().tolist()[0][0]


    '''
        读取一条微博文本
        返回list形式
    '''
    def read_blog_text(self, path, filename):
        data = self.load_json(path, filename)
        text = data['text']
        # 转成list格式
        return self.judgeRumor(text).detach().numpy().tolist()[0]

    # 读取模型，返回save_out
    def judgeRumor(self, text):
        # 读取模型
        model = bert_RCNN.Model(self.bert_RCNN_config)
        model.load_state_dict(torch.load(self.bert_RCNN_config.save_path))
        model.eval()
        # 对文本text进行encode
        encode_text = self.encodeText(text, self.bert_RCNN_config.pad_size)
        # 输入到模型中，并返回model中保存的分类之前的out
        model(encode_text)
        return model.save_out
    def encodeText(self, content, pad_size=32):
        # convert_ids_to_tokens调用此方法
        token = self.bert_RCNN_config.tokenizer.tokenize(content)
        token = [self.CLS] + token

        seq_len = len(token)
        mask = []
        token_ids = self.bert_RCNN_config.tokenizer.convert_tokens_to_ids(token)
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size

        # 将tensor变成二维的数据
         # tokenIds
        tokenIds = []
        tokenIds.append(token_ids)
        tokenIds.append([101,  137, 1062, 2128, 6956, 1036, 4997, 1927, 6679,  928, 2622, 5165,
                         2593, 1355, 2357, 2398, 1378, 2593, 2823, 2111, 2094, 8024, 3724, 6760,
                         2141, 7741, 2207, 2110, 2192,  782, 1423,  752, 9081, 8471, 8544, 8952,
                         9446, 2376, 2564, 2810, 3141, 8024,  791, 1921,  677, 1286,  671,  702,
                          676, 2259, 1914, 2207, 1957, 2111, 1762, 7239, 5323, 5709, 1736, 2207,
                         1277, 7353, 6818, 6158,  782, 2866, 6624,  749, 8024, 2207, 1957, 2111,
                         5543, 6432, 1139, 1961, 4268, 4268, 4638, 2797])
         # seqLen
        seqLen = []
        seqLen.append(seq_len)
        seqLen.append(80)
         # Mask
        Mask = []
        Mask.append(mask)
        Mask.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1])

        # 转成tensor
        token_ids = torch.tensor(tokenIds)
        seqLen = torch.tensor(seqLen)
        mask = torch.tensor(Mask)

        return (token_ids, seqLen, mask)


if __name__ == '__main__':

    # 初始化一个存储文本+评论的list
    save_text_review = []
    # 初始化一个存储谣言结果的list
    rumor_result = []

    dataset = 'dataSet'
    fo = FusionOut(dataset)


    for textFile in tqdm(fo.blogTextFiles):
        print(textFile)
        blogText = fo.read_blog_text(fo.blogTextPath, textFile)
        # 如果是谣言的评论
        if textFile in fo.blogRumorReviewFiles:
            review = fo.read_review(fo.blogRumorReviewPath, textFile)
            text_review = blogText + review
            # 向rumor_result中添加[0,1]
            rumor_result.append([0, 1])
        # 如果是非谣言的评论
        else:
            review = fo.read_review(fo.blogNonRumorReviewPath, textFile)
            text_review = blogText + review
            # 向rumor_result中添加[1,0]
            rumor_result.append([1, 0])
        save_text_review.append(text_review)
    save_text_review = torch.tensor(save_text_review)
    rumor_result = torch.tensor(rumor_result)
    print(save_text_review.size())
    print(rumor_result.size())
    torch.save(save_text_review, 'dataSet/saved_dict/save_text_review.pth')
    torch.save(rumor_result, 'dataSet/saved_dict/rumor_result.pth')
