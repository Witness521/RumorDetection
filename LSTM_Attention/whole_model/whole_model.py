'''
    整体的流程：
    将获取到的微博文本和评论输入到整体模型中，模型返回谣言的概率
'''
import json
import os
import numpy as np
import torch
import torch.nn as nn
from LSTM_Attention.models import TextRNN_Att
from LSTM_Attention.fusionOut import FusionOut
from get_weibo_data import WeiboData
from LSTM_Attention.utils import build_iterator
from LSTM_Attention.MLPUpgrade import MLP
import pickle as pkl



class WholeModel():
    MAX_VOCAB_SIZE = 10000  # 词表长度限制
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

    path = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径
    dataset = path + '/dataSet'
    # 获取config
    TextRNN_Att_config = TextRNN_Att.Config(dataset, embedding='embedding_SougouNews.npz')


    '''
        获取encode之后的微博文本和评论并将其融合
        输入：微博的url地址
    '''
    def fusion_single_out(self, blob, reviewList):
        # 1.实例化FusionOut对象并返回save_out
        print(self.dataset, '---------------------------------','whole model')
        fusionout = FusionOut(self.dataset)
        # 返回list格式的blob_out
        blob_out = fusionout.judgeRumor(blob).detach().numpy().tolist()[0]
        # 2.返回list格式的review_out
        review_out = self.build_review(reviewList)
        # 3.将blob_out和review_out进行拼接
        blob_review_out = blob_out + review_out
        # 4.返回blob_review_out
        return blob_review_out

    '''
        根据评论数据构建数据集
    '''
    def load_dataset(self, reviewList, pad_size=32):
        contents = []
        words_line = []

        tokenizer = lambda x: [y for y in x]  # char-level
        if os.path.exists(self.TextRNN_Att_config.vocab_path):
            vocab = pkl.load(open(self.TextRNN_Att_config.vocab_path, 'rb'))
        else:
            vocab = self.build_vocab(self.TextRNN_Att_config.train_path, tokenizer=tokenizer, max_size=self.MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(vocab, open(self.TextRNN_Att_config.vocab_path, 'wb'))

        # 遍历reviewList构建数据集
        for review in reviewList:
            token = tokenizer(review)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([self.PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(self.UNK)))

            contents.append((words_line, int(1), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    '''
        将review进行加工
        返回：多条评论池化之后的list(size:1*187)
    '''
    def build_review(self, reviewList):
        # 存储一条微博的评论的Out
        save_review = []
        # 构建dataset
        dataset = self.load_dataset(reviewList, self.TextRNN_Att_config.pad_size)
        # 构建迭代器
        iter = build_iterator(dataset, self.TextRNN_Att_config)
        # 读取保存的模型并将其部署在cuda上
        model = TextRNN_Att.Model(config=self.TextRNN_Att_config).to(self.TextRNN_Att_config.device)
        model.load_state_dict(torch.load(self.TextRNN_Att_config.save_path))
        model.eval()

        # 将review输入到模型中
        for i, (review, labels) in enumerate(iter):
            model(review)
            # 转成cpu上，不保存梯度信息
            review_out = model.save_out.cpu().detach().numpy().tolist()[0]
            save_review.append(review_out)

        # list转成ndarray
        save_review = np.array([save_review], dtype=np.float64)
        # numpy转torch.Tensor
        save_review_tensor = torch.from_numpy(save_review)
        # 最大池化(卷积核为len(save_review[0]) * 70)
        m = nn.MaxPool2d(kernel_size=(len(save_review[0]), 70), stride=1)
        output = m(save_review_tensor)
        return output.detach().numpy().tolist()[0][0]


    '''
        将加工过后的文本和评论送到多层感知机(MLP)中进行二分类
        返回该条微博是谣言的概率(0-1)
    '''
    def read_mlp_classification(self, blob_review_out):
        mlp = MLP(self.dataset)
        model = mlp.model
        model.load_state_dict(torch.load('F:\graduationDesign\improve\LSTM_Attention\dataSet\saved_dict\MLP_update.ckpt'))
        model.eval()
        # blob_review_out转tensor
        blob_review_out = torch.tensor(blob_review_out)
        y_pred = model(blob_review_out)
        m = nn.Softmax()
        # 获取谣言的概率，并转为float类型的数
        rumorProb = m(y_pred)[1].detach().numpy().tolist()
        # 保留三位小数
        return round(rumorProb, 3)

'''
    谣言检测的总体模型
    为前端提供服务的后端接口
'''
# 引入flask
from flask import Flask, request
# 实例化并命名为app实例
app = Flask(__name__)

@app.route('/rumorDetect', methods=['GET'])
def rumorDetect():
    # 获取谣言检测的文本
    url = request.args.get("url")

    wm = WholeModel()
    # 1.实例化WeiboData对象返回文本和评论
    wbData = WeiboData()
    blob_review = wbData.get_blob_review_by_url(url=url)
    # 微博文本
    blob = blob_review[0]
    # 评论
    reviewList = blob_review[1]
    # 将微博文本和评论进行融合操作
    blob_review_out = wm.fusion_single_out(blob, reviewList)
    # 使用MLP进行二分类
    rumor_prob = wm.read_mlp_classification(blob_review_out)
    # 封装成对象形式
    returnText = {'errMsg': 'ok', 'rumor_prob': rumor_prob, 'text': blob, 'review': reviewList}
    return json.dumps(returnText)


if __name__ == '__main__':
    # flask的代码
    # 调用run方法，设定端口号，启动服务
    app.run(port=5000, host="172.20.10.2", debug=True)

    '''
    url = 'https://m.weibo.cn/1241148864/4715914630530847'
    wm = WholeModel()
    # 1.实例化WeiboData对象返回文本和评论
    wbData = WeiboData()
    blob_review = wbData.get_blob_review_by_url(url=url)
    # 微博文本
    blob = blob_review[0]
    # 评论
    reviewList = blob_review[1]
    # 将微博文本和评论进行融合操作
    blob_review_out = wm.fusion_single_out(blob, reviewList)
    # 使用MLP进行二分类
    rumor_prob = wm.read_mlp_classification(blob_review_out)
    # 封装成对象形式
    returnText = {'errMsg': 'ok', 'text': blob, 'review': reviewList}
    print(json.dumps(returnText))
    '''