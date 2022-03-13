import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self):
#         self.lstm = nn.LSTM(input_size=config.hidden_size,
#                             hidden_size=config.rnn_hidden,
#                             num_layers=config.num_layers,
#                             bidirectional=True,
#                             batch_first=True,
#                             dropout=config.dropout)
#
#     def forward(self, x):
#         pass

import pandas as pd

df = pd.read_excel('../dataSet/data/Chinese_Rumor_dataset_clean.xls')
# 训练集占总体的80%
train_data = df.sample(frac=0.8, replace=False, random_state=0, axis=0)
# test是除了train数据集之外的
test_data = df[~df.index.isin(train_data.index)]
# valid是从train中取1/8
valid_data = train_data.sample(frac=1/8, replace=False, random_state=0, axis=0)
print(valid_data)
# for indexs in df.index:
#     # .values[num] num代表第几列
#     print(df.loc[indexs].values[1] + '    ' + str(df.loc[indexs].values[3]))
    # print(df.loc[indexs].values[0:-1])