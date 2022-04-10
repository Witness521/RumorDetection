import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,GATConv,GlobalAttention
from torch_geometric.nn import global_add_pool,global_mean_pool,global_max_pool,global_sort_pool
from torch_scatter import scatter_mean
from torch_geometric.utils import to_networkx
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import os
import random
import torch.nn  as nn
from tqdm import tqdm_notebook, tqdm
from torch import optim
import torch.functional as F
import argparse
from model import Post, BertPairs_mean, BertPairs_GAT, BertPairs_GATx,MultiModal,Coattention

# set args
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Post',help='choose from [Post, BertPairs_mean, BertPairs_GAT, BertPairs_GATx,MultiModal,Coattention]')
parser.add_argument('--lr',type=float,default=0.001,help='set learning rate')
parser.add_argument('--cuda_id',type=int,default=0,help='choose gpu')
parser.add_argument('--random_seed',type=int,default=7,help='set random num')
parser.add_argument('--patience',type=int,default=4,help='num of no_imporve_rpoch to stop')
parser.add_argument('--data_path',type=str,default='/home/xudandan/competition/BEN/FakeNewsDetection/pheme/pheme_data.jsonl',help='data path')
parser.add_argument('--batch_size',type=int,default=64,help='set batchsize')
parser.add_argument('--dropout',type=float,default=0.3,help='set dropout rate')
parser.add_argument('--use_gpu',type=str,default='True',help='whether use gpu or not')

args = parser.parse_args()

best_f1 = 0
for iters in range(200):
    # set seed
    seed = args.random_seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    if args.use_gpu == 'True':
        device = torch.device('cuda:{}'.format(args.cuda_id) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # load img resnet feature
    path = '/home/xudandan/competition/BEN/FakeNewsDetection/fakeddit/'
    train_resnet_feature = np.load(path+'train_img_features.npy',allow_pickle=True)
    test_resnet_feature = np.load(path+'test_img_features.npy',allow_pickle=True)
    val_resnet_feature = np.load(path+'val_img_features.npy',allow_pickle=True)
    train_resnet_feature,test_resnet_feature,val_resnet_feature = np.array(train_resnet_feature),np.array(test_resnet_feature),np.array(val_resnet_feature)

    # load post and comments' bert features
    pth = '/home/xudandan/competition/BEN/FakeNewsDetection/fakeddit/GAT_DATA/'
    train_news = np.load(pth+'train_news_bert_feature.npy')
    train_pairs = np.load(pth+'train_pairs_bert_feature.npy')
    train_label = np.load(pth+'train_label.npy')

    test_news = np.load(pth+'test_news_bert_feature.npy')
    test_pairs = np.load(pth+'test_pairs_bert_feature.npy')
    test_label = np.load(pth+'test_label.npy')

    val_news = np.load(pth+'val_news_bert_feature.npy')
    val_pairs = np.load(pth+'val_pairs_bert_feature.npy')
    val_label = np.load(pth+'val_label.npy')

    # build graph
    #####################################################################################################
    train_data_list = []
    train_src_id = []
    train_target_id = []
    for i in range(6):
        train_src_id.extend([i]*6)
        train_target_id.extend(list(range(6)))

    for i in range(len(train_label)):
        node_features = train_pairs[i]
        x = torch.tensor(node_features,dtype=torch.float)
        edge_index = torch.tensor([train_src_id, train_target_id], dtype=torch.long)
        y = torch.tensor([train_label[i]],dtype=torch.long)
        single_news  = torch.tensor(train_news[i],dtype=torch.float)
        img = torch.tensor(train_resnet_feature[i:i+1,:,:],dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y, news=single_news,img=img) 
        train_data_list.append(data) 
    #####################################################################################################
    test_data_list = []
    test_src_id = []
    test_target_id = []
    for i in range(6):
        test_src_id.extend([i]*6)
        test_target_id.extend(list(range(6)))

    for i in range(len(test_label)):
        node_features = test_pairs[i]
        x = torch.tensor(node_features,dtype=torch.float)
        edge_index = torch.tensor([test_src_id, test_target_id], dtype=torch.long)
        y = torch.tensor([test_label[i]],dtype=torch.long)
        single_news = torch.tensor(test_news[i],dtype=torch.float)
        img = torch.tensor(test_resnet_feature[i:i+1,:,:],dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y, news=single_news,img=img) 
        test_data_list.append(data) 
    ###########################################################################################################
    val_data_list = []
    val_src_id = []
    val_target_id = []
    for i in range(6):
        val_src_id.extend([i]*6)
        val_target_id.extend(list(range(6)))

    for i in range(len(val_label)):
        node_features = val_pairs[i]
        x = torch.tensor(node_features,dtype=torch.float)
        edge_index = torch.tensor([val_src_id, val_target_id], dtype=torch.long)
        y = torch.tensor([val_label[i]],dtype=torch.long)
        single_news  = torch.tensor(val_news[i],dtype=torch.float)
        img = torch.tensor(val_resnet_feature[i:i+1,:,:],dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y, news=single_news,img=img) 
        val_data_list.append(data)


    # visualization
    # import matplotlib.pyplot as plt
    # import networkx as nx
    # graph = to_networkx(val_data_list[3])
    # nx.draw(graph,with_labels=True)


    # build dataloader
    BATCHSIZE = args.batch_size
    train_dataloader = DataLoader(train_data_list,batch_size=BATCHSIZE,shuffle=True)
    test_dataloader = DataLoader(test_data_list,batch_size=BATCHSIZE,shuffle=False)
    val_dataloader  = DataLoader(val_data_list,batch_size=BATCHSIZE,shuffle=False)

    # define eval functiion
    def eval(model,dataloader):
        model.eval()
        labels,preds = [], [] 
        with torch.no_grad():
            for data in dataloader:
                data.to(device)        
                pred = model(data)
                pred = torch.argmax(pred,dim=1)
                labels.extend(data.y.cpu().numpy())
                preds.extend(pred.cpu().numpy())
        f1 = metrics.f1_score(labels, preds,average='macro')
        return f1

    # init model
    model_name = args.model_name
    if model_name=='MultiModal': 
        model = MultiModal()
    if model_name=='BertPairs_GATx':
        model = BertPairs_GATx()
    if model_name=='Post':
        model = Post()
    if model_name == 'BertPairs_GAT':
        model = BertPairs_GAT()
    if model_name== 'BertPairs_mean':
        model = BertPairs_mean()   
    if model_name == 'Coattention':
        model = Coattention() 


    # loss function ,lr, optimizer
    LR = args.lr
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.to(device)

    # trian

    EPOCH = 200
    running_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    best_metric = 0
    for epoch in range(EPOCH):
        losses = []
        for i,data in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            data.to(device)
            preds = model(data)

            loss = loss_function(preds, data.y)
            losses.append(loss.item())
            running_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % 20 == 19:
                train_loss.append(np.mean(running_loss))
    #             print(preds)
                running_loss = []

#         print('[{}/{}], loss:{}'.format(epoch,EPOCH,np.mean(losses)))
        tuning_metric = eval(model,val_dataloader)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
#             print(best_metric)
        else:
            n_no_improve+=1
        if n_no_improve>= args.patience:
            break  


    # test
    model.eval()
    labels,preds = [],[]
    with torch.no_grad():
        for data in test_dataloader:
            data.to(device)        
            pred = model(data)
            pred = torch.argmax(pred,dim=1)
            labels.extend(data.y.cpu().numpy())
            preds.extend(pred.cpu().numpy())


    ## 计算指标
    print('****************index {}*******************'.format(iters))
    score = metrics.accuracy_score(labels, preds)
    f1_macro = metrics.f1_score(labels,preds,average='macro')
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_index = iters
        print('******************result on test_data of model {} on device {}**********************************'.format(model_name,device))
        print(metrics.classification_report(labels, preds,digits=3))
print('best_f1:{} in index {}'.format(best_f1,best_index))



# # val
# model.eval()
# labels,preds = [],[]
# with torch.no_grad():
#     for data in val_dataloader:
#         data.to(device)        
#         pred = model(data)
#         pred = torch.argmax(pred,dim=1)
#         labels.extend(data.y.cpu().numpy())
#         preds.extend(pred.cpu().numpy())


# ## 计算指标
# score = metrics.accuracy_score(labels, preds)
# f1_macro = metrics.f1_score(labels,preds,average='macro')
# auc = metrics.roc_auc_score(labels,preds)
# print('******************result on val_data**********************************')
# print(metrics.classification_report(labels, preds,digits=3))
# print('f1_macro',f1_macro)
# print('auc',auc)
# print('accuracy',score)

# # train
# model.eval()
# labels,preds = [],[]
# with torch.no_grad():
#     for data in train_dataloader:
#         data.to(device)        
#         pred = model(data)
#         pred = torch.argmax(pred,dim=1)
#         labels.extend(data.y.cpu().numpy())
#         preds.extend(pred.cpu().numpy())


# ## 计算指标
# score = metrics.accuracy_score(labels, preds)
# f1_macro = metrics.f1_score(labels,preds,average='macro')
# auc = metrics.roc_auc_score(labels,preds)
# print(metrics.classification_report(labels, preds,digits=3))
# print('******************result on train_data**********************************')
# print('f1_macro',f1_macro)
# print('auc',auc)
# print('accuracy',score)




