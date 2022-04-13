import requests
import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATConv

from torch_geometric.utils import add_self_loops,degree
from torch_geometric.datasets import Planetoid
import ssl
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.gat1=GATConv(in_channels=dataset.num_node_features,out_channels=8,heads=8,dropout=0.6)
        self.gat2=GATConv(64,7,1,dropout=0.6)

    def forward(self,data):
        x,edge_index=data.x, data.edge_index
        x=self.gat1(x,edge_index)
        x=self.gat2(x,edge_index)
        return F.log_softmax(x,dim=1)


ssl._create_default_https_context = ssl._create_unverified_context
dataset = Planetoid(root='Cora', name='Cora')
x=dataset[0].x
edge_index=dataset[0].edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct/int(data.test_mask.sum())
print('Accuracy:{:.4f}'.format(acc))