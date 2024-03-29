#!/usr/bin/env python3
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch.nn import Sequential, ReLU, Linear
import sys
import torch.cuda.profiler as profiler
from torch_geometric.datasets import Planetoid
import pyprof
import torch


pyprof.init()

dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')

print(dataset[0])

with torch.autograd.profiler.emit_nvtx():

    #Start profiler
    profiler.start()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #self.conv1 = GINConv(dataset.num_node_features, dataset.num_classes)
            #self.conv2 = SAGEConv(16, dataset.num_classes)
            nn1 = Sequential(Linear(dataset.num_node_features, 16), ReLU(), Linear(16, dataset.num_classes))
            #nn2 = Sequential(Linear)
            self.conv1 = GINConv(nn1)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            #x = F.relu(x)
            #x = F.dropout(x, training=self.training)
            #x = self.conv2(x, edge_index)
            #return x
            return F.log_softmax(x, dim=1)

    device = torch.device('cuda')
    model = Net().to(device)

    data = dataset[0].to(device)
    out = model(data)

    profiler.stop()

