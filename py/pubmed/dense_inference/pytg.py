#!/usr/bin/env python3
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv
from torch.nn import Sequential, ReLU, Linear
import sys
import torch.cuda.profiler as profiler
from torch_geometric.datasets import Planetoid
import pyprof
import torch
import torch_geometric.utils as tu
from torch_geometric.data import DataLoader

pyprof.init()

dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(len(dataset))

#batch_edge_index = Batch.from_data_list()

print(dataset[0])

with torch.autograd.profiler.emit_nvtx():

    #Start profiler
    profiler.start()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = DenseGCNConv(dataset.num_node_features, 16)
            self.conv2 = DenseGCNConv(16, dataset.num_classes)
            #nn1 = Sequential(Linear(dataset.num_node_features, 32), ReLU(), Linear(32, dataset.num_classes))
            #nn1 = Sequential(Linear(dataset.num_node_features, 32))
            #nn2 = Sequential(Linear(32, dataset.num_classes))
            #self.conv1 = SAGEConv(nn1)
            #self.conv2 = SAGEConv(nn2)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            edge_index = tu.to_dense_adj(edge_index)

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            #x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            #return x
            return F.log_softmax(x, dim=1)

    for batch in loader:
        print(batch)
        device = torch.device('cuda')
        model = Net().to(device)

        batch = batch.to(device)
        #data = dataset[0].to(device)
        out = model(batch)

    profiler.stop()
