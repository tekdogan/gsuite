#!/usr/bin/env python3
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
import torch.cuda.profiler as profiler
from torch_geometric.datasets import Planetoid
import pyprof
import torch

pyprof.init()

dataset = Planetoid(root='/tmp/Cora', name='Cora')

with torch.autograd.profiler.emit_nvtx():

    profiler.start()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_node_features, 16)
            #self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index)
            #x = F.relu(x)
            #x = F.dropout(x, training=self.training)
            #x = self.conv2(x, edge_index)

            #return F.log_softmax(x, dim=1)
            return x

    device = torch.device('cuda')
    model = Net().to(device)

    data = dataset[0].to(device)
    out = model(data)

    profiler.stop()

