#!/usr/bin/env python3
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import sys
import torch.cuda.profiler as profiler
import pyprof
import dgl.data as da
from dgl.nn import DenseGraphConv

pyprof.init()

with torch.autograd.profiler.emit_nvtx():

    profiler.start()

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = DenseGraphConv(1433, 16)
            self.layer2 = DenseGraphConv(16, 7)

        def forward(self, g, features):
            #x = F.relu(self.layer1(g, features))
            #x = F.log_softmax(self.layer2(g, x))
            x = self.layer1(g, features)
            return x

    from dgl.data import citation_graph as citegrh
    import networkx as nx
    def load_cora_data():
        data = citegrh.load_cora()
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        train_mask = torch.BoolTensor(data.train_mask)
        test_mask = torch.BoolTensor(data.test_mask)
        g = DGLGraph(data.graph)
        return g, features, labels, train_mask, test_mask


    data = citegrh.load_cora()
    #data = load_cora_data()
    #features = torch.FloatTensor(data.features)
    #g = DGLGraph(data.graph).to(device)


    #dataset = da.CoraGraphDataset()

    device = torch.device('cuda')

    #model = Net()
    model = Net().to(device)

    print(data.labels.shape)
    print(data.features.shape)

    #s = torch.sparse_coo_tensor(data.labels, data.features, [16,1433])
    #(s.to_dense()).to(device)
    #features = (data.features).to(device).to_dense()
    g = DGLGraph(data.graph)
    g = dgl.add_self_loop(g)
    g = g.adjacency_matrix(ctx=device)

    #data = dataset[0].to(device)
    print(g.shape)
    g = g.to(device)

    out = model(g, data.features)

    profiler.stop()

    #print(net)
