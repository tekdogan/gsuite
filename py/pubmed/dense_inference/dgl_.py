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
            self.layer1 = DenseGraphConv(500, 16)
            self.layer2 = DenseGraphConv(16, 3)

        def forward(self, g, features):
            x = F.relu(self.layer1(g, features))
            x = F.log_softmax(self.layer2(g, x))
            #x = self.layer1(g, features)
            return x

    from dgl.data import citation_graph as citegrh

    device = torch.device('cuda')

    data = citegrh.load_pubmed()
    #features = torch.FloatTensor(data.features)
    #g = DGLGraph(data.graph).to(device)


    #dataset = da.CoraGraphDataset()


    #model = Net()
    #model = Net().to(device)

    features = torch.FloatTensor(data.features).to(device)

    model = Net().to(device)
    g = DGLGraph(data.graph)
    g = dgl.add_self_loop(g)
    g = g.adjacency_matrix(ctx=device)

    print(g.shape)

    #data = dataset[0].to(device)

    g = g.to(device)

    out = model(g, features)

    profiler.stop()

    #print(net)
