#!/usr/bin/env python3
import dgl
import dgl.function as fn
import torch
#import torch.nn as nn
from mxnet.gluon import nn
import torch.nn.functional as F
from dgl import DGLGraph
import sys
import torch.cuda.profiler as profiler
import pyprof
import dgl.data as da
#import dgl.nn.pytorch.conv.ginconv as gin
import mxnet as mx
from mxnet import gluon
from dgl.nn import GINConv as gin

pyprof.init()

with torch.autograd.profiler.emit_nvtx():

    profiler.start()

    gcn_msg = fn.copy_src(src='h', out='m')
    gcn_reduce = fn.sum(msg='m', out='h')


    class Net(nn.Block):
        def __init__(self):
            super(Net, self).__init__()
            #lin1 = torch.nn.Linear(1433, 16)
            lin1 = gluon.nn.Dense(in_units=1433, units=16)
            lin1.initialize(ctx=mx.gpu(0))
            conv1 = gin(lin1, 'sum')
            self.layer1 = conv1
            #lin2 = torch.nn.Linear(16, 7)
            lin2 = gluon.nn.Dense(in_units=1433, units=16)
            lin2.initialize(ctx=mx.gpu(0))
            self.layer2 = gin(lin2, 'sum')
            #self.layer2.initialize(ctx=mx.gpu(0))

        def forward(self, g, features):
            x = F.relu(self.layer1(g, features))
            x = F.log_softmax(self.layer2(g, x))
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
    #features = torch.FloatTensor(data.features)
    #g = DGLGraph(data.graph).to(device)


    #dataset = da.CoraGraphDataset()

    device = torch.device('cuda')

    #model = Net()
    #model = Net().to(device)
    model = Net().initialize(ctx=mx.gpu(0))

    features = mx.nd.array(data.features, ctx=mx.gpu(0))
    #features = torch.FloatTensor(data.features).to(device)
    g = DGLGraph(data.graph).to(device)

    #data = dataset[0].to(device)

    #g = g.to(device)

    out = model(g, features)

    profiler.stop()

    #print(net)
