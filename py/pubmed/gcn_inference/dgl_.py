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

pyprof.init()

with torch.autograd.profiler.emit_nvtx():

    profiler.start()

    gcn_msg = fn.copy_src(src='h', out='m')
    gcn_reduce = fn.sum(msg='m', out='h')

    class GCNLayer(nn.Module):
        def __init__(self, in_feats, out_feats):
            super(GCNLayer, self).__init__()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, g, feature):
            # Creating a local scope so that all the stored ndata and edata
            # (such as the `'h'` ndata below) are automatically popped out
            # when the scope exits.
            with g.local_scope():
                g.ndata['h'] = feature
                g.update_all(gcn_msg, gcn_reduce)
                h = g.ndata['h']
                return self.linear(h)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = GCNLayer(500, 16)
            self.layer2 = GCNLayer(16, 3)

        def forward(self, g, features):
            x = F.relu(self.layer1(g, features))
            x = F.log_softmax(self.layer2(g, x))
            return x

    from dgl.data import citation_graph as citegrh
    import networkx as nx
    def load_pubmed_data():
        data = citegrh.load_pubmed()
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        train_mask = torch.BoolTensor(data.train_mask)
        test_mask = torch.BoolTensor(data.test_mask)
        g = DGLGraph(data.graph)
        return g, features, labels, train_mask, test_mask


    data = citegrh.load_pubmed()
    #features = torch.FloatTensor(data.features)
    #g = DGLGraph(data.graph).to(device)


    #dataset = da.CoraGraphDataset()

    device = torch.device('cuda')

    #model = Net()
    model = Net().to(device)

    features = torch.FloatTensor(data.features).to(device)
    g = DGLGraph(data.graph).to(device)

    #data = dataset[0].to(device)

    g = g.to(device)

    out = model(g, features)

    profiler.stop()

    #print(net)
