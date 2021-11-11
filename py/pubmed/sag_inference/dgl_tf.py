#!/usr/bin/env python3
import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import sys
import torch.cuda.profiler as profiler
import pyprof
import dgl.data as da
from dgl.nn import SAGEConv
import tensorflow as tf
from tensorflow.keras import layers

pyprof.init()

with torch.autograd.profiler.emit_nvtx():

    profiler.start()

    class Net(layers.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.layer1 = SAGEConv(1433, 16, 'gcn')
            self.layer2 = SAGEConv(16, 7, 'gcn')

        def call(self, g, features):
            x = tf.nn.relu(self.layer1(g, features))
            x = tf.nn.log_softmax(self.layer2(g, x))
            return x

    from dgl.data import citation_graph as citegrh
    import networkx as nx
#    def load_cora_data():
#        data = citegrh.load_cora()
#        features = torch.FloatTensor(data.features)
#        labels = torch.LongTensor(data.labels)
#        train_mask = torch.BoolTensor(data.train_mask)
#        test_mask = torch.BoolTensor(data.test_mask)
#        g = DGLGraph(data.graph)
#        return g, features, labels, train_mask, test_mask


    data = citegrh.load_cora()
    #features = torch.FloatTensor(data.features)
    #g = DGLGraph(data.graph).to(device)


    #dataset = da.CoraGraphDataset()

    device = torch.device('cuda')

    with tf.device('/GPU:0'):

        model = Net()
        #model = Net().to(device)

        features = torch.FloatTensor(data.features)
        g = DGLGraph(data.graph)

        #data = dataset[0].to(device)

        out = model(g, features)

    profiler.stop()

        #print(net)
