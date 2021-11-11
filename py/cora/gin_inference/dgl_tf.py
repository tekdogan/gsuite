#!/usr/bin/env python3
import dgl
import dgl.function as fn
from dgl import DGLGraph
import sys
import pyprof
import dgl.data as da
from dgl.nn.tensorflow import GINConv
import tensorflow as tf
from tensorflow.keras import layers
import numpy

pyprof.init()

class Net(layers.Layer):
    def __init__(self):
        super(Net, self).__init__()
        lin1 = tf.keras.layers.Dense(16)
        self.layer1 = GINConv(lin1, 'max')
        lin2 = tf.keras.layers.Dense(7)
        self.layer2 = GINConv(lin2, 'max')

    def call(self, g, features):
        x = tf.nn.relu(self.layer1(g, features))
        x = tf.nn.log_softmax(self.layer2(g, x))
        return x

from dgl.data import citation_graph as citegrh
import networkx as nx

#data = citegrh.load_cora()

#with tf.device('/GPU:0'):
with tf.device('/device:CPU:0'):

    #model = Net()

    #features = tf.convert_to_tensor(torch.FloatTensor(data.features).numpy(), numpy.float32)
    #feaures = torch.FloatTensor(data.features)
    #with torch.no_grad():

    with tf.profiler.experimental.Profile('logdir'):
    #tf.profiler.experimental.start('logdir'):
        data = citegrh.load_cora()

        model = Net()

        features = tf.convert_to_tensor(data.features, numpy.float32)
        g = DGLGraph(data.graph)

        out = model(g, features)
    pass
    #tf.profiler.experimental.stop()

