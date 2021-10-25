import os
import argparse, json

import ctypes
from ctypes import *
lib = cdll.LoadLibrary('./lib/libGCN_DGL.so')

class GCNdgl:
    def __init__(self, in_dim, out_dim):
        self.inD = in_dim
        self.outD = out_dim

    def clean(self):
        lib.clean()

    def deneme(self):
        lib.deneme()
