import os
import argparse, json

import ctypes
from ctypes import *
lib = cdll.LoadLibrary('./lib/libCU_SpMM_GCN.so')

def gcnlayer():
    print('libCU_SpMM_GCN loaded successfully! This is func gcnlayer()')
