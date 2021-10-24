
import importlib
importlib.import_module('gcn_dgl_wrapper')
importlib.import_module('gcn_spmm_cu')

import os
import argparse, json

def main():

    # add command line arguments to parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu_id')
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    args = parser.parse_args()

    if(args.config is not None):
        with open(args.config) as f:
            config = json.load(f)
    else:
        print('Please call function with parameter --config file.json')
        exit()

    print('Current configurations from config file:')

    ### GPU Configurations ###

    # if gpu id is passed as command line argument
    if(args.gpu_id is not None):
        conf_gpu_usage = True
        conf_gpu_id = int(args.gpu_id)
    # otherwise retrieve it from config file
    else:
        conf_gpu_usage = config['gpu']['use']
        conf_gpu_id = config['gpu']['id']

    print('GPU usage: ' + str(conf_gpu_usage))

    print('GPU ID: ' + str(conf_gpu_id))


    ### GNN Model Configurations ###

    # if gnn model is passed as command line argument
    if(args.model is not None):
        conf_model = str(args.model)
    # otherwise retrieve it from config file
    else:
        conf_model = config['model']

    print('GNN Model: ' + conf_model)


    ### Dataset Configurations ###

    # if dataset is passed as command line argument
    if(args.dataset is not None):
        conf_dataset = str(args.dataset)
    # otherwise retrieve it from config file
    else:
        conf_dataset = config['dataset']


    print('Dataset: ' + conf_dataset)
    GCN_Class = getattr(importlib.import_module('gcn_dgl_wrapper'), 'GCNdgl')
    gcn = GCN_Class(16,4)
    print("Number of input dimensions of GCN model is: " + str(gcn.inD))
    print("Number of putput dimensions of GCN model is: " + str(gcn.outD))
    gcn.deneme()
    #lib.clean()
    #lib.deneme()




main()
