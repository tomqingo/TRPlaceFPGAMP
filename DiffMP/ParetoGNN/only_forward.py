import sys
sys.path.append('./ParetoGNN')

from src.options import Options
import os
import torch
import dgl
from pathlib import Path
import numpy as np
import src.utils
from src.model import PretrainModule, BigModel, GCN
import json
import src.data
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import src.min_norm_solvers
from src.data_train import Graph_Dataset, Universal_Collator


def get_node_emb(checkpoint_path, design_name='MacroPlacement'):

    print("Loading Netlist Data...")
    pretrain_label_dir = './pretrain_labels'
    mask_edge = False
    tvt_addr = None
    split = 'random'
    hetero_graph_path = './hetero_graphs'
    g = src.data.load_data(design_name, 
        pretrain_label_dir = pretrain_label_dir, 
        mask_edge = mask_edge, 
        tvt_addr = tvt_addr, 
        split = split, 
        hetero_graph_path = hetero_graph_path)
    
    hid_dim = [512, 256]
    dropout = 0.
    norm = 'batch'
    use_prelu = True
    node_module = GCN(g.ndata['feat'].shape[1], hid_dim, dropout, norm, use_prelu)

    inter_dim = 0
    bigM = BigModel(node_module, None, inter_dim)

    predictor_dim = 512 ### No use
    device = 'cuda:0'
    ParetoGNN = PretrainModule(bigM, predictor_dim).to(device)

    model = ParetoGNN

    checkpoint_path = checkpoint_path
    print('Loading parameter from: ', checkpoint_path)
    print("Start Farward Pass...")
    node_emb = forward(g, model, checkpoint_path, device)

    return node_emb

    
def forward(g, model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        use_g = g
        X = model.compute_representation(use_g.to(device), g.ndata['feat'].to(device))

        print(X)
        print(X.shape)

        return X

if __name__ == '__main__':
    checkpoint_path = './scripts/chameleon/experiment_name/checkpoint/step-10000_ssnc/model.pth.tar'
    exp_name = 'chameleon'
    get_node_emb(checkpoint_path, exp_name)

