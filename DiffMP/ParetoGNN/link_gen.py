from dgl.data import CoraGraphDataset, PubmedGraphDataset, CiteseerGraphDataset, WikiCSDataset, CoauthorCSDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorPhysicsDataset
import dgl
import torch
import pickle
from copy import deepcopy
import scipy.sparse as sp
import numpy as np
import os 

def mask_test_edges(adj_orig, val_frac, test_frac):

    # Remove diagonal elements
    adj = deepcopy(adj_orig)
    # set diag as all zero
    adj.setdiag(0)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj, 1)
    edges = sparse_to_tuple(adj_triu)[0]
    num_test = int(np.floor(edges.shape[0] * test_frac))
    num_val = int(np.floor(edges.shape[0] * val_frac))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = edges[all_edge_idx[num_val + num_test:]]

    noedge_mask = np.ones(adj.shape) - adj
    noedges = np.asarray(sp.triu(noedge_mask, 1).nonzero()).T
    all_edge_idx = list(range(noedges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges_false = noedges[test_edge_idx]
    val_edges_false = noedges[val_edge_idx]

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T


    train_mask = np.ones(adj_train.shape)
    for edges_tmp in [val_edges, val_edges_false, test_edges, test_edges_false]:
        for e in edges_tmp:
            assert e[0] < e[1]
        train_mask[edges_tmp.T[0], edges_tmp.T[1]] = 0
        train_mask[edges_tmp.T[1], edges_tmp.T[0]] = 0

    train_edges = np.asarray(sp.triu(adj_train, 1).nonzero()).T
    train_edges_false = np.asarray((sp.triu(train_mask, 1) - sp.triu(adj_train, 1)).nonzero()).T

    # NOTE: all these edge lists only contain single direction of edge!
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


if not os.path.exists('./MP_links'):
    os.mkdir('./MP_links')
if not os.path.exists('./MP_pretrain_labels'):
    os.mkdir('./MP_pretrain_labels')

for root, ds, _ in os.walk('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2'):
    for d in ds:
        if 'Design' in d:
            base_design_name = d
            print('processing:', base_design_name)

            g, _ = dgl.load_graphs(f'MP_hetero_graphs/{base_design_name}.bin')
            g = g[0]
            total_pos_edges = torch.randperm(g.num_edges())
            adj_train = g.adjacency_matrix(scipy_fmt='csr')
            # adj_train = g.adj_external(scipy_fmt='csr')
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train, 0.1, 0.2)
            tvt_edges_file = f'MP_links/{base_design_name}_tvtEdges.pkl'
            pickle.dump((train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), open(tvt_edges_file, 'wb'))
            node_assignment = dgl.metis_partition_assignment(g, 10)
            torch.save(node_assignment, f'MP_pretrain_labels/metis_label_{base_design_name}.pt')

