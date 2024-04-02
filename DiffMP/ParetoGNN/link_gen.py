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
    # remove the zero entries (including the diagonal)
    adj.eliminate_zeros()
    # Check that diag is zero:
    # assert np.diag(adj.todense()).sum() == 0

    # obtain the matrix with elements above the 1st diagnal axis (one directional connection)
    adj_triu = sp.triu(adj, 1)

    # Convert the sparse matrix to the tuple (start_id, end_id)
    edges = sparse_to_tuple(adj_triu)[0]
    
    # testing percentage (0.2)
    num_test = int(np.floor(edges.shape[0] * test_frac))

    # validation percentage (0.1)
    num_val = int(np.floor(edges.shape[0] * val_frac))

    # The shape of the edges (randomly select)
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)

    # val edge idx
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

    # divide into the train/val/test
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = edges[all_edge_idx[num_val + num_test:]]

    # noedge_mask
    # get the mask where there are no edges
    noedge_mask = np.ones(adj.shape) - adj
    # get the part above the 1th diagnal axis indicating noedging between these two nodes
    
    noedges = np.asarray(sp.triu(noedge_mask, 1).nonzero()).T
    
    all_edge_idx = list(range(noedges.shape[0]))
    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

    test_edges_false = noedges[test_edge_idx]
    val_edges_false = noedges[val_edge_idx]

    data = np.ones(train_edges.shape[0])
    # create csr matrix for training
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # bi-direction
    adj_train = adj_train + adj_train.T

    # get the training mask for train_dges and train_edges_false
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

# sparse matrix format to tuple format (coordinates, values, shape)
def sparse_to_tuple(sparse_mx):
    # only store the value indicated by row and col
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# links
if not os.path.exists('./MP_links'):
    os.mkdir('./MP_links')

# true labels
if not os.path.exists('./MP_pretrain_labels'):
    os.mkdir('./MP_pretrain_labels')

for root, ds, _ in os.walk('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2'):
    for d in ds:
        if 'Design' in d:
            base_design_name = d
            print('processing:', base_design_name)

            g, _ = dgl.load_graphs(f'MP_hetero_graphs/{base_design_name}.bin')
            g = g[0]
            
            # select some of the edges of the graph (randomly permutation for the edges)
            total_pos_edges = torch.randperm(g.num_edges())

            # compressed sparse row matrix (row, col)
            adj_train = g.adjacency_matrix(scipy_fmt='csr')
            # adj_train = g.adj_external(scipy_fmt='csr')

            # split the edges in whole graph in train, validation and test (there are the true and false connections)
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj_train, 0.1, 0.2)
            tvt_edges_file = f'MP_links/{base_design_name}_tvtEdges.pkl'
            pickle.dump((train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false), open(tvt_edges_file, 'wb'))
            
            # get the node assignment for the graph
            node_assignment = dgl.metis_partition_assignment(g, 10)
            torch.save(node_assignment, f'MP_pretrain_labels/metis_label_{base_design_name}.pt')

