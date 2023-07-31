# This file is copied from the official repo of Geom-GCN
import os
import networkx as nx
import numpy as np
import dgl
import scipy.sparse as sp
import torch

if not os.path.exists('./MP_hetero_graphs'):
    os.mkdir('./MP_hetero_graphs')

for root, ds, _ in os.walk('/research/d1/gds/qluo22/dataset_col/mlcad2023_v2'):
    for d in ds:
        if 'Design' in d:
            base_design_name = d
            print('processing:', base_design_name)

            graph_adjacency_list_file_path = os.path.join(root, base_design_name, 'netlist_feature/PU_link.txt')
            graph_node_features_and_labels_file_path = os.path.join(root, base_design_name, 'netlist_feature/PU_feature.txt')

            G = nx.DiGraph()
            graph_node_features_dict = {}
            graph_labels_dict = {}

            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels = graph_node_features_and_labels_file.readlines()
                for line in graph_node_features_and_labels:
                    line = line.strip().split(' ')
                    assert (len(line) == 2)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    label = np.sum(np.array(line[1].split(',')[:3], dtype=np.uint8) * np.array([0,1,2]))
                    graph_labels_dict[int(line[0])] = label

                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                    label=graph_labels_dict[int(line[0])])

            with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
                graph_adjacency_list = graph_adjacency_list_file.readlines()
                for line in graph_adjacency_list:
                    line = line.strip().split(' ')
                    # print('line:', line)
                    # print('graph_node_features_dict[int(line[0])]:', graph_node_features_dict[int(line[0])])
                    # print('graph_labels_dict[int(line[0])]:', graph_labels_dict[int(line[0])])
                    assert (len(line) == 2)
                    if int(line[0]) not in G:
                        G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                    label=graph_labels_dict[int(line[0])])
                    if int(line[1]) not in G:
                        G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                    label=graph_labels_dict[int(line[1])])
                    G.add_edge(int(line[0]), int(line[1]))

            adj = nx.adjacency_matrix(G, sorted(G.nodes()))
            features = np.array(
                [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
            labels = np.array(
                [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

            g = dgl.DGLGraph(adj)
            g = dgl.add_reverse_edges(g)
            g.ndata['feat'] = torch.tensor(features, dtype=torch.float32) 
            g.ndata['label'] = torch.tensor(labels, dtype=torch.int32) 

            dgl.save_graphs('MP_hetero_graphs/'+base_design_name+'.bin', [g])
