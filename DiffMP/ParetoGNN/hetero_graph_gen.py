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

            # The link between the placement units file
            graph_adjacency_list_file_path = os.path.join(root, base_design_name, 'netlist_feature/PU_link.txt')
           
            # The link between the placement units file            
            graph_node_features_and_labels_file_path = os.path.join(root, base_design_name, 'netlist_feature/PU_feature.txt')

            # Why directed graph? (Graph)
            G = nx.DiGraph()
            graph_node_features_dict = {}
            graph_labels_dict = {}

            # node classification
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels = graph_node_features_and_labels_file.readlines()
                for line in graph_node_features_and_labels:
                    line = line.strip().split(' ')
                    assert (len(line) == 2)
                    # there are no repeated lines
                    assert (int(float(line[0])) not in graph_node_features_dict and int(float(line[0])) not in graph_labels_dict)
                    # include the category feature (BRAM/DSP)
                    graph_node_features_dict[int(float(line[0]))] = np.array(line[1].split(','), dtype=float).astype(np.uint8)
                    # classify whether it is the BRAM or the DSP (it is hard to tell according to this)
                    label = np.sum(np.array(line[1].split(',')[:2], dtype=float).astype(np.uint8) * np.array([0,1]))
                    graph_labels_dict[int(float(line[0]))] = label
                    # features (including the node category)
                    G.add_node(int(float(line[0])), features=graph_node_features_dict[int(float(line[0]))],
                                    label=graph_labels_dict[int(float(line[0]))])

            # net classification
            with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
                # read the nets
                graph_adjacency_list = graph_adjacency_list_file.readlines()
                for line in graph_adjacency_list:
                    line = line.strip().split(' ')
                    # print(float('line:', line)
                    # print(float('graph_node_features_dict[int(float(line[0])]:', graph_node_features_dict[int(float(line[0])])
                    # print(float('graph_labels_dict[int(float(line[0])]:', graph_labels_dict[int(float(line[0])])
                    assert (len(line) == 2)
                    # if not exists?
                    if int(float(line[0])) not in G:
                        G.add_node(int(float(line[0])), features=graph_node_features_dict[int(float(line[0]))],
                                    label=graph_labels_dict[int(float(line[0]))])
                    if int(float(line[1])) not in G:
                        G.add_node(int(float(line[1])), features=graph_node_features_dict[int(float(line[1]))],
                                    label=graph_labels_dict[int(float(line[1]))])
                    G.add_edge(int(float(line[0])), int(float(line[1])))

            # adjacency matrix and features, labels are not corresponded (G.nodes() return 2-d tuple (nodeid, dict_val))
            adj = nx.adjacency_matrix(G, sorted(G.nodes()))
            features = np.array(
                [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
            labels = np.array(
                [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

            # Why networkx first then dgl?
            g = dgl.DGLGraph(adj)
            # directional graph -> undirectional graph
            g = dgl.add_reverse_edges(g)

            # features and labels
            g.ndata['feat'] = torch.tensor(features, dtype=torch.float32) 
            g.ndata['label'] = torch.tensor(labels, dtype=torch.int32)

            # save the graphs
            dgl.save_graphs('MP_hetero_graphs/'+base_design_name+'.bin', [g])
