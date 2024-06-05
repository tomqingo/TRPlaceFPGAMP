import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import time
from tqdm import tqdm
import random
import pdb

# three layer CNN
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )
    def forward(self, x):
        return self.cnn(x)

class MyCNNCoarse(nn.Module):
    def __init__(self, res_net, device):
        super(MyCNNCoarse, self).__init__()
        self.cnn = res_net.to(device)
        self.cnn.fc = torch.nn.Linear(512, 16*33*300)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        # pdb.set_trace()
        x = x.reshape(-1, 16, 33, 300)
        return self.conv2(x)

class Actor(nn.Module):
    def __init__(self, cnn, gcn, cnn_coarse, grid_width, grid_height):
        super(Actor, self).__init__()
        # fc1, fc2 and fc3 are three linear layers
        num_emb_state = 64 + 2 + 1
        num_state = 1 + grid_width * grid_height * 5 + 2
        self.fc1 = nn.Linear(num_emb_state, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, grid_width * grid_height)
        # convolution neural network
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        # graph convolution network
        self.gcn = None
        self.softmax = nn.Softmax(dim=-1)
        self.merge = nn.Conv2d(2, 1, 1)
        # (grid_width, grid_height)
        self.grid_width = grid_width
        self.grid_height = grid_height

    def forward(self, x, soft_coefficient, column_ids, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        # cnn resource
        if not cnn_res:
            # (batch, 1+grid*grid*1) (net_img, mask, net_img_2, mask_2)
            cnn_input = x[:, 1+self.grid_width*self.grid_height*1: 1+self.grid_width*self.grid_height*5].reshape(-1, 4, self.grid_width, self.grid_height)
            # mask of the canvas (It is easily to be incorporated with the coarse_input)
            mask = x[:, 1+self.grid_width*self.grid_height*2: 1+self.grid_width*self.grid_height*3].reshape(-1, self.grid_width, self.grid_height)

            # modify the mask according to column ids
            for id in range(self.grid_width):
                if id not in column_ids:
                    mask[:, id, :] = 1
            
            # mask distance
            mask = mask.flatten(start_dim=1, end_dim=2)
            cnn_res = self.cnn(cnn_input)

            # Why should these two information be processed sperately?
            # How to calculate the two masks seperately?
            coarse_input = torch.cat((x[:, 1: 1+self.grid_width*self.grid_height*2].reshape(-1, 2, self.grid_width, self.grid_height),
                                        x[:, 1+self.grid_width*self.grid_height*3: 1+self.grid_width*self.grid_height*4].reshape(-1, 1, self.grid_width, self.grid_height)
                                        ),dim= 1).reshape(-1, 3, self.grid_width, self.grid_height)
            cnn_coarse_res = self.cnn_coarse(coarse_input)
            # pdb.set_trace()
            # pdb.set_trace()
            cnn_res = self.merge(torch.cat((cnn_res, cnn_coarse_res), dim=1))
        # net img could be used as an individual indicator
        net_img = x[:, 1+self.grid_width*self.grid_height: 1+self.grid_width*self.grid_height*2]
        # If the mask is 1, we could not place them at their places
        net_img = net_img + x[:, 1+self.grid_width*self.grid_height*2: 1+self.grid_width*self.grid_height*3] * 10

        # whether to constrain the actions on the wire mask (smaller wirelengt increase)
        net_img_min = net_img.min() + soft_coefficient
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = cnn_res
        x = x.reshape(-1, self.grid_width * self.grid_height)
        # Which is the threshold
        # the place in the mask and mask2 could not accomodate the macros
        x = torch.where(mask + mask2 >=1.0, -1.0e10, x.double())
        # the probability distribution of x
        # Where is the congestion satisfaction?
        x = self.softmax(x)
        # pdb.set_trace()

        return x, cnn_res, gcn_res

# One dimension
class ActorCol(nn.Module):
    def __init__(self, grid_col):
        # (33,1)
        super(ActorCol, self).__init__()
        self.grid_col = grid_col
        self.hidden_emb = 100

        # utilization head
        self.utilization_extractor = nn.Sequential(
            nn.Linear(self.grid_col, int(self.hidden_emb / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_emb / 2), int(self.hidden_emb / 2)),
            nn.ReLU()
        )

        # coarse head
        self.coarse_extractor = nn.Sequential(
            nn.Linear(self.grid_col*3, int(self.hidden_emb / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_emb / 2), int(self.hidden_emb / 2)),
            nn.ReLU()
        )

        # merge head
        self.merge_extractor = nn.Sequential(
            nn.Linear(self.hidden_emb, self.hidden_emb),
            nn.ReLU(),
            nn.Linear(self.hidden_emb, self.grid_col),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, soft_coefficient):
        # (cell_id, utilization, wire_mask_1, mask_1, wire_mask_2, mask_2, cell_width, cell_height)
        # pdb.set_trace()
        mask = x[:, 1+self.grid_col*2 : 1+self.grid_col*3]

        # utilization mask
        utilization_mask = x [:, 1:1+self.grid_col]

        # utilization resource
        utilization_res = self.utilization_extractor(utilization_mask)

        # wire_mask_1 and wire_mask_2
        # (curr_view, wire_mask_1, wire_mask_2)
        coarse_input = torch.cat((x[:, 1:1+self.grid_col*2], x[:, 1+self.grid_col*3:1+self.grid_col*4]), axis=1)
        coarse_res = self.coarse_extractor(coarse_input)

        res = self.merge_extractor(torch.cat([utilization_res, coarse_res], axis=1))

        # wire_mask_1
        net_img = x[:, 1+self.grid_col: 1+self.grid_col*2]
        # wire_mask_1 + large_value * mask
        net_img = net_img + x[:, 1+self.grid_col*2: 1+self.grid_col*3]*10

        # soft coefficient
        # the increasement of the HPWL is restricted to min_net_img + 1
        net_img_min = net_img.min() + soft_coefficient

        # the minimum resource
        mask2 = net_img.le(net_img_min).logical_not().float()

        x = res

        x = torch.where(mask + mask2 >= 1.0, -1.0e10, x.double())

        x = self.softmax(x)

        return x, utilization_res

# class ActorColnew(nn.Module):
#     def __init__ (self, grid_col):


class Critic(nn.Module):
    def __init__(self, cnn, gcn, res_net):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        # position
        self.pos_emb = nn.Embedding(9900, 64)
        self.cnn = cnn
        self.gcn = gcn
    def forward(self, x, graph = None, cnn_res = None, gcn_res = None, graph_node = None):
        # we should see the critic net
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value

class CriticCol(nn.Module):
    def __init__(self):
        super(CriticCol, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.state_value = nn.Linear(64, 1)
        self.pos_emb = nn.Embedding(9900, 64)
    
    def forward(self, x):
        x1 = F.relu(self.fc1(self.pos_emb(x[:, 0].long())))
        x2 = F.relu(self.fc2(x1))
        value = self.state_value(x2)
        return value
    