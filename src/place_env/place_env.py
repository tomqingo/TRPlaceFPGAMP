import math
import gym
from gym import spaces
import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import pdb
import math


class PlaceColEnv(gym.Env):
    
    def __init__(self, database, grid_col, grid_col_capacity, placed_num_macro=None):
        print("grid_col", grid_col)
        print("grid_capacity", grid_col_capacity)

        assert sum(grid_col_capacity) >= database.num_macro

        # the number of the cols in the grid graph
        self.grid_col = grid_col
        self.sitemap_height = database.sitemap_height
        self.sitemap_width = database.sitemap_width

        # grid capacity
        self.grid_col_capacity = grid_col_capacity
        self.database = database

        self.num_macro = database.num_macro
        self.num_net = database.num_placementnets
        self.nodeidlist = database.nodeidlist

        self.action_space = spaces.Discrete(self.grid_col)

        self.state = None

        self.net_min_max_ord ={}
        self.node_pos = {}
        self.net_placed_set = {}
        self.last_reward = 0
        self.num_macro_placed = 0

        self.placed_num_macro = placed_num_macro
    

    def reset(self):
        self.num_macro_placed = 0
        num_macro = self.num_macro
        # Use the one dimensional array to place all the cells
        arr = np.zeros(self.grid_col)
        arr_prop = np.zeros(self.grid_col)

        for port in self.database.ports:
            gridlocX = self.database.colid2gridcolid[port.locX]
            arr_prop[gridlocX] = 1
        
        self.node_pos = {}
        self.net_min_max_ord = {}

        self.net_fea = np.zeros((self.database.num_placementnets, 4))
        self.net_fea[:, 0] = 0
        self.net_fea[:, 1] = 1.0

        for port in self.database.ports:
            for netId in port.netIds:
                pin_x = port.locX
                pin_grid_x = self.database.colid2gridcolid[pin_x]
                arr[pin_grid_x] += 1
                # pin_grid_x = 1
                if netId in self.database.netid2placementnetid.keys():
                    placementnetid = self.database.netid2placementnetid[netId]
                    if placementnetid in self.net_min_max_ord:
                        if pin_x > self.net_min_max_ord[placementnetid]['max_x']:
                            self.net_min_max_ord[placementnetid]['max_x'] = pin_x
                            self.net_fea[placementnetid]['max_x'] = pin_x * 1.0 / self.sitemap_width
                        elif pin_x < self.net_min_max_ord[placementnetid]['min_x']:
                            self.net_min_max_ord[placementnetid]['min_x'] = pin_x
                            self.net_fea[placementnetid]['min_x'] = pin_x * 1.0 / self.sitemap_width
                    else:
                        self.net_min_max_ord[placementnetid] = {}
                        self.net_min_max_ord[placementnetid]['max_x'] = pin_x
                        self.net_min_max_ord[placementnetid]['min_x'] = pin_x

                        self.net_fea[placementnetid][1] = pin_x * 1.0 / self.sitemap_width
                        self.net_fea[placementnetid][0] = pin_x * 1.0 / self.sitemap_width
        
        self.net_placed_set = {}
        net_img = np.zeros(self.grid_col)
        net_img_2 = np.zeros(self.grid_col)

        next_y = math.ceil(max(1, self.database.macros[self.nodeidlist[self.num_macro_placed]].num_cells))
        macrotype = self.database.macros[self.nodeidlist[self.num_macro_placed]].macrotype
        mask = self.get_mask(arr, next_y, macrotype)

        # calculate the number of sites in one column
        numsitescol = self.sitemap_height
        if "BRAM" in macrotype or "RAMB" in macrotype:
            numsitescol = int(self.sitemap_height / 5)
        elif "DSP" in macrotype:
            numsitescol = int(self.sitemap_height / 2.5)

        next_y_2 = math.ceil(max(1, self.database.macros[self.nodeidlist[self.num_macro_placed + 1]].num_cells))
        macrotype_2 = self.database.macros[self.nodeidlist[self.num_macro_placed + 1]].macrotype
        mask_2 = self.get_mask(arr, next_y_2, macrotype_2)

        for net in self.database.placementnets:
            self.net_placed_set[net.id] = set()
        
        self.state = np.concatenate((np.array([self.num_macro_placed]), arr_prop, net_img, mask, net_img_2, 
            mask_2, np.array([next_y / numsitescol])), axis = 0)
        
        return self.state

    def get_net_img(self, is_next_next = False):

        net_img = np.zeros(self.grid_col)

        if not is_next_next:
            next_macro_id = self.nodeidlist[self.num_macro_placed]
        elif self.num_macro_placed + 1 < len(self.nodeidlist):
            next_macro_id = self.nodeidlist[self.num_macro_placed + 1]
        else:
            return net_img
        
        macro = self.database.macros[next_macro_id]

        for node in macro.Macronodecol:
            for netid in node.netIds:
                if netid not in self.database.netid2placementnetid.keys():
                    continue
                placementnetid = self.database.netid2placementnetid[netid]
                if placementnetid in self.net_min_max_ord:
                    start_x = self.net_min_max_ord[placementnetid]['min_x']
                    start_x_grid = self.database.colid2gridcolid[start_x]

                    end_x = self.net_min_max_ord[placementnetid]['max_x']
                    end_x_grid = self.database.colid2gridcolid[end_x]
                    weight = 1.0

                    for i in range(0, start_x_grid):
                        real_x = self.database.gridcolid2colid[i]
                        net_img[i] += (start_x - real_x)*weight
                    
                    for i in range(end_x_grid+1, self.grid_col):
                        real_x = self.database.gridcolid2colid[i]
                        net_img[i] += (real_x - end_x)*weight
        
        return net_img
    
    def step(self, action):
        arr_prop = self.state[1:1+self.grid_col]
        arr = arr_prop * self.grid_col_capacity

        mask = self.state[1+self.grid_col*2:1+self.grid_col*3]
        reward = 0

        # actions
        if mask[action] == 1:
            reward += (-200000)
        
        # cascade id
        cascade_id = self.nodeidlist[self.num_macro_placed]
        macro = self.database.macros[cascade_id]

        size_x = 1
        size_y = macro.num_cells

        arr[action] += size_y
        arr_prop[action] = arr[action] / self.grid_col_capacity[action]

        self.node_pos[cascade_id] = action

        # the connected nets
        net_col = []
        for node in macro.Macronodecol:
            for netid in node.netIds:
                if netid not in self.net_placed_set:
                    continue
                if netid not in net_col:
                    net_col.append(netid)
                self.net_placed_set[netid].add(node.id)
                placementnetid = self.database.netid2placementnetid[netid]
                pin_x = action
                pin_x_real = self.database.gridcolid2colid[pin_x]

                if placementnetid in self.net_min_max_ord:
                    start_x = self.net_min_max_ord[placementnetid]['min_x']
                    end_x = self.net_min_max_ord[placementnetid]['max_x']
                    weight = 1.0

                    if pin_x_real > self.net_min_max_ord[placementnetid]['max_x']:
                        reward += weight * (self.net_min_max_ord[placementnetid]['max_x'] - pin_x_real)
                        self.net_min_max_ord[placementnetid]['max_x'] = pin_x_real
                        self.net_fea[placementnetid][1] = pin_x_real / self.sitemap_width
                    elif pin_x_real < self.net_min_max_ord[placementnetid]['min_x']:
                        reward += weight * (pin_x_real - self.net_min_max_ord[placementnetid]['min_x'])
                        self.net_min_max_ord[placementnetid]['min_x'] = pin_x_real
                        self.net_fea[placementnetid][0] = pin_x_real / self.sitemap_width
                    
                else:
                    self.net_min_max_ord[placementnetid] = {}
                    self.net_min_max_ord[placementnetid]['max_x'] = pin_x_real
                    self.net_min_max_ord[placementnetid]['min_x'] = pin_x_real
                    self.net_fea[placementnetid][1] = pin_x_real / self.sitemap_width
                    self.net_fea[placementnetid][0] = pin_x_real / self.sitemap_width

                    reward += 0
        
        self.num_macro_placed += 1
        net_img = np.zeros(self.grid_col)
        net_img_2 = np.zeros(self.grid_col)

        if self.num_macro_placed < self.placed_num_macro:
            net_img = self.get_net_img()
            net_img_2 = self.get_net_img(is_next_next=True)
            if net_img.max() > 0 or net_img_2.max() > 0:
                net_img /= (max(net_img.max(), net_img_2.max())*1.0)
                net_img_2 /= (max(net_img.max(), net_img_2.max()*1.0))
        
        if self.num_macro_placed == self.num_macro or (self.placed_num_macro is not None and self.num_macro_placed == self.placed_num_macro):
            done = True
        else:
            done = False
        
        mask = np.ones(self.grid_col)
        mask_2 = np.ones(self.grid_col)
        numsitescol = self.sitemap_height

        if not done:
            next_y = self.database.macros[self.nodeidlist[self.num_macro_placed]].num_cells
            macrotype = self.database.macros[self.nodeidlist[self.num_macro_placed]].macrotype
            if "BRAM" in macrotype or "RAMB" in macrotype:
                numsitescol = int(self.sitemap_height / 5)
            elif "DSP" in macrotype:
                numsitescol = int(self.sitemap_height / 2.5)
            mask = self.get_mask(arr, next_y, macrotype)
            if self.num_macro_placed + 1 < self.placed_num_macro:
                next_y_2 = self.database.macros[self.nodeidlist[self.num_macro_placed+1]].num_cells
                macrotype_2 = self.database.macros[self.nodeidlist[self.num_macro_placed+1]].macrotype                
                mask_2 = self.get_mask(arr, next_y_2, macrotype_2)
        else:
            next_y = 0

        #print(arr.shape, net_img.shape, mask.shape, net_img_2.shape, mask_2.shape, np.array([next_y/numsitescol]).shape)
        self.state = np.concatenate((np.array([self.num_macro_placed]), arr_prop, net_img,
                                    mask, net_img_2, mask_2, np.array([next_y/numsitescol])), axis=0)
        return self.state, reward, done, {"raw_reward": reward, "net_img":net_img, "mask":mask}
    
    def get_mask(self, arr, next_y, macrotype):
        mask = np.zeros(self.grid_col)
        uitlization_coeff = 1.0

        # select the sites for placing BRAMs or DSPs
        if "BRAM" in macrotype or "RAMB" in macrotype:
            selectcol = self.database.BRAMgridcols
            selectrow = self.database.BRAMgridrows
        elif "DSP" in macrotype:
            selectcol = self.database.DSPgridcols
            selectrow = self.database.DSPgridrows

        for colid in range(self.grid_col):
            if colid not in selectcol:
                mask[colid] = 1
            if arr[colid] + next_y > uitlization_coeff * self.grid_col_capacity[colid]:
                mask[colid] = 1

        return mask


class PlaceEnv(gym.Env):

    def __init__(self, database, grid_width, grid_height, placed_num_macro = None):

        print("grid_width * grid_height", grid_width * grid_height)
        print("database.node_cnt", database.num_macro)
        print("database.net_cnt", database.num_placementnets)
        assert grid_width * grid_height >= database.num_macro

        # the height and the width of the grid map
        self.grid_width = grid_width
        self.grid_height = grid_height

        # the height and the width of the site map
        self.sitemap_width = database.sitemap_width
        self.sitemap_height = database.sitemap_height

        # database
        self.database = database

        self.num_macro = database.num_macro # the total number of the macros considering the simple and cascaded macros
        self.num_net = database.num_placementnets # the total number of the nets connecting the macros and the IOs
        self.nodeidlist = database.nodeidlist # the list of the macro placement

        # the action space is the probability matrix (grid_width * grid_height)
        self.action_space = spaces.Discrete(self.grid_width * self.grid_height)
        
        self.state = None

        self.net_min_max_ord = {}
        self.node_pos = {}
        self.net_placed_set = {}
        self.last_reward = 0
        self.num_macro_placed = 0

        self.placed_num_macro = placed_num_macro

        
    def reset (self):
        self.num_macro_placed = 0
        num_macro = self.num_macro
        canvas = np.zeros((self.grid_width, self.grid_height))

        # add the IO ports
        for port in self.database.ports:
            gridlocX = self.database.colid2gridcolid[port.locX]
            gridlocY = port.locY
            canvas[gridlocX][gridlocY] = 1
        
        self.node_pos = {}
        self.net_min_max_ord = {}  # the real coordinate on the FPGA board

        self.net_fea = np.zeros((self.database.num_placementnets, 4))
        self.net_fea[:, 0] = 0
        self.net_fea[:, 1] = 1.0
        self.net_fea[:, 2] = 0
        self.net_fea[:, 3] = 1.0   # the real coordinates on the PFGA board
        
        # Obtain the port information
        for port in self.database.ports:
            for netId in port.netIds:
                pin_x = port.locX
                pin_y = port.locY
                if netId in self.database.netid2placementnetid.keys():
                    placementnetid = self.database.netid2placementnetid[netId]
                    # (pin_x, pin_y) are the real coordinates
                    if placementnetid in self.net_min_max_ord:
                        if pin_x > self.net_min_max_ord[placementnetid]['max_x']:
                            self.net_min_max_ord[placementnetid]['max_x'] = pin_x
                            self.net_fea[placementnetid][1] = pin_x * 1.0 / self.sitemap_width
                        elif pin_x < self.net_min_max_ord[placementnetid]['min_x']:
                            self.net_min_max_ord[placementnetid]['min_x'] = pin_x
                            self.net_fea[placementnetid][0] = pin_x * 1.0 / self.sitemap_width
                        if pin_y > self.net_min_max_ord[placementnetid]['max_y']:
                            self.net_min_max_ord[placementnetid]['max_y'] = pin_y
                            self.net_fea[placementnetid][3] = pin_y * 1.0 / self.sitemap_height
                        elif pin_y < self.net_min_max_ord[placementnetid]['min_y']:
                            self.net_min_max_ord[placementnetid]['min_y'] = pin_y
                            self.net_fea[placementnetid][2] = pin_y * 1.0 / self.sitemap_height
                    else:
                        self.net_min_max_ord[placementnetid] = {}
                        self.net_min_max_ord[placementnetid]['max_x'] = pin_x
                        self.net_min_max_ord[placementnetid]['min_x'] = pin_x

                        self.net_min_max_ord[placementnetid]['max_y'] = pin_y
                        self.net_min_max_ord[placementnetid]['min_y'] = pin_y
                        self.net_fea[placementnetid][1] = pin_x * 1.0 / self.sitemap_width
                        self.net_fea[placementnetid][0] = pin_x * 1.0 / self.sitemap_width
                        self.net_fea[placementnetid][3] = pin_y * 1.0 / self.sitemap_height
                        self.net_fea[placementnetid][2] = pin_y * 1.0 / self.sitemap_height

        self.net_placed_set = {}

        net_img = np.zeros((self.grid_width, self.grid_height))
        net_img_2 = np.zeros((self.grid_width, self.grid_height))

        # the width of the next macro
        next_x = 1
        # the height of the next macro
        next_y = math.ceil(max(1, self.database.macros[self.nodeidlist[self.num_macro_placed]].height))
        macrotype = self.database.macros[self.nodeidlist[self.num_macro_placed]].macrotype
        mask = self.get_mask(canvas, next_x, next_y, macrotype)
        # the width of the next and the next macro
        next_x_2 = 1
        # the height of the next and the next macro (Q learning process)
        next_y_2 = math.ceil(max(1, self.database.macros[self.nodeidlist[self.num_macro_placed + 1]].height))
        macrotype = self.database.macros[self.nodeidlist[self.num_macro_placed + 1]].macrotype
        mask_2 = self.get_mask(canvas, next_x_2, next_y_2, macrotype)

        # net_placed_set
        for net in self.database.placementnets:
            self.net_placed_set[net.id] = set()
        
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(), net_img.flatten(), mask.flatten(), net_img_2.flatten(),
            mask_2.flatten(), np.array([next_x / self.sitemap_width, next_y / self.sitemap_height])), axis = 0)

        return self.state
    
    # get the wire mask (net image calculation)
    def get_net_img(self, is_next_next = False):

        net_img = np.zeros ((self.grid_width, self.grid_height))

        # next_macro_id
        if not is_next_next:
            next_macro_id = self.nodeidlist[self.num_macro_placed]
        elif self.num_macro_placed + 1 < len(self.nodeidlist):
            next_macro_id = self.nodeidlist[self.num_macro_placed + 1]
        else:
            return net_img

        # timing complexity: num_macro * avg_connected_nets
        macro  = self.database.macros[next_macro_id]
        
        # All the nodes in the cascaded macros
        cnt = 0
        for node in macro.Macronodecol:
            net_cnt = 0
            for netid in node.netIds:
                # net is the placement net
                if netid not in self.database.netid2placementnetid.keys():
                    continue
                placementnetid = self.database.netid2placementnetid[netid]
                # Whether there are some problems concerning the cell order
                if placementnetid in self.net_min_max_ord:
                    # print(net_img)
                    delta_pin_x = 0   # the pin offset of the macro is 0
                    delta_pin_y = cnt * macro.height / macro.num_row  # delta_pin_y
                    # net_min_max_ord: the boundary of the nets (real coordinates)
                    # <macro size, pin offset>
                    start_x = self.net_min_max_ord[placementnetid]["min_x"] - delta_pin_x
                    start_x_grid = self.database.colid2gridcolid[start_x]

                    end_x = self.net_min_max_ord[placementnetid]['max_x'] - delta_pin_x
                    end_x_grid = self.database.colid2gridcolid[end_x]

                    start_y = int(self.net_min_max_ord[placementnetid]['min_y'] - delta_pin_y)
                    end_y = int(self.net_min_max_ord[placementnetid]['max_y'] - delta_pin_y)
                    
                    start_x = min(start_x, self.database.sitemap_width)
                    start_x_grid = min(start_x_grid, self.database.grid_width)
                    start_y = min(start_y, self.database.sitemap_width)

                    weight = self.database.nets[netid].weight

                    for i in range(0, start_x_grid):
                        real_x = self.database.gridcolid2colid[i]
                        net_img[i, :] += (start_x - real_x) * weight

                    for i in range(end_x_grid+1, self.grid_width):
                        real_x = self.database.gridcolid2colid[i]
                        net_img[i, :] +=  (real_x - end_x) * weight
                    
                    for j in range(0, start_y):
                        net_img[:, j] += (start_y - j) * weight

                    for j in range(end_y+1, self.grid_height):
                        net_img[:, j] += (j - end_y) * weight
            cnt += 1
        return net_img

    def step(self, action):

        canvas = self.state[1: 1+self.grid_width*self.grid_height].reshape(self.grid_width, self.grid_height)
        mask = self.state[1+self.grid_width*self.grid_height*2: 1+self.grid_width*self.grid_height*3].reshape(self.grid_width, self.grid_height)
        reward = 0
        x_reward  = 0

        # actions (grid_col, grid_row)
        x = round(action // self.grid_height)
        y = round(action % self.grid_height)

        # Why don't we select the action at the action selection period
        # Awards
        if mask[x][y] == 1:
            reward += -200000
            x_reward = -200000
        
        cascade_id = self.nodeidlist[self.num_macro_placed]
        macro = self.database.macros[cascade_id]

        size_x = 1
        size_y = max(1, macro.height)

        # For the DSP, because the size of the macro is 2.5
        # the position could be upper or lower, see which is in the DSPgridrows
        end_cell_y_upper = math.ceil(y+size_y)
        end_cell_y_lower = math.floor(y+size_y)

        if end_cell_y_upper in self.database.DSPgridrows:
            end_cell_y = end_cell_y_upper
        else:
            end_cell_y = end_cell_y_lower

        canvas[x : x+size_x, y: end_cell_y] = 1.0

        # (x, y, size_x, size_y): location (x,y), macro size: (size_x, size_y)
        self.node_pos[cascade_id] = (x, y, size_x, size_y)
        
        # Print the macro location
        # print(macro.name, self.database.gridcolid2colid[x], y)
        # print(macro.name, "Mask Non Zero: ", np.sum(np.where(mask == 0, 1, 0)))
        # if "BRAM" in macro.macrotype or "RAMB" in macro.macrotype:
        #     if x not in self.database.BRAMgridcols or y not in self.database.BRAMgridrows:
        #         print("BRAM wrong coordinates: ", macro.name, self.database.gridcolid2colid[x], y)
        # elif "DSP" in macro.macrotype:
        #     # pdb.set_trace()
        #     print("DSP coordinates: ", macro.name, self.database.gridcolid2colid[x], y)           
        #     if x not in self.database.DSPgridcols or y not in self.database.DSPgridrows:
        #         print("DSP wrong coordinates: ", macro.name, self.database.gridcolid2colid[x], y)

        # node.id
        cnt = 0
        cnt_pdb = 0
        
        # print(len(self.net_min_max_ord))
        # pdb.set_trace()
        net_col = []

        for node in macro.Macronodecol:
            for netid in node.netIds:
                # If the net is not the placement net (the netid); If the net is the placement net
                if netid not in self.net_placed_set:
                    continue
                
                if netid not in net_col:
                    net_col.append(netid)

                # Add the macros to the net_placed_set
                self.net_placed_set[netid].add(node.id)

                pin_x = x
                pin_x_real = self.database.gridcolid2colid[pin_x] # the real pin x index

                offset_y = cnt * macro.height / macro.num_row # the offset of the node

                end_cell_y_upper = math.ceil(y+offset_y)
                end_cell_y_lower = math.floor(y+offset_y)

                if end_cell_y_upper in self.database.DSPgridrows:
                    end_cell_y = end_cell_y_upper
                else:
                    end_cell_y = end_cell_y_lower

                pin_y =  end_cell_y  # we need to ensure the y is 5 times
                placementnetid = self.database.netid2placementnetid[netid]
                
                if placementnetid in self.net_min_max_ord:
                    start_x = self.net_min_max_ord[placementnetid]['min_x']
                    end_x = self.net_min_max_ord[placementnetid]['max_x']
                    start_y = self.net_min_max_ord[placementnetid]['min_y']
                    end_y = self.net_min_max_ord[placementnetid]['max_y']
                    weight = 1.0

                    if pin_x_real > self.net_min_max_ord[placementnetid]['max_x']:
                        reward += weight * (self.net_min_max_ord[placementnetid]['max_x'] - pin_x_real)
                        x_reward += weight * (self.net_min_max_ord[placementnetid]['max_x'] - pin_x_real)
                        self.net_min_max_ord[placementnetid]['max_x'] = pin_x_real
                        self.net_fea[placementnetid][1] = pin_x_real / self.sitemap_width
                    elif pin_x_real < self.net_min_max_ord[placementnetid]['min_x']:
                        reward += weight * (pin_x_real - self.net_min_max_ord[placementnetid]['min_x'])
                        x_reward += weight * (pin_x_real - self.net_min_max_ord[placementnetid]['min_x'])
                        self.net_min_max_ord[placementnetid]['min_x'] = pin_x_real
                        self.net_fea[placementnetid][0] = pin_x_real / self.sitemap_width
                    if pin_y > self.net_min_max_ord[placementnetid]['max_y']:
                        reward += weight * (self.net_min_max_ord[placementnetid]['max_y'] - pin_y)
                        self.net_min_max_ord[placementnetid]['max_y'] = pin_y
                        self.net_fea[placementnetid][3] = pin_y / self.sitemap_height
                    elif pin_y < self.net_min_max_ord[placementnetid]['min_y']:
                        reward += weight * (pin_y - self.net_min_max_ord[placementnetid]['min_y'])
                        self.net_min_max_ord[placementnetid]['min_y'] = pin_y
                        self.net_fea[placementnetid][2] = pin_y / self.sitemap_height                        
                else:
                    self.net_min_max_ord[placementnetid] = {}
                    self.net_min_max_ord[placementnetid]['max_x'] = pin_x_real
                    self.net_min_max_ord[placementnetid]['min_x'] = pin_x_real
                    self.net_min_max_ord[placementnetid]['max_y'] = pin_y
                    self.net_min_max_ord[placementnetid]['min_y'] = pin_y
                    self.net_fea[placementnetid][1] = pin_x_real / self.sitemap_width
                    self.net_fea[placementnetid][0] = pin_x_real / self.sitemap_width
                    self.net_fea[placementnetid][3] = pin_y / self.sitemap_height
                    self.net_fea[placementnetid][2] = pin_y / self.sitemap_height
                    reward += 0
                    x_reward += 0
                    cnt_pdb += 1
            cnt += 1

        # print(cnt_pdb)
        # print(len(net_col))
        # pdb.set_trace()

        self.num_macro_placed += 1
        net_img = np.zeros((self.grid_width, self.grid_height))
        net_img_2 = np.zeros((self.grid_width, self.grid_height))

        if self.num_macro_placed < self.placed_num_macro:
            net_img = self.get_net_img()
            net_img_2 = self.get_net_img(is_next_next = True)
            if net_img.max() > 0 or net_img_2.max() > 0:
                net_img /= (max(net_img.max(), net_img_2.max())*1.0)
                net_img_2 /= (max(net_img.max(), net_img_2.max())*1.0)
        
        if self.num_macro_placed == self.num_macro or \
            (self.placed_num_macro is not None and self.num_macro_placed == self.placed_num_macro): 
            done = True
        else:
            done = False

        mask = np.ones((self.grid_width, self.grid_height))
        mask_2 = np.ones((self.grid_width, self.grid_height))
        if not done:
            next_x = 1
            next_y = self.database.macros[self.nodeidlist[self.num_macro_placed]].height
            macrotype = self.database.macros[self.nodeidlist[self.num_macro_placed]].macrotype
            mask = self.get_mask(canvas, next_x, next_y, macrotype)
            if self.num_macro_placed + 1 < self.placed_num_macro:
                next_x_2 = 1
                next_y_2 = self.database.macros[self.nodeidlist[self.num_macro_placed+1]].height
                macrotype_2 = self.database.macros[self.nodeidlist[self.num_macro_placed+1]].macrotype
                mask_2 = self.get_mask(canvas, next_x_2, next_y_2, macrotype)
        else:
            next_x = 0
            next_y = 0
        
        self.state = np.concatenate((np.array([self.num_macro_placed]), canvas.flatten(),
            net_img.flatten(), mask.flatten(), net_img_2.flatten(), mask_2.flatten(),
            np.array([next_x / self.sitemap_width, next_y/self.sitemap_height])), axis = 0)
        return self.state, reward, done, {"raw_reward": reward, "net_img": net_img, "mask": mask, "x_reward": x_reward}
    
    # For different kinds of macros
    # get the mask of the placement
    def get_mask(self, canvas, next_x, next_y, macrotype):
        mask = np.zeros((self.grid_width, self.grid_height))

        # select the site for placing BRAMs or DSPs
        if "BRAM" in macrotype or "RAMB" in macrotype:
            selectcol = self.database.BRAMgridcols
            selectrow = self.database.BRAMgridrows
        elif "DSP" in macrotype:
            selectcol = self.database.DSPgridcols
            selectrow = self.database.DSPgridrows
        
        for colid in range(self.grid_width):
            if colid not in selectcol:
                mask[colid, :] = 1
        
        for rowid in range(self.grid_height):
            if rowid not in selectrow:
                mask[:, rowid] = 1

        # print("Mask Non Zero Stage 1: ", np.sum(np.where(mask == 0, 1, 0)))

        # No overlap with the placed macros
        cnt = 0
        for nodeid in self.node_pos:
            startx = max(0, self.node_pos[nodeid][0] - next_x + 1)
            # starty
            start_cell_y_lower = math.floor(self.node_pos[nodeid][1] - next_y)
            start_cell_y_upper = math.ceil(self.node_pos[nodeid][1] - next_y)
            if start_cell_y_upper in self.database.DSPgridrows:
                starty = start_cell_y_upper
            else:
                starty = start_cell_y_lower

            starty = max(0, starty + 1)

            endx = min(self.node_pos[nodeid][0] + self.node_pos[nodeid][2] - 1, self.grid_width - 1)
            
            # endy
            end_cell_y_lower = math.floor(self.node_pos[nodeid][1] + self.node_pos[nodeid][3])
            end_cell_y_upper = math.ceil(self.node_pos[nodeid][1] + self.node_pos[nodeid][3])

            if end_cell_y_upper in self.database.DSPgridrows:
                endy = end_cell_y_upper
            else:
                endy = end_cell_y_lower

            endy = min(endy - 1, self.grid_height - 1)
            # print(self.database.macros[nodeid].name, startx, starty, endx, endy)
            mask[startx: endx + 1, starty : endy + 1] = 1

            # if cnt == len(self.node_pos) - 1:
            #     print(startx, endx, starty, endy, self.node_pos[nodeid][0], self.node_pos[nodeid][1], self.node_pos[nodeid][2], self.node_pos[nodeid][3])
            #     print(next_x, next_y)

            cnt += 1

        # print("Mask Non Zero Stage 2: ", np.sum(np.where(mask == 0, 1, 0)))

        # Not exceed the boundary

        mask[self.grid_width - next_x + 1:,:] = 1

        boundary_y_lower = math.floor(self.grid_height - next_y)
        boundary_y_upper = math.ceil(self.grid_height - next_y)

        if boundary_y_upper in self.database.DSPgridrows:
            boundary_y = boundary_y_upper
        else:
            boundary_y = boundary_y_lower

        mask[:, boundary_y + 1:] = 1

        # print("Mask Non Zero Stage 3: ", np.sum(np.where(mask == 0, 1, 0)))

        return mask
        










                    


        
