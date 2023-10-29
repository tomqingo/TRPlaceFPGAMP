import collections
import copy
import math
from utils import *
from src.db import *
from src.MacroPl import *
import os
from random import choice

class Dataset:
    def __init__(self, params):
        self.params = params #The path to each file

        # Conversion dict
        self.cellLib = {} #cellLib, the dict variable
        self.nodeName2Id = {} #Find the id using node name
        self.netName2Id = {} #Find the id using net name
        self.ResourceName2Id = {} #Find the id using resource name
        self.CellType2ResourceType = {} #Find the resource types corresponding to the cell types 
        self.sitetype2Id = {} #Find the id using resource name
        self.cascademacrotype2Id = {} #Find the id using the macro type name
        self.sitecolumns = {} #The site columns for each 

        # node, net, site sets
        self.nodes = [] #nodes col, the list variable
        self.nets = [] #nets col, the list variable
        self.nodeswithRegionConstr = [] #nodes col with regional constraint
        self.clknets = [] #clk_nets col, the list variable
        self.highdegreenets = [] #high degree nets col, the list variable
        self.sites = [] #The sites that could place the nodes

        ## type sets
        self.resources = [] #resource type col {LUT, FF, CARRY8, DSP48E2, RAMB36E2, IO}
        self.sitetypes = [] #site type col {SLICE, DSP, BRAM, IO}
        self.cascademacrotypes = [] # macro type col {BRAM_CASCADE_x, DSP_CASCADE_x}
        self.cascademacros = [] #All the cascaded macro instances
        self.regionconstrtype = [] #Region Constraint Type

        ##Basic information
        #netlist statistics
        self.num_cells = 0 #number of the cells in lib
        self.num_nodes = 0 
        self.num_nets = 0
        self.num_resource_demand = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "IO":0}
        self.num_resource_supply = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "IO":0}
        
        #Macro number statistics
        self.num_basic_macro = 0  
        self.num_cascade_macro = 0  #number of cascaded macros
        self.num_cascade_node = 0   #number of nodes in the cascade macros
        self.num_macro = 0

        #SiteMap
        self.sitemap_width = 0
        self.sitemap_height = 0
        self.num_site = 0 #number of the sites
        self.num_site_dict = {"SLICE":0, "DSP":0, "BRAM":0, "IO":0}
        self.num_bel = 0

        #fix node
        self.num_fix = 0

        #regional constraints
        self.num_region_constr = 0
        self.num_region_constr_node = 0
        self.num_region_constr_macronode = 0
        self.num_region_constr_cascademacronode = 0
        self.regionconstr_area = []

        #number of clk nets
        self.num_clk_nets = 0

        #number of high degree nets
        self.num_high_degree_nets = 0
    
    # Read the design.nodes file
    def readNodes(self):
        with open(self.params["nodes"], "r") as f_node:
            all_lines = f_node.read().splitlines()
            for id in range(len(all_lines)):
                cur_line = all_lines[id]
                if cur_line[0] == "#":
                    continue
                node_info = cur_line.split()
                node = Node(node_info[0], len(self.nodes), node_info[1])
                # There exists some node types not in the cell library
                if node_info[1] in list(self.cellLib.keys()):
                    node.addPin(self.cellLib[node_info[1]].pins)
                self.nodeName2Id[node_info[0]] = len(self.nodes)
                self.nodes.append(node)   
    
    # Read the design.nets file
    def readNets(self):
        with open(self.params["nets"], "r") as f_net:
            all_lines = f_net.read().splitlines()
            cur_line_id = 0
            while cur_line_id < len(all_lines):
                cur_line = all_lines[cur_line_id]
                if cur_line[0]== "#":
                    cur_line_id += 1
                    continue
                cur_line_col = cur_line.strip().split()
                if cur_line_col[0] == "net":
                    net_name = cur_line_col[1]
                    net = Net(net_name, len(self.nets))
                    while cur_line_id < len(all_lines):
                        cur_line_id += 1
                        cur_line = all_lines[cur_line_id]
                        #print(cur_line)
                        netpin_info = cur_line.strip().split()
                        if netpin_info[0] == "endnet":
                            self.nodes[self.nodeName2Id[nodename]].addNeighboringOutNets(net.id)
                            break
                        nodename = netpin_info[0]
                        pinname = netpin_info[1]
                        # Update the connected nets for different nodes
                        nodeid = self.nodeName2Id[nodename]
                        self.nodes[nodeid].addNeighboringNets(net.id)
                        self.nodes[nodeid].addNeighboringInNets(net.id)
                        # Some pins are not in the pin list of the cells
                        celltype = self.nodes[nodeid].celltype
                        if not pinname in list(self.cellLib[celltype].pinName2Id):
                            if "IN" in pinname:
                                pin_IO = 1
                            else:
                                pin_IO = 0
                            self.cellLib[celltype].addPin(pinname, pin_IO, False, False)
                            self.nodes[nodeid].addPin([self.cellLib[celltype].pins[-1]])
                        # Add the pins to the node set of the nets
                        pinId = self.cellLib[celltype].pinName2Id[pinname]
                        pin = self.cellLib[celltype].pins[pinId]
                        net.addPin([nodeid, pin])
                        # Add the macro pins to the macro set of the nets
                        if self.nodes[nodeid].is_macro:
                            net.addMacroPin(nodeid)
                    # Judge whether the net is high-degree net
                    net.setHighDegreeNet()
                    if net.ishighdegree:
                        self.highdegreenets.append(net)
                    # Judge whether the net is clk net
                    if net.isclknet:
                        self.clknets.append(net)
                    self.netName2Id[net_name] = len(self.nets)
                    self.nets.append(net)
                    cur_line_id += 1


    # Read the design.lib file                     
    def readCellLibs(self):
        with open(self.params["lib"], "r") as f_lib:
            all_lines = f_lib.read().splitlines()
            cell_name = ""
            for id in range(len(all_lines)):
                cur_line = all_lines[id]
                if len(cur_line) == 0 or (cur_line[0] == "#" or "END" in cur_line):
                    continue
                if "CELL" in cur_line:
                    cell_name = cur_line.split()[1]
                    cell = Cell(cell_name, len(self.cellLib))
                    self.cellLib[cell_name] = cell
                elif "PIN" in cur_line:
                    pin_info = cur_line.strip().split()
                    if pin_info[2] == "INPUT":
                        pin_IO = 1
                    else:
                        pin_IO = 0                        
                    pin_clock = False
                    pin_ctrl = False
                    if len(pin_info) > 3:
                        if pin_info[3] == "CLOCK":
                            pin_clock = True
                        if pin_info[3] == "CTRL":
                            pin_ctrl = True
                    self.cellLib[cell_name].addPin(pin_info[1], pin_IO, pin_clock, pin_ctrl)
    
    # Read the design.scl file
    def readSiteMaps(self):
        with open(self.params["sitemap"],"r") as f_scl:
            all_lines = f_scl.read().splitlines()
            cur_line_id = 0
            while cur_line_id < len(all_lines):
                cur_line = all_lines[cur_line_id]
                if len(cur_line) == 0 or ("END" in cur_line or "#" in cur_line):
                    cur_line_id += 1
                    continue
                cur_line_col = cur_line.strip().split()
                cur_line_id += 1
                if "SITE" in cur_line_col:
                    if cur_line_col[1] == "URAM":
                        cur_line_id += 3
                        continue
                    sitetype = SiteType(cur_line_col[1], len(self.sitetypes))
                    while cur_line_id < len(all_lines):
                        sub_cur_line = all_lines[cur_line_id]
                        sub_cur_line_col = sub_cur_line.strip().split()
                        if sub_cur_line_col[0] == "END" and sub_cur_line_col[1] == "SITE":
                            break
                        sitetype.AddResourceMulti(sub_cur_line_col[0], int(sub_cur_line_col[1]))
                        cur_line_id += 1
                    self.sitetype2Id[cur_line_col[1]] = sitetype
                    self.sitetypes.append(sitetype)
                elif "RESOURCES" in cur_line_col:
                    while cur_line_id < len(all_lines):
                        cur_line = all_lines[cur_line_id]
                        if "URAM" in cur_line:
                            cur_line_id += 1
                            continue
                        cur_line_col = cur_line.strip().split()
                        if cur_line_col[0] == "END" and cur_line_col[1] == "RESOURCES":
                            break
                        resource = Resource(cur_line_col[0], len(self.resources))
                        for subid in range(1,len(cur_line_col)):
                            resource.AddCelltype(cur_line_col[subid])
                            self.CellType2ResourceType[cur_line_col[subid]] = len(self.resources)
                        # FDSE also belongs to FF, to be mentioned
                        if cur_line_col[0] == "FF":
                            resource.AddCelltype("FDSE")
                            self.CellType2ResourceType["FDSE"] = len(self.resources)
                        cur_line_id += 1
                        self.ResourceName2Id[len(self.resources)] = resource
                        self.resources.append(resource)
                elif "SITEMAP" in cur_line_col:
                    self.sitemap_width = int(cur_line_col[1])
                    self.sitemap_height = int(cur_line_col[2])
                    self.sitemaps = np.ones([self.sitemap_width, self.sitemap_height])*(-1) #2d site maps with the 1d site value, -1 refers to the sites that could not be placed
                    self.sitemap_res = np.ones([self.sitemap_width, self.sitemap_height])*(-1) #2d site maps with the site category, -1 refers to the sites that could not be placed
                    while cur_line_id < len(all_lines):
                        cur_line = all_lines[cur_line_id]
                        if "URAM" in cur_line:
                            cur_line_id += 1
                            continue
                        cur_line_col = cur_line.strip().split()
                        if cur_line_col[0] == "END" and cur_line_col[1] == "SITEMAP":
                            break
                        site = Site(cur_line_col[2], len(self.sites), int(cur_line_col[0]), int(cur_line_col[1]))
                        sitetype_id = self.sitetype2Id[cur_line_col[2]].id
                        site.addSupplyResource(self.sitetypes[sitetype_id].resource)
                        self.sitemaps[int(cur_line_col[0])][int(cur_line_col[1])] = len(self.sites)
                        self.sitemap_res[int(cur_line_col[0])][int(cur_line_col[1])] = self.sitetype2Id[cur_line_col[2]].id
                        self.sites.append(site)
                        for _, res_name in enumerate(list(self.sitetypes[sitetype_id].resource.keys())):     
                            self.num_resource_supply[res_name] += self.sitetypes[sitetype_id].resource[res_name] 
                            self.num_bel += 1
                        cur_line_id += 1
                        self.num_site_dict[cur_line_col[2]] += 1

            for nodeid in range(len(self.nodes)):
                celltype = self.nodes[nodeid].celltype
                if celltype in list(self.CellType2ResourceType.keys()):
                    restype_id = self.CellType2ResourceType[celltype]
                    self.nodes[nodeid].SetResourceType(self.resources[restype_id].name)
            self.num_site = len(self.sites)

    # Read the design.pl file
    def readFixedPl(self):
        with open(self.params["fixed"], "r") as f_fixed:
            all_lines = f_fixed.read().splitlines()
            for id in range(len(all_lines)):
                cur_line = all_lines[id]
                if cur_line[0] == "#":
                    continue
                cur_line_col = cur_line.strip().split()
                nodeid = self.nodeName2Id[cur_line_col[0]]
                self.nodes[nodeid].is_fixed = True
                self.nodes[nodeid].fixed_corr.append(int(cur_line_col[1]))
                self.nodes[nodeid].fixed_corr.append(int(cur_line_col[2]))
                locX = int(cur_line_col[1])
                locY = int(cur_line_col[2])
                bel = int(cur_line_col[3])
                realX = locX
                realY = locY*26/30 + bel
                siteid = self.sitemaps[int(cur_line_col[1])][int(cur_line_col[2])].astype("int")
                self.nodes[nodeid].SetPlaceLocation(locX, locY, realX, realY, siteid) 

    # Read the design.cascade_shape and design.cascade_shape_instances files
    def readCascadeMacros(self):
        with open(self.params["cascade_shape"], "r") as f_casshape:
            all_lines = f_casshape.read().splitlines()
            cur_line_id = 0
            while cur_line_id  < len(all_lines):
                cur_line = all_lines[cur_line_id]
                if len(cur_line) == 0 or ("End" in cur_line or "#" in cur_line):
                    cur_line_id += 1
                    continue
                cur_line_col = cur_line.strip().split()
                # print(cur_line, cur_line_id)
                if cur_line_col[0] == "Shape":
                    if "URAM" in cur_line_col[1]:
                        cur_line_id += (int(cur_line_col[2])+2)
                        continue 
                    cascademacrotype  = CascadeMacroType(cur_line_col[1], len(self.cascademacrotypes), int(cur_line_col[2]), int(cur_line_col[3]))
                    cascademacrotype.getCellType(all_lines[cur_line_id+2].strip().split()[0])
                    cur_line_id += (int(cur_line_col[2]) + 2) # skip the BEGIN and End
                    self.cascademacrotype2Id[cur_line_col[1].upper()] = len(self.cascademacrotypes)
                    self.cascademacrotypes.append(cascademacrotype)
        
        with open(self.params["cascade_instance"], "r") as f_casinst:
            all_lines = f_casinst.read().splitlines()
            cur_line_id = 0
            while cur_line_id < len(all_lines):
                cur_line = all_lines[cur_line_id]
                if len(cur_line) == 0 or ("END" in cur_line or "#" in cur_line):
                    cur_line_id += 1
                    continue
                cur_line_col = cur_line.strip().split()
                # A bug in the file
                if cur_line_col[0] == "BRAM_cascade":
                    cur_line_col[0] = "BRAM_cascade_2"
                if cur_line_col[0].upper() in list(self.cascademacrotype2Id.keys()):
                    cascademacroinst = CascadeMacro(cur_line_col[3], len(self.cascademacros), cur_line_col[0].upper(), int(cur_line_col[1]), int(cur_line_col[2]))
                    sub_cur_line = all_lines[cur_line_id+2]
                    sub_cur_line_col = sub_cur_line.strip().split()
                    if not sub_cur_line_col[0] in list(self.nodeName2Id.keys()):
                        cur_line_id += (int(cur_line_col[1])+2)
                        continue
                    refnode_id = self.nodeName2Id[sub_cur_line_col[0]]
                    self.nodes[refnode_id].is_cascade_refer = True
                    cascademacroinst.SetReferenceNode(refnode_id)
                    # Add the nodes in the cascade macro
                    for subid in range(len(self.nodes)):
                        if cur_line_col[3] in self.nodes[subid].name:
                            self.nodes[subid].cascade_id = cascademacroinst.id
                            cascademacrotypeid = self.cascademacrotype2Id[cascademacroinst.macrotype]
                            if self.nodes[subid].celltype == self.cascademacrotypes[cascademacrotypeid].celltype:
                                cascademacroinst.addNode(self.nodes[subid], is_macro=True)
                            else:
                                cascademacroinst.addNode(self.nodes[subid], is_macro=False)
                    self.cascademacros.append(cascademacroinst)
                cur_line_id += (int(cur_line_col[1])+2)

    # Read the design.regions file
    def readRegionConstraints(self):
        with open(self.params["region_constr"], "r") as f_constr:
            all_lines = f_constr.read().splitlines()
            cur_line_id = 0
            while cur_line_id < len(all_lines):
                cur_line = all_lines[cur_line_id]
                if len(cur_line) == 0 or "#" in cur_line:
                    cur_line_id += 1
                    continue
                cur_line_col = cur_line.strip().split()
                if cur_line_col[0] == "RegionConstraint" and cur_line_col[1] == "BEGIN":
                    constr = RegionConstrType(int(cur_line_col[2]), int(cur_line_col[3]))
                    for subid in range(1, int(cur_line_col[3])+1):
                        sub_cur_line = all_lines[cur_line_id + subid]
                        sub_cur_line_col = sub_cur_line.strip().split()
                        constr.AddBox(int(sub_cur_line_col[1]), int(sub_cur_line_col[2]), int(sub_cur_line_col[3]), int(sub_cur_line_col[4]))
                    cur_line_id += (int(cur_line_col[3])+2)
                    # Add the corresponding site in the constraint region
                    for siteid in range(len(self.sites)):
                        site = self.sites[siteid]
                        Xcorr = site.locX
                        Ycorr = site.locY
                        sitetypename = site.sitetype
                        left_right_boundary_slack = 1
                        if "BRAM" in sitetypename:
                            up_down_boundary_slack = 5
                        elif "DSP" in sitetypename:
                            up_down_boundary_slack = 3
                        else:
                            up_down_boundary_slack = 1
                        # Judge whether the site is added 
                        if(constr.IsinRegion(Xcorr, Ycorr, left_right_boundary_slack, up_down_boundary_slack)):
                            constr.AddSite(site)
                    self.regionconstrtype.append(constr)
                elif cur_line_col[0] == "InstanceToRegionConstraintMapping" and cur_line_col[1] == "BEGIN":
                    cur_line_id += 1
                    while cur_line_id < len(all_lines):
                        cur_line = all_lines[cur_line_id]
                        cur_line_col = cur_line.strip().split()
                        if cur_line_col[0] == "InstanceToRegionConstraintMapping" and cur_line_col[1] == "END":
                            cur_line_id += 1
                            break
                        #For DSP device
                        if "DSP_config" in cur_line_col[0]:
                            sub_cur_line_col = cur_line_col[0].split("/")
                            cur_line_col[0] = sub_cur_line_col[0] + "/" + sub_cur_line_col[1]
                        nodeid = self.nodeName2Id[cur_line_col[0]]
                        self.nodes[nodeid].regionconstr_type = int(cur_line_col[1])
                        #print(self.nodes[nodeid].name, self.nodes[nodeid].resourcetype)
                        self.regionconstrtype[int(cur_line_col[1])].AddNode(self.nodes[nodeid])
                        self.nodes[nodeid].regionconstr.extend(self.regionconstrtype[int(cur_line_col[1])].constrcol)
                        if self.nodes[nodeid].is_macro:
                            self.nodeswithRegionConstr.append(nodeid)
                        cur_line_id += 1
            
            # calculate the area of the constrained region
            for regionconstr in self.regionconstrtype:
                self.regionconstr_area.append(regionconstr.area)

    # Read the solution.pl file
    def readSamplePl(self, solution_path, logger):
        with open(solution_path, "r") as f_samp:
            logger.info("loading the macro placement result from sample.pl")
            all_lines = f_samp.read().splitlines()
            # read all the locations for the nodes in the solution.pl file
            for id in range(len(all_lines)):
                cur_line = all_lines[id]
                cur_line_col = cur_line.strip().split()
                if cur_line_col[0] in list(self.nodeName2Id.keys()):
                    nodeid = self.nodeName2Id[cur_line_col[0]]
                    locX = int(cur_line_col[1])
                    locY = int(cur_line_col[2])
                    siteid = self.sitemaps[locX][locY].astype("int")
                    locX, locY, realX, realY = self.sites[siteid].getLocation()
                    self.nodes[nodeid].SetPlaceLocation(locX, locY, realX, realY, siteid)
                    self.sites[siteid].addNode(self.nodes[nodeid])
                else:
                    logger.info(cur_line_col[0]+" is not in the netlist!! Error!")
                
            #For other cells in the cascaded macro, calculate the coordinate of these cells
            for id in range(len(self.nodes)):
                if self.nodes[id].is_cascade_refer:
                    locX_refer, locY_refer, _, _ = self.nodes[id].getLocation()
                    site_refer_id = self.nodes[id].getlocatedsiteid()
                    cascade_id = self.nodes[id].cascade_id
                    self.cascademacros[cascade_id].SetCascadeMacroLoc(locX_refer, locY_refer)
                    cascademacro_inst = self.cascademacros[cascade_id]
                    for subid in range(0,len(cascademacro_inst.Macronodecol)):
                        nodeid = cascademacro_inst.Macronodecol[subid].id
                        site_id = site_refer_id + subid
                        site_locX, site_locY, real_locX, real_locY = self.sites[site_id].locX
                        if not self.nodes.isPlace:
                            self.nodes[nodeid].SetPlaceLocation(site_locX, site_locY, real_locX, real_locY, site_id)
                            self.cascademacros[cascade_id].Macronodecol[subid].SetPlaceLocation(site_locX, site_locY, real_locX, real_locY, site_id)
                            self.sites[site_id].addNode(self.nodes[nodeid])
                        else:
                            node = self.nodes[nodeid]
                            self.cascademacros[cascade_id].Macronodecol[subid].SetPlaceLocation(node.site_locX, node.site_locY, node.real_locX, node.real_locY, node.sitr_id)

    # Check whether the regions conatining (X, Y) is overflow
    def checkRegionFull(self, restype, Xcorr, Ycorr, left_right_boundary_slack, up_down_boundary_slack):
        for id in range(len(self.regionconstrtype)):
            region = self.regionconstrtype[id]
            if region.IsinRegion(Xcorr, Ycorr, left_right_boundary_slack, up_down_boundary_slack) and \
            region.CheckIsFull(restype):
                return True
        return False
    
    # Check whether the coordinate (X,Y) is in the constrained region
    def checkIsinRegion(self, region, Xcorr, Ycorr):
        if region.IsinRegion(Xcorr, Ycorr, 0, 0):
            return True
        return False
    
    # Randomly generate the macro coordinates
    def RandomCordGenerate(self, logger):
        logger.info("randomly generated macro placement results")
        # Record the location containing in each type of resource
        restype_loc = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "IO":[]}
        
        left_right_region_slack = 1.0 # The slack distance from the region left/right boundary
        bram_up_down_region_slack = 5.0 # The slack distance from the region up/down boundary for bram
        dsp_up_down_region_slack = 3.0 # The slack distance from the region up/down boundary for dsp

        # Find the sites belonging to each resource category
        for i in range(len(self.sites)):
            resource_supply = list(self.sites[i].resource_supply.keys())
            for j in range(len(resource_supply)):
                res_name = list(resource_supply)[j]
                if self.sites[i].resource_supply[res_name] > 0:
                    restype_loc[res_name].append(i)
        
        # Radnomly legalize the cascade macro at first
        for id in range(len(self.cascademacros)):
            cascademacro = self.cascademacros[id]
            nodecol = cascademacro.Macronodecol
            macroheight = cascademacro.height
            macronumcells = cascademacro.num_cells
            noderefer = nodecol[0]
            node_restype = noderefer.resourcetype
            # Determine the slack from the region
            if node_restype == "RAMB36E2":
                up_down_region_slack = bram_up_down_region_slack
            elif node_restype == "DSP48E2":
                up_down_region_slack = dsp_up_down_region_slack
            flag = False
            # Find the legal locations for all the cascade macros
            while not flag:
                candidate = restype_loc[node_restype]
                place_site_id = choice(candidate)
                X_refer = self.sites[place_site_id].locX
                # If the upper boundary of the cascade macro is above the device height
                if X_refer + macroheight > self.sitemap_height:
                    candidate.remove(place_site_id)
                else:
                    flag = True
                    # If there are some overlaps fot the macros and the constrained region overflow
                    for j in range(1, macronumcells):
                        immed_site_id = place_site_id+j
                        X_immed, Y_immed, _, _ = self.sites[immed_site_id].getLocation()
                        if self.sites[immed_site_id].CheckIsFull(noderefer.resourcetype) or \
                        self.checkRegionFull(node_restype, X_immed, Y_immed, left_right_region_slack, up_down_region_slack):
                            flag = False
                            break

                    # place macro in the legal position and update
                    if flag:
                        X_macro, Y_macro, realX_macro, realY_macro = self.sites[place_site_id].getLocation()
                        self.cascademacros[id].SetCascadeMacroLoc(X_macro, Y_macro)
                        for j in range(0, macronumcells):
                            nodeid = nodecol[j].id
                            immed_site_id = place_site_id+j
                            immed_site_X, immed_site_Y, immed_realX, immed_realY = self.sites[immed_site_id].getLocation()
                            self.nodes[nodeid].SetPlaceLocation(immed_site_X, immed_site_Y, immed_realX, immed_realY, immed_site_id)
                            self.cascademacros[id].Macronodecol[j].SetPlaceLocation(immed_site_X, immed_site_Y, immed_realX, immed_realY, immed_site_id)
                            self.sites[immed_site_id].addNode(self.nodes[nodeid])
                            for regionid in range(len(self.regionconstrtype)):
                                region = self.regionconstrtype[regionid]
                                if(region.IsinRegion(immed_site_X, immed_site_Y, left_right_region_slack, up_down_region_slack)):
                                    self.regionconstrtype[regionid].AddNode(self.nodes[nodeid])
                            if immed_site_id in restype_loc[noderefer.resourcetype]:
                                restype_loc[noderefer.resourcetype].remove(immed_site_id)
                    else:
                        candidate.remove(place_site_id)                                                       

        # Place the nonmacrocascade nodes
        for id in range(len(self.nodeswithRegionConstr)):
            nodeid = self.nodeswithRegionConstr[id]
            candidate = restype_loc[self.nodes[nodeid].resourcetype]
            candidate_in_RegionConstr = []
            for siteid in candidate:
                site = self.sites[siteid]
                locX, locY, realX, realY = site.getLocation()
                self.nodes[nodeid].SetPlaceLocation(locX, locY, realX, realY, siteid)
                if self.nodes[nodeid].IsBRAM():
                    up_down_region_slack = bram_up_down_region_slack
                if self.nodes[nodeid].IsDSP():
                    up_down_region_slack = dsp_up_down_region_slack  
                if self.nodes[nodeid].IsinRegionConstr(left_right_region_slack, up_down_region_slack):
                    candidate_in_RegionConstr.append(siteid)
                self.nodes[nodeid].ResetPlaceLocation()
            if len(candidate_in_RegionConstr) == 0:
                logger.info("The Nonmacro Node with regional constraints could not find feasible place!!")
            
            flag = False
            if len(candidate_in_RegionConstr) > 0:
                while not flag:
                    place_site_id = choice(candidate_in_RegionConstr)
                    if not self.sites[place_site_id].CheckIsFull(self.nodes[nodeid].resourcetype):
                        locX, locY, realX, realY = self.sites[place_site_id].getLocation()
                        self.nodes[nodeid].SetPlaceLocation(locX,locY,realX,realY,place_site_id)
                        self.sites[place_site_id].addNode(self.nodes[nodeid])
                        restype_loc[self.nodes[nodeid].resourcetype].remove(place_site_id)
                        flag = True

        # Nonmacrocascade nodes with no regional constraints 
        for id in range(len(self.nodes)):
            node = self.nodes[id]
            if node.is_macro and (node.cascade_id == -1 and node.regionconstr_type == -1):
                flag = False
                while not flag:
                    candidate = restype_loc[node.resourcetype]
                    place_site_id = choice(candidate)
                    if not self.sites[place_site_id].CheckIsFull(node.resourcetype):
                        locX, locY, realX, realY = self.sites[place_site_id].getLocation()
                        self.nodes[id].SetPlaceLocation(locX, locY, realX, realY, place_site_id)
                        self.sites[place_site_id].addNode(self.nodes[id])
                        restype_loc[node.resourcetype].remove(place_site_id)
                        flag = True
   
   # calculate the macro HPWL
    def calMacroHPWL(self):
        totalMacroHPWL = 0
        for id in range(len(self.nets)):
            net_inst = self.nets[id]
            min_X = 100000
            min_Y = 100000
            max_X = -100000
            max_Y = -100000
            for pinid in range(len(net_inst.pins)):
                nodeid = net_inst.pins[pinid][0]
                node = self.nodes[nodeid]
                X_corr, Y_corr, _, _ = node.getLocation()
                if node.is_macro or node.is_fixed:
                    if X_corr < min_X:
                        min_X = X_corr
                    if X_corr > max_X:
                        max_X = X_corr
                    if Y_corr < min_Y:
                        min_Y = Y_corr
                    if Y_corr > max_Y:
                        max_Y = Y_corr
            if min_X == 100000:
                macroHPWL = 0
            else:
                macroHPWL = (max_X - min_X) + (max_Y - min_Y)
            totalMacroHPWL += macroHPWL
        return totalMacroHPWL

    # output the macro placement results
    def OutputSolutionpl(self, output_path):
        with open(output_path, "w") as f_sol:
            output_str = ""
            for id in range(len(self.nodes)):
                if self.nodes[id].is_macro:
                    output_str += self.nodes[id].name
                    output_str += " "
                    output_str += str(self.nodes[id].locX)
                    output_str += " "
                    output_str += str(self.nodes[id].locY)
                    output_str += " "
                    output_str += "0\n"
            f_sol.write(output_str)

    # read all the files in testcase
    def readAll(self, logger):
        logger.info("loading cell library")
        self.readCellLibs()
        logger.info("Number of cells in the library:"+str(len(self.cellLib)))
        logger.info("loading netlist")
        self.readNodes()
        logger.info("Number of nodes:"+str(len(self.nodes)))
        self.readNets()
        logger.info("Number of nets:"+str(len(self.nets)))
        logger.info("loading sitemap")
        self.readSiteMaps()
        logger.info("Size of the site map:("+str(self.sitemap_width)+","+str(self.sitemap_height)+")")
        logger.info("Available Site:"+str(self.num_site))
        logger.info("loading fixed position")
        self.readFixedPl()
        logger.info("loading cascaded macro info")
        self.readCascadeMacros()
        logger.info("loading region constraints")
        self.readRegionConstraints()

        logger.info("====Statistics====")
        self.num_cells = len(self.cellLib)
        self.num_nodes = len(self.nodes)
        self.num_nets = len(self.nets)
        self.num_clk_nets = len(self.clknets)
        self.num_high_degree_nets = len(self.highdegreenets)
        self.num_region_constr = len(self.regionconstrtype)
        self.num_cascade_macro = len(self.cascademacros)

        for id in range(len(self.nodes)):
            nodeper = self.nodes[id]
            if nodeper.resourcetype is not None:
                self.num_resource_demand[nodeper.resourcetype] += 1
            if nodeper.is_fixed:
                self.num_fix += 1
            if nodeper.is_macro and nodeper.cascade_id == -1:
                self.num_basic_macro += 1
            if nodeper.is_macro and nodeper.cascade_id != -1:
                self.num_cascade_node += 1
            if nodeper.regionconstr_type != -1:
                self.num_region_constr_node += 1
                if nodeper.is_macro:
                    self.num_region_constr_macronode += 1
                    if nodeper.cascade_id != -1:
                        self.num_region_constr_cascademacronode += 1
            
        self.num_macro = self.num_basic_macro + self.num_cascade_macro
        if self.num_region_constr > 0:
            self.min_region_constr = min(self.regionconstr_area)
        else:
            self.min_region_constr = 0
        if self.num_region_constr > 0:
            self.max_region_constr = max(self.regionconstr_area)
        else:
            self.max_region_constr = 0
        if self.num_region_constr > 0:
            self.avg_region_constr = sum(self.regionconstr_area)*1.0/len(self.regionconstr_area)
        else:
            self.avg_region_constr = 0          

        logger.info("Number of clock nets:"+str(self.num_clk_nets))
        logger.info("Number of high-fanout nets:"+str(self.num_high_degree_nets))        
        logger.info("Number of fix nodes:"+str(self.num_fix))
        logger.info("Number of macros:"+str(self.num_macro))
        logger.info("Number of basic macros:"+str(self.num_basic_macro)+",cascade macros:"+str(self.num_cascade_macro))
        logger.info("Number of cascade macro nodes:"+str(self.num_cascade_node))
        logger.info("Region Constraints:"+str(self.num_region_constr))
        logger.info("Min Region Constraint Area:"+str(self.min_region_constr))
        logger.info("Max Region Constraint Area:"+str(self.max_region_constr))
        logger.info("Avg Region Constraint Area:"+str(self.avg_region_constr))       
        logger.info("Region Constraint Node:"+str(self.num_region_constr_node))
        logger.info("Region Constraint Macro Node:"+str(self.num_region_constr_macronode))
        logger.info("Region Constraint Cascade Node:"+str(self.num_region_constr_cascademacronode))
        str_out = "Resource for the nodes in the circuit:"
        for res_id, res_name in enumerate(list(self.num_resource_demand.keys())):
            str_out = str_out + res_name + ":"+str(self.num_resource_demand[res_name])+" "
        logger.info(str_out)
        str_out = "Supply Resource in the FPGA:"
        for res_id, res_name in enumerate(list(self.num_resource_supply.keys())):
            str_out = str_out + res_name + ":"+str(self.num_resource_supply[res_name])+" "
        logger.info(str_out)


# load the benchamark
def load_dataset(args, logger, placement=None):
    if args.custom_path != "":
        params = get_custom_design_params(args)
    else:
        params = get_single_design_params(
            args.dataset_root, args.dataset, args.design_name, placement
        )
    
    dataset = Dataset(params)
    if checkparam(params, logger):
        logger.info("loading from original benchmark...")
        dataset.readAll(logger)
    
    if args.feature_extract:
        logger.info("=======Initial feature construction=======")
        logger.info("Generate the netlist feature for:"+args.design_name)
        placementinfo = PlacementInfo(dataset)
        placementinfo.ConvertDB2PlacementInfo(logger)
        feature_extractor = FeatureExtractor(placementinfo)
        if args.output_dir == "":
            log_dir = os.path.join(args.result_dir, args.exp_id, args.log_dir, args.design_name)
            output_path = os.path.join(log_dir, "netlist_feature")
        else:
            output_path = os.path.join(args.output_dir, args.design_name, "netlist_feature")
        feature_extractor.OutputPUGraph(output_path)

    return dataset
