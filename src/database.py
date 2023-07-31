import collections
import copy
import math
from utils import *
from src.db import *
import os
from random import choice

class Dataset:
    def __init__(self, params):
        self.params = params #The path to each file
        self.cellLib = {} #cellLib, the dict variable
        self.nodeNameIdMap = {} #Find the id using node name
        self.netNameIdMap = {} #Find the id using net name
        self.ResourceIdMap = {} #Find the id using resource name
        self.ResourceCellMap = {} #Find the resource types corresponding to the cell types 
        self.sitetypeIdMap = {} #Find the id using resource name
        self.macrotypeIdMap = {} #Find the id using the macro type name
        self.sitecolumns = {} #The site columns for each

        self.nodes = [] #nodes col, the list variable
        self.macronodeswithRegionConstr = [] #nodes col with regional constraint
        self.nets = [] #nets col, the list variable
        self.sites = [] #The sites that could place the nodes

        self.resources = [] #resource type col {LUT,FF,CARRY8,DSP48E2,RAMB36E2,URAM288,IO}
        self.sitetypes = [] #site type col {SLICE, DSP, BRAM, URAM, IO}
        self.macrotypes = [] #macro type col
        self.cascademacros = [] #All the cascaded macros
        self.regionconstrtype = [] #Region Constraint Type

        #Basic information
        self.num_cells = 0
        self.num_nodes = 0
        self.num_nets = 0
        self.num_resource_demand = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "URAM288":0, "IO":0}
        self.num_resource_supply = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "URAM288":0, "IO":0}
        self.num_bel = 0
        
        #Macro number statistics
        self.num_basic_macro = 0
        self.num_cascade_macro = 0
        self.num_cascade_node = 0
        self.num_macro = 0

        #SiteMap
        self.sitemap_width = 0
        self.sitemap_height = 0
        self.num_avail_site = 0
        self.num_avail_site_dict = {"SLICE":0, "DSP":0, "BRAM":0, "URAM":0, "IO":0}

        #fix node and regional constraints
        self.num_fix = 0
        self.num_region_constr = 0
        self.num_region_constr_node = 0
        self.num_region_constr_maceonode = 0
        self.num_region_constr_cascademaceonode = 0
    
    def readNodes(self):
        if os.path.exists(self.params["nodes"]):
            with open(self.params["nodes"],"r") as f_node:
                all_lines = f_node.read().splitlines()
                cell_id = 0
                for id in range(len(all_lines)):
                    cur_line = all_lines[id]
                    new_node_info = cur_line.split()
                    if new_node_info[0][0] == "#":
                        continue
                    new_node = Node(new_node_info[0], cell_id, new_node_info[1])
                    if new_node_info[1] in list(self.cellLib.keys()):
                        new_node.addPin(self.cellLib[new_node_info[1]].pins)
                    self.nodes.append(new_node)
                    self.nodeNameIdMap[new_node_info[0]] = cell_id
                    cell_id = cell_id + 1    
    
    def readNets(self):
        if os.path.exists(self.params["nets"]):
            with open(self.params["nets"], "r") as f_net:
                all_lines = f_net.read().splitlines()
                id = 0
                netid = 0
                while id < len(all_lines):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    if cur_line_col[0][0] == "#":
                        id = id + 1
                        continue
                    if cur_line_col[0] == "net":
                        new_net = Net(cur_line_col[1], netid)
                        while id < len(all_lines):
                            id = id + 1
                            cur_line = all_lines[id]
                            new_pin_info = cur_line.strip().split()
                            if new_pin_info[0] == "endnet":
                                break
                            nodename = new_pin_info[0]
                            pinname = new_pin_info[1]
                            self.nodes[self.nodeNameIdMap[nodename]].addNeighboringNets(netid)
                            if all_lines[id+1] == "endnet":
                                self.nodes[self.nodeNameIdMap[nodename]].addNeighboringOutNets(netid)
                            else:
                                self.nodes[self.nodeNameIdMap[nodename]].addNeighboringInNets(netid)                       
                            nodeid = self.nodeNameIdMap[nodename]
                            celltype = self.nodes[nodeid].celltype
                            if celltype in list(self.cellLib.keys()):
                                if not pinname in list(self.cellLib[celltype].pinNameIdMap):
                                    #self.cellLib[celltype].pinNameIdMap[pinname] = len(self.cellLib[celltype].pins)
                                    if "IN" in pinname:
                                        pin_IO = 1
                                    else:
                                        pin_IO = 0
                                    self.cellLib[celltype].addPin(pinname, pin_IO, False, False)
                                    self.nodes[nodeid].addPin([self.cellLib[celltype].pins[-1]])
                                pinId = self.cellLib[celltype].pinNameIdMap[pinname]
                                pin = self.cellLib[celltype].pins[pinId]
                                new_net.addPin([nodeid, pin])
                                if self.nodes[nodeid].is_macro:
                                    new_net.addMacroNodeAdj(nodeid)
                        self.nets.append(new_net)
                        self.netNameIdMap[cur_line_col[1]] = netid
                        id = id + 1
                        netid = netid + 1
                        
    def readCellLibs(self):
        if os.path.exists(self.params["lib"]):
            with open(self.params["lib"],"r") as f_lib:
                all_lines = f_lib.read().splitlines()
                cell_id = 0
                cell_name = ""
                for id in range(len(all_lines)):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    if len(cur_line_col) == 0:
                        continue
                    if cur_line_col[0][0] == "#" or "END" in cur_line:
                        continue
                    if "CELL" in cur_line:
                        cell_name = cur_line.split()[1]
                        new_cell = Cell(cell_name, cell_id)
                        self.cellLib[cell_name] = new_cell
                        cell_id = cell_id + 1
                    elif "PIN" in cur_line:
                        new_pin_info = cur_line.strip().split()
                        if new_pin_info[2] == "INPUT":
                            pin_IO = 1
                        else:
                            pin_IO = 0                        
                        pin_clock = False
                        pin_ctrl = False
                        if len(new_pin_info) > 3:
                            if new_pin_info[3] == "CLOCK":
                                pin_clock = True
                            if new_pin_info[3] == "CTRL":
                                pin_ctrl = True
                        self.cellLib[cell_name].addPin(new_pin_info[1], pin_IO, pin_clock, pin_ctrl)
    
    def readSiteMaps(self):
        if os.path.exists(self.params["sitemap"]):
            with open(self.params["sitemap"],"r") as f_scl:
                all_lines = f_scl.read().splitlines()
                site_type_id = 0
                resource_type_id = 0
                site_id = 0
                id = 0
                while id < len(all_lines):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    id = id + 1
                    if "END" in cur_line_col or len(cur_line_col)==0:
                        continue
                    if "SITE" in cur_line_col:
                        new_sitetype = SiteType(cur_line_col[1], site_type_id)
                        while id < len(all_lines):
                            sub_cur_line = all_lines[id]
                            sub_cur_line_col = sub_cur_line.strip().split()
                            if sub_cur_line_col[0] == "END" and sub_cur_line_col[1] == "SITE":
                                break
                            if sub_cur_line_col[0] == "URAM288":
                                new_sitetype.AddResourceMulti(sub_cur_line_col[0], 4)
                            else:
                                new_sitetype.AddResourceMulti(sub_cur_line_col[0], int(sub_cur_line_col[1]))
                            id = id + 1
                        self.sitetypes.append(new_sitetype)
                        self.sitetypeIdMap[cur_line_col[1]] = new_sitetype
                        site_type_id = site_type_id + 1
                    elif "RESOURCES" in cur_line_col:
                        while id < len(all_lines):
                            cur_line = all_lines[id]
                            cur_line_col = cur_line.strip().split()
                            if cur_line_col[0] == "END" and cur_line_col[1] == "RESOURCES":
                                break
                            new_resource = Resource(cur_line_col[0], resource_type_id)
                            for subid in range(1,len(cur_line_col)):
                                new_resource.AddCelltype(cur_line_col[subid])
                                self.ResourceCellMap[cur_line_col[subid]] = resource_type_id
                            if cur_line_col[0] == "FF":
                                self.ResourceCellMap["FDSE"] = resource_type_id
                            id = id + 1
                            self.resources.append(new_resource)
                            self.ResourceIdMap[resource_type_id] = new_resource
                            resource_type_id = resource_type_id + 1
                    elif "SITEMAP" in cur_line_col:
                        self.sitemap_width = int(cur_line_col[1])
                        self.sitemap_height = int(cur_line_col[2])
                        self.sitemaps = np.ones([self.sitemap_width, self.sitemap_height])*(-1) #2d site maps with the 1d site value, -1 refers to the sites that could not be placed
                        self.sitemap_res = np.ones([self.sitemap_width, self.sitemap_height])*(-1) #2d site maps with the site category, -1 refers to the sites that could not be placed
                        while id < len(all_lines):
                            cur_line = all_lines[id]
                            cur_line_col = cur_line.strip().split()
                            if cur_line_col[0] == "END" and cur_line_col[1] == "SITEMAP":
                                break
                            new_site = Site(cur_line_col[2], site_id, int(cur_line_col[0]), int(cur_line_col[1]))
                            sitetype_id = self.sitetypeIdMap[cur_line_col[2]].id
                            new_site.addSupplyResource(self.sitetypes[sitetype_id].resourcecap)
                            self.sites.append(new_site)
                            self.sitemaps[int(cur_line_col[0])][int(cur_line_col[1])] = site_id
                            self.sitemap_res[int(cur_line_col[0])][int(cur_line_col[1])] = self.sitetypeIdMap[cur_line_col[2]].id
                            for res_id, res_name in enumerate(list(self.sitetypes[sitetype_id].resourcecap.keys())):     
                                self.num_resource_supply[res_name] += self.sitetypes[sitetype_id].resourcecap[res_name] 
                                self.num_bel += 1
                            id = id + 1
                            site_id = site_id + 1
                            self.num_avail_site_dict[cur_line_col[2]] += 1
                self.num_avail_site = site_id

                for nodeid in range(len(self.nodes)):
                    celltype = self.nodes[nodeid].celltype
                    if celltype in list(self.ResourceCellMap.keys()):
                        restype_id = self.ResourceCellMap[celltype]
                        self.nodes[nodeid].SetResourceType(self.resources[restype_id].name)

    def readFixedPl(self):
        if os.path.exists(self.params["fixed"]):
            with open(self.params["fixed"], "r") as f_fixed:
                all_lines = f_fixed.read().splitlines()
                for id in range(len(all_lines)):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    if cur_line_col[0][0] == "#":
                        continue
                    nodeid = self.nodeNameIdMap[cur_line_col[0]]
                    self.nodes[nodeid].is_fixed = True
                    self.nodes[nodeid].fixed_corr.append(int(cur_line_col[1]))
                    self.nodes[nodeid].fixed_corr.append(int(cur_line_col[2]))
                    siteid = self.sitemaps[int(cur_line_col[1])][int(cur_line_col[2])]
                    self.nodes[nodeid].SetPlaceLocation(int(cur_line_col[1]),int(cur_line_col[2]),siteid) 
    
    def readCascadeMacros(self):
        if os.path.exists(self.params["cascade_shape"]):
            with open(self.params["cascade_shape"], "r") as f_casshape:
                all_lines = f_casshape.read().splitlines()
                macrotype_id = 0
                id = 0
                while id  < len(all_lines):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    id = id + 1
                    if cur_line_col[0] == "Shape":
                        macrotype  = CascadeMacroType(cur_line_col[1], macrotype_id, int(cur_line_col[2]), int(cur_line_col[3]))
                        id = id + 1
                        for subid in range(int(cur_line_col[2])):
                            sub_cur_line = all_lines[id]
                            if subid == 0:
                                macrotype.getCellType(sub_cur_line.strip().split()[0])
                            id = id + 1
                        id = id + 2
                        self.macrotypeIdMap[cur_line_col[1].upper()] = macrotype_id
                        self.macrotypes.append(macrotype)
                        macrotype_id = macrotype_id + 1
        
        if os.path.exists(self.params["cascade_instance"]):
            with open(self.params["cascade_instance"], "r") as f_casinst:
                all_lines = f_casinst.read().splitlines()
                macroinst_id = 0
                id = 0
                while id < len(all_lines):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    id = id + 1
                    if len(cur_line_col) == 0:
                        continue
                    macroinst = None
                    macroinst_sub = None
                    if cur_line_col[0].upper() == "BRAM_CASCADE":
                        cur_line_col[0] = "BRAM_CASCADE_2"
                    if cur_line_col[0].upper() in list(self.macrotypeIdMap.keys()):
                        # For the macro URAM, you have to split it into two cascaded macros
                        if "URAM_CASCADE_8x2" in cur_line_col[3]:
                            macroinst = CascadeMacro(cur_line_col[3], macroinst_id, cur_line_col[0].upper(), int(cur_line_col[1]), int(cur_line_col[2])/2)
                            macroinst_id = macroinst_id + 1
                            macro_name = cur_line_col[3] + "/URAM_cascade_sub_instance"
                            macroinst_sub = CascadeMacro(macro_name, macroinst_id, cur_line_col[0].upper(), int(cur_line_col[1]), int(cur_line_col[2])/2)
                        else:
                            macroinst = CascadeMacro(cur_line_col[3], macroinst_id, cur_line_col[0].upper(), int(cur_line_col[1]), int(cur_line_col[2]))

                        id = id + 1
                        sub_cur_line = all_lines[id]
                        sub_cur_line_col = sub_cur_line.strip().split()
                        nodeid = self.nodeNameIdMap[sub_cur_line_col[0]]
                        self.nodes[nodeid].is_cascade_refer = True
                        macroinst.SetReferenceNode(nodeid)
                        id = id + 1
                        #print(cur_line_col[3])
                        for subid in range(len(self.nodes)):
                            if cur_line_col[3] in self.nodes[subid].name:
                                if "URAM_cascade_sub_instance" in self.nodes[subid].name:
                                    self.nodes[subid].cascade_id = macroinst_sub.id
                                    if self.nodes[subid].celltype == self.macrotypes[self.macrotypeIdMap[macroinst_sub.macrotype]].celltype:
                                        macroinst_sub.addNode(self.nodes[subid], is_macro=True)
                                    else:
                                        macroinst_sub.addNode(self.nodes[subid], is_macro=False)
                                    if "URAM288_inst9" in self.nodes[subid].name:
                                        self.nodes[subid].is_cascade_refer = True
                                        macroinst_sub.SetReferenceNode(self.nodes[subid].id)
                                else:
                                    self.nodes[subid].cascade_id = macroinst.id
                                    if self.nodes[subid].celltype == self.macrotypes[self.macrotypeIdMap[macroinst.macrotype]].celltype:
                                        macroinst.addNode(self.nodes[subid], is_macro=True)
                                    else:
                                        macroinst.addNode(self.nodes[subid], is_macro=False)
                        self.cascademacros.append(macroinst)
                        if "URAM_CASCADE_8x2" in cur_line_col[3]:
                            self.cascademacros.append(macroinst_sub)
                        macroinst_id = macroinst_id + 1
                    else:
                        continue


    def readRegionConstraints(self):
        if os.path.exists(self.params["region_constr"]):
            with open(self.params["region_constr"], "r") as f_constr:
                all_lines = f_constr.read().splitlines()
                id = 0
                while id < len(all_lines):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    #print(cur_line_col)
                    id = id + 1
                    if len(cur_line_col) == 0:
                        continue
                    if cur_line_col[0] == "RegionConstraint" and cur_line_col[1] == "BEGIN":
                        new_constr = RegionConstrType(int(cur_line_col[2]), int(cur_line_col[3]))
                        for subid in range(int(cur_line_col[3])):
                            sub_cur_line = all_lines[id]
                            sub_cur_line_col = sub_cur_line.strip().split()
                            new_constr.AddBox(int(sub_cur_line_col[1]), int(sub_cur_line_col[2]), int(sub_cur_line_col[3]), int(sub_cur_line_col[4]))
                            id = id + 1
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
                            if(new_constr.IsinRegion(Xcorr, Ycorr, left_right_boundary_slack, up_down_boundary_slack)):
                                new_constr.AddSite(site)
                        self.regionconstrtype.append(new_constr)
                        id = id + 1
                    elif cur_line_col[0] == "InstanceToRegionConstraintMapping" and cur_line_col[1] == "BEGIN":
                        while id < len(all_lines):
                            cur_line = all_lines[id]
                            cur_line_col = cur_line.strip().split()
                            if cur_line_col[0] == "InstanceToRegionConstraintMapping" and cur_line_col[1] == "END":
                                break
                            #For DSP device
                            if "DSP_config" in cur_line_col[0]:
                                sub_cur_line_col = cur_line_col[0].split("/")
                                cur_line_col[0] = sub_cur_line_col[0] + "/" + sub_cur_line_col[1]
                            nodeid = self.nodeNameIdMap[cur_line_col[0]]
                            self.nodes[nodeid].regionconstr_type = int(cur_line_col[1])
                            #print(self.nodes[nodeid].name, self.nodes[nodeid].resourcetype)
                            self.regionconstrtype[int(cur_line_col[1])].AddNode(self.nodes[nodeid])
                            self.nodes[nodeid].regionconstr.extend(self.regionconstrtype[int(cur_line_col[1])].constrcol)
                            if self.nodes[nodeid].is_macro:
                                self.macronodeswithRegionConstr.append(nodeid)
                            id = id + 1

    def readSamplePl(self, solution_path, logger):
        if os.path.join(solution_path):
            with open(solution_path, "r") as f_samp:
                logger.info("loading the macro placement result from sample.pl")
                all_lines = f_samp.read().splitlines()
                for id in range(len(all_lines)):
                    cur_line = all_lines[id]
                    cur_line_col = cur_line.strip().split()
                    if cur_line_col[0] in list(self.nodeNameIdMap.keys()):
                        nodeid = self.nodeNameIdMap[cur_line_col[0]]
                        locX = int(cur_line_col[1])
                        locY = int(cur_line_col[2])
                        site = self.sitemaps[locX][locY].astype("int")
                        self.nodes[nodeid].SetPlaceLocation(locX,locY,site)
                        self.sites[site].addNode(self.nodes[nodeid])
                    else:
                        logger.info(cur_line_col[0]+" is not in the netlist!! Error!")
                
                #For other cells in the cascaded macro
                for id in range(len(self.nodes)):
                    if self.nodes[id].is_cascade_refer:
                        uram_cnt = 0 #use for uram_cascade
                        locX_refer = self.nodes[id].locX
                        locY_refer = self.nodes[id].locY
                        site_id = self.sitemaps[locX_refer][locY_refer].astype("int")
                        cascade_id = self.nodes[id].cascade_id
                        macro_inst = self.cascademacros[cascade_id]
                        for subid in range(1,len(macro_inst.Macronodecol)):
                            nodeid = macro_inst.Macronodecol[subid].id
                            if self.nodes[nodeid].resourcetype == "URAM288":
                                if subid % 4 == 0:
                                    uram_cnt += 1
                                new_site_id = site_id + uram_cnt
                            else:
                                new_site_id = site_id + subid
                            new_site_locX = self.sites[new_site_id].locX
                            new_site_locY = self.sites[new_site_id].locY
                            placeflag = self.nodes[nodeid].isPlace
                            self.nodes[nodeid].SetPlaceLocation(new_site_locX, new_site_locY, new_site_id)
                            self.cascademacros[cascade_id].Macronodecol[subid].SetPlaceLocation(new_site_locX, new_site_locY, new_site_id)
                            if not placeflag:
                                self.sites[new_site_id].addNode(self.nodes[nodeid])


    def RandomCordGenerate(self, logger):
        logger.info("generated macro placement results")
        restype_loc = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "URAM288":[], "IO":[]}
        
        left_right_region_slack = 1.0
        bram_up_down_region_slack = 5.0 # The slack distance from the region up/down boundary for bram
        dsp_up_down_region_slack = 3.0 # The slack distance from the region up/down boundary for dsp

        for id in range(len(self.sites)):
            for j in range(len(list(self.sites[id].resource_supply.keys()))):
                res_name = list(self.sites[id].resource_supply.keys())[j]
                if self.sites[id].resource_supply[res_name] > 0:
                    restype_loc[res_name].append(id)
        
        # Place cascade macro at first
        for id in range(len(self.cascademacros)):
            macro = self.cascademacros[id]
            nodecol = macro.Macronodecol
            macrolength = len(nodecol)
            reference_node = nodecol[0]
            left_right_region_slack = 1
            if reference_node.resourcetype == "RAMB36E2":
                up_down_region_slack = 5
            elif reference_node.resourcetype == "DSP48E2":
                up_down_region_slack = 3
            flag = False
            while not flag:
                uram_cnt = 0
                candidate = restype_loc[reference_node.resourcetype]
                place_site_id = choice(candidate)
                X_ref = self.sites[place_site_id].locX
                Y_ref = self.sites[place_site_id].locY
                if place_site_id+macrolength>=len(self.sites):
                    candidate.remove(place_site_id)
                else:
                    subflag = True
                    for j in range(1, macrolength):
                        if reference_node.resourcetype == "URAM288":
                            if j!=0 and j % 4 == 0:
                                uram_cnt += 1
                            immed_site_id = place_site_id+uram_cnt
                        else:
                            immed_site_id = place_site_id+j
                        X_immed = self.sites[immed_site_id].locX
                        Y_immed = self.sites[immed_site_id].locY
                        if self.sites[immed_site_id].CheckIsFull(reference_node.resourcetype):
                            subflag = False
                            break
                        if X_immed != X_ref:
                            subflag = False
                            break
                        if self.checkRegionFull(reference_node.resourcetype, X_immed, Y_immed, left_right_region_slack, up_down_region_slack):
                            subflag = False
                            break

                    if subflag:
                        uram_cnt = 0
                        for j in range(0, macrolength):
                            nodeid = nodecol[j].id
                            if reference_node.resourcetype == "URAM288":
                                if j!=0 and j % 4 == 0:
                                    uram_cnt += 1
                                immed_site_id = place_site_id+uram_cnt
                            else:
                                immed_site_id = place_site_id+j
                            immed_site_X = self.sites[immed_site_id].locX
                            immed_site_Y = self.sites[immed_site_id].locY
                            self.nodes[nodeid].SetPlaceLocation(immed_site_X, immed_site_Y, immed_site_id)
                            self.cascademacros[id].Macronodecol[j].SetPlaceLocation(immed_site_X, immed_site_Y, immed_site_id)
                            self.sites[immed_site_id].addNode(self.nodes[nodeid])
                            for regionid in range(len(self.regionconstrtype)):
                                region = self.regionconstrtype[regionid]
                                if(region.IsinRegion(immed_site_X, immed_site_Y, left_right_region_slack, up_down_region_slack)):
                                    self.regionconstrtype[regionid].AddNode(self.nodes[nodeid])
                            if immed_site_id in restype_loc[reference_node.resourcetype]:
                                restype_loc[reference_node.resourcetype].remove(immed_site_id)
                    else:
                        candidate.remove(place_site_id)

                    flag = subflag                                                       

        # Place the nonmacrocascade nodes
        for id in range(len(self.macronodeswithRegionConstr)):
            nodeid = self.macronodeswithRegionConstr[id]
            candidate = restype_loc[self.nodes[nodeid].resourcetype]
            candidate_in_RegionConstr = []
            for siteid in candidate:
                site = self.sites[siteid]
                locX = site.locX
                locY = site.locY
                self.nodes[nodeid].SetPlaceLocation(locX, locY, siteid)
                if self.nodes[nodeid].IsBRAM():
                    up_down_region_slack = bram_up_down_region_slack
                if self.nodes[nodeid].IsDSP():
                    up_down_region_slack = dsp_up_down_region_slack  
                if self.nodes[nodeid].IsinRegionConstr(left_right_region_slack, up_down_region_slack):
                    candidate_in_RegionConstr.append(siteid)
                self.nodes[nodeid].ReturnToDefaultPlaceLocation()
            if len(candidate_in_RegionConstr) == 0:
                logger.info("The Nonmacro Node with regional constraints could not find feasible place!!")
            
            flag = False
            if len(candidate_in_RegionConstr) > 0:
                while not flag:
                    place_site_id = choice(candidate_in_RegionConstr)
                    if not self.sites[place_site_id].CheckIsFull(self.nodes[nodeid].resourcetype):
                        locX = self.sites[place_site_id].locX
                        locY = self.sites[place_site_id].locY
                        self.nodes[nodeid].SetPlaceLocation(locX,locY,place_site_id)
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
                        locX = self.sites[place_site_id].locX
                        locY = self.sites[place_site_id].locY
                        self.nodes[id].SetPlaceLocation(locX,locY,place_site_id)
                        self.sites[place_site_id].addNode(self.nodes[id])
                        restype_loc[node.resourcetype].remove(place_site_id)
                        flag = True

    def RandomAugment(self, displacement_thres, augment_numper_thres, logger):
        num_macro_adjust_thres = int(self.num_macro*augment_numper_thres)
        num_macro_adjust = choice(list(range(4, num_macro_adjust_thres)))
        num_cascade_macro_adjust = choice(list(range(0, min(int(num_macro_adjust*0.3), self.num_cascade_macro)+1)))
        # The site and site column col to place the macros
        restype_loc = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "URAM288":[], "IO":[]}
        restype_col = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "URAM288":[], "IO":[]}
        col2loc = {}
        left_right_region_slack = 1.0 # The slack distance from the region left/right boundary
        bram_up_down_region_slack = 5.0 # The slack distance from the region up/down boundary for bram
        dsp_up_down_region_slack = 3.0 # The slack distance from the region up/down boundary for dsp

        for id in range(len(self.sites)):
            col_id = self.sites[id].locX
            if not col_id in col2loc:
                col2loc[col_id] = [id]
            else:
                col2loc[col_id].append(id)
            for j in range(len(list(self.sites[id].resource_supply.keys()))):
                res_name = list(self.sites[id].resource_supply.keys())[j]
                if self.sites[id].resource_supply[res_name] > 0:
                    restype_loc[res_name].append(id)
                    if not col_id in restype_col[res_name]:
                        restype_col[res_name].append(col_id)
        #print(restype_loc)
        #print(restype_col)

        # temporally clear the uncascaded macro on the sites
        for id in range(len(self.sites)):
            site_current = self.sites[id]
            nodecol = site_current.nodecol
            for nodeid in nodecol:
                if self.nodes[nodeid].cascade_id == -1:
                    self.sites[id].removeNode(self.nodes[nodeid])

        # choose the cascaded macros to be replaced in a range
        macro_candidate = self.cascademacros[:]
        #macro_candidate = []
        id = 0
        while len(macro_candidate) > 0 and id < num_cascade_macro_adjust:
            candidate_id = choice(list(range(len(macro_candidate))))
            macro = macro_candidate[candidate_id]
            macroid = macro.id
            macro_reference_id = macro.reference_node
            macro_res_type = self.nodes[macro_reference_id].resourcetype
            if macro_res_type == "RAMB36E2":
                up_down_region_slack = bram_up_down_region_slack
            elif macro_res_type == "DSP48E2":
                up_down_region_slack = dsp_up_down_region_slack
            nodecol = macro.Macronodecol
            macrolength = len(nodecol)
            Xcorr_macro = self.nodes[macro_reference_id].locX
            Ycorr_macro = self.nodes[macro_reference_id].locY
            site_macro = self.nodes[macro_reference_id].site
            
            # Construct feasible set
            column_candidate = []
            site_candidate = []
            for column_id in restype_col[macro_res_type]:
                if abs(column_id - Xcorr_macro) <= displacement_thres:
                    column_candidate.append(column_id)

            for column_id in column_candidate:
                loc = col2loc[column_id]
                for site_id in loc:
                    site_current = self.sites[site_id]
                    Xcorr_site_current = site_current.locX
                    Ycorr_site_current = site_current.locY
                    if Xcorr_site_current == Xcorr_macro and Ycorr_site_current == Ycorr_macro:
                        continue
                    if abs(Xcorr_site_current - Xcorr_macro) + abs(Ycorr_site_current - Ycorr_macro) > displacement_thres:
                        continue
                    subflag = True
                    uram_cnt = 0
                    for j in range(0, macrolength):
                        if macro_res_type == "URAM288":
                            if j!=0 and j % 4 == 0:
                                uram_cnt += 1
                            immed_site_id = site_id+uram_cnt
                        else:
                            immed_site_id = site_id+j
                        
                        X_immed = self.sites[immed_site_id].locX
                        Y_immed = self.sites[immed_site_id].locY
                        if self.sites[immed_site_id].CheckIsFull(macro_res_type):
                            subflag = False
                            break
                        if X_immed != Xcorr_site_current:
                            subflag = False
                            break
                        if self.checkRegionFull(macro_res_type, X_immed, Y_immed, left_right_region_slack, up_down_region_slack):
                            subflag = False
                            break
                                    
                    if subflag:
                        site_candidate.append(site_id)
            
            if len(site_candidate) == 0:
                macro_candidate.remove(macro)
                continue
            
            # Randomly selected a site and adjust the location
            site_chosen_id = choice(site_candidate)
            Xcorr_site_chosen = self.sites[site_chosen_id]
            Ycorr_site_chosen = self.sites[site_chosen_id]
            uram_cnt = 0
            for j in range(0, macrolength):
                nodeid = nodecol[j].id
                if macro_res_type == "URAM288":
                    if j != 0 and j % 4 == 0:
                        uram_cnt += 1
                    immed_site_id = site_chosen_id+uram_cnt
                    immed_site_id_org = site_macro+uram_cnt
                else:
                    immed_site_id = site_chosen_id+j
                    immed_site_id_org = site_macro+j
                immed_site_X = self.sites[immed_site_id].locX
                immed_site_Y = self.sites[immed_site_id].locY
                self.nodes[nodeid].ReSetPlaceLocation(immed_site_X, immed_site_Y, immed_site_id)
                self.cascademacros[macroid].Macronodecol[j].ReSetPlaceLocation(immed_site_X, immed_site_Y, immed_site_id)
                self.sites[immed_site_id].addNode(self.nodes[nodeid])
                #print(macro.name, self.nodes[nodeid].id, self.nodes[nodeid].name, self.sites[immed_site_id_org].nodecol, immed_site_id_org)
                for regionid in range(len(self.regionconstrtype)):
                    region = self.regionconstrtype[regionid]
                    if(region.IsinRegion(immed_site_X, immed_site_Y, left_right_region_slack, up_down_region_slack)):
                        self.regionconstrtype[regionid].AddNode(self.nodes[nodeid])
                self.sites[immed_site_id_org].removeNode(self.nodes[nodeid])
                if immed_site_id in restype_loc[macro_res_type]:
                    restype_loc[macro_res_type].remove(immed_site_id)
                    col2loc[immed_site_X].remove(immed_site_id)
            id = id + 1
            macro_candidate.remove(macro)
        
        cascade_adjust = id
        # Reduce the site that is full after placing the cascaded macro
        for id in range(len(self.sites)):
            siteid = self.sites[id].id
            colid = self.sites[id].locX
            for j in range(len(list(self.sites[id].resource_supply.keys()))):
                res_name = list(self.sites[id].resource_supply.keys())[j]
                if self.sites[id].resource_supply[res_name] > 0:
                    if self.sites[id].CheckIsFull(res_name) and (siteid in restype_loc[res_name]):
                        restype_loc[res_name].remove(siteid)
                        col2loc[colid].remove(siteid)

        # Fill the location with the non-cascade nodes (no overlapping with the cascaded macros)
        num_non_cascade_macro_adjust = num_macro_adjust - cascade_adjust
        num_non_cascade_macro_adjust = min(num_non_cascade_macro_adjust, self.num_basic_macro)
        non_cascade_macro_candidate = []
        non_cover_with_RC = []
        non_cover_wo_RC = []

        logger.info("Augment cascaded macros:"+str(cascade_adjust)+",basic macros:"+str(num_non_cascade_macro_adjust))
        for nodeid in range(len(self.nodes)):
            node_current = self.nodes[nodeid]
            node_restype = node_current.resourcetype
            if node_current.is_macro and node_current.cascade_id == -1:
                site_current = node_current.site
                Xcorr_current = node_current.locX
                if not self.sites[site_current].CheckIsFull(node_restype):
                    if node_current.regionconstr_type == -1:
                        non_cover_wo_RC.append(nodeid)
                    else:
                        non_cover_with_RC.append(nodeid)
                else:
                    non_cascade_macro_candidate.append(nodeid)
        
        #print(non_cover)
        num_choice = num_non_cascade_macro_adjust - len(non_cascade_macro_candidate)
        num_choice_nodes_with_RC = int(num_choice*0.4)
        num_choice_nodes_with_RC = min(num_choice_nodes_with_RC, len(non_cover_with_RC))
        num_choice_nodes_wo_RC = num_choice - num_choice_nodes_with_RC
        num_choice_nodes_wo_RC = min(num_choice_nodes_wo_RC, len(non_cover_wo_RC))

        if len(non_cover_with_RC) > 0:
            for id in range(num_choice_nodes_with_RC):
                node_choice_id = choice(non_cover_with_RC)
                non_cascade_macro_candidate.append(node_choice_id)
                non_cover_with_RC.remove(node_choice_id)

        if len(non_cover_wo_RC) > 0:
            for id in range(num_choice_nodes_wo_RC):
                node_choice_id = choice(non_cover_wo_RC)
                non_cascade_macro_candidate.append(node_choice_id)
                non_cover_wo_RC.remove(node_choice_id)
        
        for nodeid in non_cover_with_RC:
            site_current = self.nodes[nodeid].site
            column_current = self.nodes[nodeid].locX
            self.sites[site_current].addNode(self.nodes[nodeid])
            restype = self.nodes[nodeid].resourcetype
            restype_loc[restype].remove(site_current)
            col2loc[column_current].remove(site_current)

        for nodeid in non_cover_wo_RC:
            site_current = self.nodes[nodeid].site
            column_current = self.nodes[nodeid].locX
            self.sites[site_current].addNode(self.nodes[nodeid])
            restype = self.nodes[nodeid].resourcetype
            restype_loc[restype].remove(site_current)
            col2loc[column_current].remove(site_current)
        
        # Randomly find the corresponding place for each region-constrained node
        for nodeid in non_cascade_macro_candidate:
            if self.nodes[nodeid].regionconstr_type == -1:
                continue
            init_displacement_thres = displacement_thres
            site_candidate = []
            macro_locX = self.nodes[nodeid].locX
            macro_locY = self.nodes[nodeid].locY
            macro_siteid = self.nodes[nodeid].site
            macro_res_type = self.nodes[nodeid].resourcetype
            while len(site_candidate) == 0 and init_displacement_thres < 1000:
                for site_id in restype_loc[macro_res_type]:
                    site_current = self.sites[site_id]
                    site_locX = self.sites[site_id].locX
                    site_locY = self.sites[site_id].locY
                    #if macro_locX == site_locX and macro_locY == site_locY:
                    #    continue
                    self.nodes[nodeid].ReSetPlaceLocation(site_locX, site_locY, site_id)
                    if self.nodes[nodeid].IsBRAM():
                        up_down_region_slack = bram_up_down_region_slack
                    if self.nodes[nodeid].IsDSP():
                        up_down_region_slack = dsp_up_down_region_slack                        
                    if not self.nodes[nodeid].IsinRegionConstr(left_right_region_slack, up_down_region_slack):
                        self.nodes[nodeid].ReSetPlaceLocation(macro_locX, macro_locY, macro_siteid)
                        continue
                    self.nodes[nodeid].ReSetPlaceLocation(macro_locX, macro_locY, macro_siteid)
                    if abs(macro_locX - site_locX) + abs(macro_locY - site_locY) <= init_displacement_thres:
                        site_candidate.append(site_id)
                init_displacement_thres = init_displacement_thres * 1.5
            
            if len(site_candidate) > 0:
                placed_site_id = choice(site_candidate)
                placed_X = self.sites[placed_site_id].locX
                placed_Y = self.sites[placed_site_id].locY
                self.nodes[nodeid].ReSetPlaceLocation(placed_X, placed_Y, placed_site_id)
                self.sites[placed_site_id].addNode(self.nodes[nodeid])
                restype_loc[macro_res_type].remove(placed_site_id)
                col2loc[placed_X].remove(placed_site_id)
            else:
                place_id = self.nodes[nodeid].site
                self.sites[place_id].addNode(self.nodes[nodeid])


        # Randomly find the corresponding place for each non-region_constrained node
        for nodeid in non_cascade_macro_candidate:
            if self.nodes[nodeid].regionconstr_type != -1:
                continue
            # Generate the feasible location set for each node
            init_displacement_thres = displacement_thres
            site_candidate = []
            macro_locX = self.nodes[nodeid].locX
            macro_locY = self.nodes[nodeid].locY
            macro_res_type = self.nodes[nodeid].resourcetype
            while len(site_candidate) == 0:
                for site_id in restype_loc[macro_res_type]:
                    site_current = self.sites[site_id]
                    site_locX = self.sites[site_id].locX
                    site_locY = self.sites[site_id].locY
                    if macro_locX == site_locX and macro_locY == site_locY:
                        continue
                    if abs(macro_locX - site_locX) + abs(macro_locY - site_locY) <= init_displacement_thres:
                        site_candidate.append(site_id)
                init_displacement_thres = init_displacement_thres * 1.5
            
            placed_site_id = choice(site_candidate)
            placed_X = self.sites[placed_site_id].locX
            placed_Y = self.sites[placed_site_id].locY
            self.nodes[nodeid].ReSetPlaceLocation(placed_X, placed_Y, placed_site_id)
            self.sites[placed_site_id].addNode(self.nodes[nodeid])
            restype_loc[macro_res_type].remove(placed_site_id)
            col2loc[placed_X].remove(placed_site_id)
    
    def checkRegionFull(self, restype, Xcorr, Ycorr, left_right_boundary_slack, up_down_boundary_slack):
        flag_RegionFull = False
        for id in range(len(self.regionconstrtype)):
            region = self.regionconstrtype[id]
            if region.IsinRegion(Xcorr, Ycorr, left_right_boundary_slack, up_down_boundary_slack):
                if region.CheckIsFull(restype):
                    flag_RegionFull = True
        return flag_RegionFull

    
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
                X_corr = node.locX
                Y_corr = node.locY
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
        
    def OutputSolutionpl(self, output_path):
        with open(output_path, "w") as f_sol:
            output_str = ""
            for id in range(len(self.nodes)):
                if self.nodes[id].is_macro:
                    if self.nodes[id].cascade_id == -1 or self.nodes[id].is_cascade_refer:
                        output_str += self.nodes[id].name
                        output_str += " "
                        output_str += str(self.nodes[id].locX)
                        output_str += " "
                        output_str += str(self.nodes[id].locY)
                        output_str += " "
                        output_str += "0\n"
            f_sol.write(output_str)
            #print(output_path)
    
    def CheckLegalCoordinate(self, error_str, logger):
        is_legal = True
        for id in range(len(self.nodes)):
            if not self.nodes[id].is_macro:
                continue
            if (self.nodes[id].locX < 0 or self.nodes[id].locX > self.sitemap_width) or (self.nodes[id].locY < 0 or self.nodes[id].locY > self.sitemap_height):
                #logger.info("Invalid x location for node "+self.nodes[id].name+" "+str(self.nodes[id].locX))
                error_str += ("Invalid site ("+ str(self.nodes[id].locX) + ","+str(self.nodes[id].locY)+") for node "+self.nodes[id].name+"\n")
                is_legal = False
            if not self.nodes[id].IsinRegionConstr():
                error_str += ("The location ("+str(self.nodes[id].locX) + ","+str(self.nodes[id].locY)+") for node "+self.nodes[id].name+" not in the region constraints\n")
                #logger.info("The location for node "+self.nodes[id].name+" not in the region constraints")                
                is_legal = False
            siteid = self.nodes[id].site
            resourcetype = self.nodes[id].resourcetype
            if self.sites[siteid].resource_supply[resourcetype]<=0:
                #logger.info("Invalid capatible site for node:"+self.nodes[id].name+" site:("+str(self.sites[siteid].locX)+","+str(self.sites[siteid].locY)+")")
                error_str += ("Invalid site for node:"+self.nodes[id].name+" site:("+str(self.sites[siteid].locX)+","+str(self.sites[siteid].locY)+")\n")
                is_legal = False
        return is_legal, error_str

    def CheckResource(self, error_str, logger):
        is_legal = True
        for i in range(len(self.sites)):
            for j in range(len(list(self.sites[i].resource_supply.keys()))):
                res_name = list(self.sites[i].resource_supply.keys())[j]
                if self.sites[i].resource_usage[res_name] > self.sites[i].resource_supply[res_name]:
                    for id in range(len(self.sites[i].nodecol)):
                        nodeid = self.sites[i].nodecol[id]
                        #print(self.nodes[nodeid].name)
                    #logger.info("Excessive resource demand in Site:("+str(self.sites[i].locX)+","+str(self.sites[i].locY)+")"+" Demand:"+str(self.sites[i].resource_usage[res_name])+" Supply:"+str(self.sites[i].resource_supply[res_name]))
                    error_str += ("Excessive resource demand in Site:("+str(self.sites[i].locX)+","+str(self.sites[i].locY)+")"+" Demand:"+str(self.sites[i].resource_usage[res_name])+" Supply:"+str(self.sites[i].resource_supply[res_name])+"\n")
                    is_legal = False
        return is_legal, error_str
    
    def CheckMacroShape(self, error_str, logger):
        is_legal = True
        for i in range(len(self.cascademacros)):
            macro_node_col = self.cascademacros[i].Macronodecol
            macro_node_id_col = []
            for j in range(len(macro_node_col)):
                macro_node_id_col.append(macro_node_col[j].id)
            uram_cnt = 0
            ref_node = self.nodes[macro_node_id_col[0]]
            site_id = ref_node.site
            for j in range(0,len(macro_node_id_col)):
                nodeid = macro_node_id_col[j]
                if ref_node.resourcetype == "URAM288":
                    if j!=0 and j % 4 == 0:
                        uram_cnt += 1
                    new_site_id = site_id + uram_cnt
                else:
                    new_site_id = site_id + j
                gt_site_locX = self.sites[new_site_id].locX
                gt_site_locY = self.sites[new_site_id].locY
                cal_site_locX = self.nodes[nodeid].locX
                cal_site_locY = self.nodes[nodeid].locY
                if not (gt_site_locX==cal_site_locX and gt_site_locY==cal_site_locY):
                    #logger.info("Invalid Placement Result for Macro:"+ self.cascademacros[i].name)
                    error_str += ("The placed sites for cells in Macro:"+ self.cascademacros[i].name+"are not neighboring\n")
                    is_legal = False
                    break
        return is_legal, error_str
    
    def CheckLegality(self, placelegal_path, logger):
        is_legal = True
        f_legal = open(placelegal_path, "w")
        error_str = ""
        
        is_legal_sub, error_str = self.CheckLegalCoordinate(error_str, logger)
        if not is_legal_sub:
            logger.info("Cell Coordinate Legality Check does not pass")
            is_legal = False

        is_legal_sub, error_str = self.CheckResource(error_str, logger)        
        if not is_legal_sub:
            logger.info("Site Resource Check does not pass")
            is_legal = False
        
        is_legal_sub, error_str = self.CheckMacroShape(error_str, logger) 
        if not is_legal_sub:
            logger.info("Macro Placement Shape Check does not pass")
            is_legal = False
        
        if is_legal:
            logger.info("Legality Check pass!!")
        else:
            logger.info("Legality Check not pass!!")
            f_legal.write(error_str)
        
        f_legal.close()
        return is_legal
      
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
        logger.info("Available Site:"+str(self.num_avail_site))
        logger.info("loading fixed position")
        self.readFixedPl()
        logger.info("loading cascaded macro info")
        self.readCascadeMacros()
        logger.info("loading region constraints")
        self.readRegionConstraints()

        self.num_cells = len(self.cellLib)
        self.num_nodes = len(self.nodes)
        self.num_nets = len(self.nets)
        self.num_region_constr = len(self.regionconstrtype)
        self.num_cascade_macro = len(self.cascademacros)
        for id in range(len(self.nodes)):
            nodeper = self.nodes[id]
            if nodeper.resourcetype is not None:
                self.num_resource_demand[nodeper.resourcetype] = self.num_resource_demand[nodeper.resourcetype] + 1
            if nodeper.is_fixed:
                self.num_fix = self.num_fix + 1
            if nodeper.is_macro and nodeper.cascade_id == -1:
                self.num_basic_macro = self.num_basic_macro + 1
            if nodeper.is_macro and nodeper.cascade_id != -1:
                self.num_cascade_node = self.num_cascade_node + 1
            if nodeper.regionconstr_type!=-1:
                self.num_region_constr_node = self.num_region_constr_node + 1
                if nodeper.is_macro:
                    self.num_region_constr_maceonode = self.num_region_constr_maceonode + 1
                    if nodeper.cascade_id != -1:
                        self.num_region_constr_cascademaceonode += 1
            
        self.num_macro = self.num_basic_macro + self.num_cascade_macro

        logger.info("Number of fix nodes:"+str(self.num_fix))
        logger.info("Number of macros:"+str(self.num_macro))
        logger.info("Number of basic macros:"+str(self.num_basic_macro)+",cascade macros:"+str(self.num_cascade_macro))
        logger.info("Number of cascade macro nodes:"+str(self.num_cascade_node))
        logger.info("Region Constraints:"+str(self.num_region_constr))
        logger.info("Region Constraint Node:"+str(self.num_region_constr_node))
        logger.info("Region Constraint Macro Node:"+str(self.num_region_constr_maceonode))
        logger.info("Region Constraint Cascade Node:"+str(self.num_region_constr_cascademaceonode))
        str_out = "Resource for the nodes in the circuit:"
        for res_id, res_name in enumerate(list(self.num_resource_demand.keys())):
            str_out = str_out + res_name + ":"+str(self.num_resource_demand[res_name])+" "
        logger.info(str_out)
        str_out = "Supply Resource in the FPGA:"
        for res_id, res_name in enumerate(list(self.num_resource_supply.keys())):
            str_out = str_out + res_name + ":"+str(self.num_resource_supply[res_name])+" "
        logger.info(str_out)

def load_dataset(args, logger, placement=None):
    if args.custom_path != "":
        params = get_custom_design_params(args)
    else:
        params = get_single_design_params(
            args.dataset_root, args.dataset, args.design_name, placement
        )
    
    dataset = Dataset(params)
    if checkparam(params):
        logger.info("loading from original benchmark...")
        dataset.readAll(logger)
    return dataset