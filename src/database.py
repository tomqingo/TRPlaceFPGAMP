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
        
        #Macro number statistics
        self.num_basic_macro = 0
        self.num_cascade_macro = 0
        self.num_cascade_node = 0
        self.num_macro = 0

        #SiteMap
        self.sitemap_width = 0
        self.sitemap_height = 0
        self.num_avail_site = 0

        #fix node and regional constraints
        self.num_fix = 0
        self.num_region_constr = 0
        self.num_region_constr_node = 0
    
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
                        new_node.AddPins(self.cellLib[new_node_info[1]].pins)
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
                                pinId = self.cellLib[celltype].pinNameIdMap[pinname]
                                pin = self.cellLib[celltype].pins[pinId]
                                new_net.addPin([nodeid, pin])
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
                            if cur_line_col[2] == "URAM":
                                for iter in range(3):
                                    id = id + 1
                            new_site = Site(cur_line_col[2], site_id, int(cur_line_col[0]), int(cur_line_col[1]))
                            sitetype_id = self.sitetypeIdMap[cur_line_col[2]].id
                            new_site.addSupplyResource(self.sitetypes[sitetype_id].resourcecap)
                            self.sites.append(new_site)
                            self.sitemaps[int(cur_line_col[0])][int(cur_line_col[1])] = site_id
                            self.sitemap_res[int(cur_line_col[0])][int(cur_line_col[1])] = self.sitetypeIdMap[cur_line_col[2]].id
                            for res_id, res_name in enumerate(list(self.sitetypes[sitetype_id].resourcecap.keys())):     
                                self.num_resource_supply[res_name] += self.sitetypes[sitetype_id].resourcecap[res_name] 
                            id = id + 1
                            site_id = site_id + 1
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
                    if cur_line_col[0].upper() in list(self.macrotypeIdMap.keys()):
                        macroinst = CascadeMacro(cur_line_col[3], macroinst_id, cur_line_col[0].upper(), int(cur_line_col[1]), int(cur_line_col[2]))
                        id = id + 1
                        sub_cur_line = all_lines[id]
                        sub_cur_line_col = sub_cur_line.strip().split()
                        nodeid = self.nodeNameIdMap[sub_cur_line_col[0]]
                        self.nodes[nodeid].is_cascade_refer = True
                        macroinst.SetReferenceNode(nodeid)
                        id = id + 1
                        for subid in range(len(self.nodes)):
                            if cur_line_col[3] in self.nodes[subid].name:
                                self.nodes[subid].cascade_id = macroinst_id
                                if self.nodes[subid].celltype == self.macrotypes[self.macrotypeIdMap[macroinst.macrotype]].celltype:
                                    macroinst.addNode(self.nodes[subid], is_macro=True)
                                else:
                                    macroinst.addNode(self.nodes[subid], is_macro=False)
                        self.cascademacros.append(macroinst)
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
                            self.nodes[nodeid].regionconstr.extend(self.regionconstrtype[int(cur_line_col[1])].constrcol)
                            id = id + 1

    def readSamplePl(self, logger):
        if os.path.join(self.params["sample"]):
            with open(self.params["sample"], "r") as f_samp:
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
                        uram_cnt = 1 #use for uram_cascade
                        locX_refer = self.nodes[id].locX
                        locY_refer = self.nodes[id].locY
                        site_id = self.sitemaps[locX_refer][locY_refer].astype("int")
                        cascade_id = self.nodes[id].cascade_id
                        macro_inst = self.cascademacros[cascade_id]
                        for subid in range(1,len(macro_inst.Macronodecol)):
                            nodeid = macro_inst.Macronodecol[subid].id
                            if self.nodes[nodeid].resourcetype == "URAM288":
                                new_site_id = site_id + uram_cnt
                                if subid % 4 == 0:
                                    uram_cnt += 1
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
        logger.info("random generated macro placement results")
        restype_loc = {"LUT":[], "FF":[], "CARRY8":[], "DSP48E2":[], "RAMB36E2":[], "URAM288":[], "IO":[]}
        for id in range(len(self.sites)):
            for j in range(len(list(self.sites[id].resource_supply.keys()))):
                res_name = list(self.sites[id].resource_supply.keys())[j]
                restype_loc[res_name].append(id)
        
        for id in range(len(self.nodes)):
            node = self.nodes[id]
            if node.is_macro:
                if node.cascade_id == -1 or node.is_cascade_refer:
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
        
                    if node.is_cascade_refer:
                        uram_cnt = 1
                        locX_refer = node.locX
                        locY_refer = node.locY
                        site_id = self.sitemaps[locX_refer][locY_refer].astype("int")
                        cascade_id = node.cascade_id
                        macro_inst = self.cascademacros[cascade_id]
                        for subid in range(1,len(macro_inst.Macronodecol)):
                            nodeid = macro_inst.Macronodecol[subid].id
                            if self.nodes[nodeid].resourcetype == "URAM288":
                                new_site_id = site_id + uram_cnt
                                if subid % 4 == 0:
                                    uram_cnt += 1
                            else:
                                new_site_id = site_id + subid
                            new_site_locX = self.sites[new_site_id].locX
                            new_site_locY = self.sites[new_site_id].locY
                            self.nodes[nodeid].SetPlaceLocation(new_site_locX, new_site_locY, new_site_id)
                            self.cascademacros[cascade_id].Macronodecol[subid].SetPlaceLocation(new_site_locX, new_site_locY, new_site_id)
                            self.sites[new_site_id].addNode(self.nodes[nodeid])
                            if new_site_id in restype_loc[self.nodes[nodeid].resourcetype]:
                                restype_loc[self.nodes[nodeid].resourcetype].remove(new_site_id)                    

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
    
    def CheckLegalCoordinate(self, logger):
        for id in range(len(self.nodes)):
            if not self.nodes[id].is_macro:
                continue
            if self.nodes[id].locX < 0 or self.nodes[id].locX > self.sitemap_width:
                logger.info("Invalid x location for node "+self.nodes[id].name+" "+str(self.nodes[id].locX))
                return False
            if self.nodes[id].locY < 0 or self.nodes[id].locY > self.sitemap_height:
                logger.info("Invalid y location for node "+self.nodes[id].name+" "+str(self.nodes[id].locY))
                return False
            if not self.nodes[id].IsinRegionConstr():
                logger.info("The location for node "+self.nodes[id].name+" not in the region constraints")                
                return False
            siteid = self.nodes[id].site
            resourcetype = self.nodes[id].resourcetype
            if self.sites[siteid].resource_supply[resourcetype]<=0:
                logger.info("Invalid capatible site for node:"+self.nodes[id].name+" site:("+str(self.sites[siteid].locX)+","+str(self.sites[siteid].locY)+")")
                return False
        return True

    def CheckResource(self, logger):
        for i in range(len(self.sites)):
            for j in range(len(list(self.sites[i].resource_supply.keys()))):
                res_name = list(self.sites[i].resource_supply.keys())[j]
                if self.sites[i].resource_usage[res_name] > self.sites[i].resource_supply[res_name]:
                    logger.info("Excessive resource demand in Site:("+str(self.sites[i].locX)+","+str(self.sites[i].locY)+")"+" Demand:"+str(self.sites[i].resource_usage[res_name])+" Supply:"+str(self.sites[i].resource_supply[res_name]))
                    return False
        return True
    
    def CheckMacroShape(self, logger):
        for i in range(len(self.cascademacros)):
            macro_node_col = self.cascademacros[i].Macronodecol
            macro_node_id_col = []
            for j in range(len(macro_node_col)):
                macro_node_id_col.append(macro_node_col[j].id)
            uram_cnt = 1
            ref_node = self.nodes[macro_node_id_col[0]]
            site_id = ref_node.site
            for j in range(0,len(macro_node_id_col)):
                nodeid = macro_node_id_col[j]
                if ref_node.resourcetype == "URAM288":
                    new_site_id = site_id + uram_cnt
                    if j % 4 == 0:
                        uram_cnt += 1
                else:
                    new_site_id = site_id + j
                gt_site_locX = self.sites[new_site_id].locX
                gt_site_locY = self.sites[new_site_id].locY
                cal_site_locX = self.nodes[nodeid].locX
                cal_site_locY = self.nodes[nodeid].locY
                if not (gt_site_locX==cal_site_locX and gt_site_locY==cal_site_locY):
                    logger.info("Invalid Placement Result for Macro:"+ self.cascademacros[i].name)
                    return False
        return True
    
    def CheckLegality(self, logger):
        if not self.CheckResource(logger):
            logger.info("Site Resource Check does not pass")
            return False
        if not self.CheckMacroShape(logger):
            logger.info("Macro Placement Shape Check does not pass")
            return False
        if not self.CheckLegalCoordinate(logger):
            logger.info("Cell Coordinate Legality Check does not pass")
            return False
        logger.info("Check pass!!")
        return True
      
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
            if nodeper.is_macro:
                self.num_basic_macro = self.num_basic_macro + 1
            if nodeper.is_macro and nodeper.cascade_id != -1:
                self.num_cascade_node = self.num_cascade_node + 1
            if nodeper.regionconstr_type!=-1:
                self.num_region_constr_node = self.num_region_constr_node + 1
        self.num_macro = self.num_basic_macro - self.num_cascade_node + self.num_cascade_macro

        logger.info("Number of fix nodes:"+str(self.num_fix))
        logger.info("Number of macros:"+str(self.num_macro))
        logger.info("Number of basic macros:"+str(self.num_basic_macro)+",cascade macros:"+str(self.num_cascade_macro))
        logger.info("Number of cascade macro nodes:"+str(self.num_cascade_node))
        logger.info("Region Constraints:"+str(self.num_region_constr))
        logger.info("Region Constraint Node:"+str(self.num_region_constr_node))
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