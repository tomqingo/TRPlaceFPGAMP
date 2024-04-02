import numpy as np

class RegionConstrType:
    def __init__(self, id, num_boxes):
        self.id = id
        self.num_boxes = num_boxes
        self.area = 0
        self.bramarea = 0
        self.dsparea = 0
        self.constrcol = []
        self.resource_usage = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "IO":0}
        self.resource_supply = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "IO":0}
        self.site_col = [] #Site In the Region
        self.node_col = [] #Node In the Region
    
    def AddBox(self, xLo, yLo, xHi, yHi):
        self.constrcol.append([xLo, yLo, xHi, yHi])
    
    def IsinRegion(self, Xcorr, Ycorr, left_right_boundary_slack=0, up_down_boundary_slack=0):
        flag_IsinRegion = False
        for rect_id in range(len(self.constrcol)):
            constr = self.constrcol[rect_id]
            xLo = constr[0]
            yLo = constr[1]
            xHi = constr[2]
            yHi = constr[3]
            if (xLo+left_right_boundary_slack<=Xcorr and Xcorr<= xHi-left_right_boundary_slack) and (yLo+up_down_boundary_slack<=Ycorr and Ycorr <= yHi-up_down_boundary_slack):
                flag_IsinRegion = True
        return flag_IsinRegion

    def AddSite(self, site):
        site_id = site.id
        res_supply = site.resource_supply
        for id in range(len(list(res_supply.keys()))):
            self.resource_supply[list(res_supply.keys())[id]] += res_supply[list(res_supply.keys())[id]]
        self.site_col.append(site_id)
        # For different kinds of cells, area is different
        if site.sitetype == "DSP":
            self.area += 2.5
            self.dsparea += 2.5
        elif site.sitetype == "BRAM":
            self.area += 5
            self.bramarea += 5
        else:
            self.area += 1

    def AddNode(self, node):
        node_id = node.id
        node_restype = node.resourcetype
        self.resource_usage[node_restype] += 1
        self.node_col.append(node_id)

    def CheckIsFull(self, res_type):
        return self.resource_supply[res_type] <= self.resource_usage[res_type]
        
