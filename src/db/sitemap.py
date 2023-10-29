import numpy as np

class Resource:
    def __init__(self, name, id):
        self.resource_table = {"LUT":0, "FF":1, "CARRY8":2, "DSP48E2":3, "RAMB36E2":4, "IO":5}
        self.name = name
        self.id = id
        self.celltypecol = []
    def GetResourceName(self, res_id):
        res_name_col = [k for k,v in self.resource_table.items() if v==res_id]
        res_name = res_name_col[0]
        return res_name
    def GetResourceId(self, res_name):
        return self.resource_table[res_name]
    def AddCelltype(self, celltype):
        self.celltypecol.append(celltype)

class SiteType:
    def __init__(self, name, id):
        self.site_table = {"SLICE":0, "DSP":1, "BRAM":2, "IO":3}
        self.name = name
        self.id = id
        #self.resource_col = []
        self.resource = {}
    def GetSiteTypeName(self, sitetype_id):
        sitetype_name_col = [k for k,v in self.site_table.items() if v==sitetype_id]
        sitetype_name = sitetype_name_col[0]
        return sitetype_name
    def GetSiteTypeId(self, sitetype_name):
        return self.site_table[sitetype_name]
    #def AddResource(self, resource):
    #    self.resource_col.append(resource)
    def AddResourceMulti(self, resource, cnt):
        #for i in range(cnt):
        #    self.AddResource(resource)
        self.resource[resource] = cnt

class Site:
    def __init__(self, sitetype, id, locX, locY):
        self.sitetype = sitetype
        self.id = id
        # site (locX, locY)
        self.locX = locX
        self.locY = locY
        # (width, height)
        self.width = 1
        if sitetype == "BRAM":
            self.height = 5
        elif sitetype == "DSP":
            self.height = 2.5
        elif sitetype == "IO":
            self.height = 26/30
        else:
            self.height = 1
        # real location (realX, realY)
        self.realX = self.locX
        self.realY = self.locY + self.height*1.0/2
        
        self.resource_usage = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "IO":0}
        self.resource_supply = {"LUT":0, "FF":0, "CARRY8":0, "DSP48E2":0, "RAMB36E2":0, "IO":0}
        self.nodecol = []
    
    def addSupplyResource(self, supply):
        for id in range(len(list(supply.keys()))):
            self.resource_supply[list(supply.keys())[id]] = supply[list(supply.keys())[id]]
    
    def addNode(self, node):
        self.nodecol.append(node.id)
        self.resource_usage[node.resourcetype] += 1
    
    def removeNode(self, node):
        self.nodecol.remove(node.id)
        self.resource_usage[node.resourcetype] -= 1
    
    def CheckIsFull(self, res_name):
        return self.resource_supply[res_name] <= self.resource_usage[res_name]
    
    def getLocation(self):
        return self.locX, self.locY, self.realX, self.realY



