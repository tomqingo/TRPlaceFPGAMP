import numpy as np

class Node:
    def __init__(self, name, id, celltype):        
        self.name = name
        self.id = id
        self.celltype = celltype
        self.resourcetype = None
        # The pins
        self.pins = []
        self.pin_num = 0
        # macro and cascade
        self.is_macro = self.IsMacro() #True for the nodes in the macro
        self.cascade_id = -1 #-1 means not in other the cascaded cells 
        self.is_cascade_refer = False
        # Whether the node is fixed nor not
        self.is_fixed = False
        self.fixed_corr = [] #corr (x_{i},y_{i},index)
        # Region constrain
        self.regionconstr_type = -1
        self.regionconstr = [] #corr [(xLo,yLo,xHi,yHi)_{1},...(xLo,yLo,xHi,yHi)_{N}]
        # net information
        self.netIds = []
        self.outnetIds = []
        self.innetIds = []
        # position
        self.isPlace = False
        self.locX = 0
        self.locY = 0
        self.site = 0
    
    def IsMacro(self):
        # Whther 'BRAM', 'URAM' and 'DSP' is in the node name
        if 'BRAM' in self.name or ('URAM' in self.name or 'DSP' in self.name):
            return True
        else:
            return False
    
    def IsinRegionConstr(self):
        if self.regionconstr_type == -1:
            return True
        else:
            for constrid in range(len(self.regionconstr)):
                XLo = self.regionconstr[constrid][0]
                YLo = self.regionconstr[constrid][1]
                XHi = self.regionconstr[constrid][2]
                YHi = self.regionconstr[constrid][3]
                if (self.locX>=XLo and self.locX<=XHi) and (self.locY>=YLo and self.locY<=YHi):
                    return True
            return False

    def SetPlaceLocation(self, locX, locY, site):
        if not self.isPlace:
            self.locX = locX
            self.locY = locY
            self.site = site
            self.isPlace = True
    
    def SetResourceType(self, resourcetype):
        self.resourcetype = resourcetype
    
    def AddPins(self, pins):
        self.pins.extend(pins)
        self.pin_num = len(self.pins)

    def addNeighboringNets(self, netid):
        self.netIds.append(netid)
    
    def addNeighboringOutNets(self, outnetid):
        self.outnetIds.append(outnetid)
    
    def addNeighboringInNets(self, innetid):
        self.innetIds.append(innetid)     


