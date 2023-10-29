import numpy as np

class Node:
    def __init__(self, name, id, celltype):  
        # basic information      
        self.name = name
        self.id = id
        self.celltype = celltype
        self.resourcetype = None
        # The pins
        self.pinName2Id = {}
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
        # whether the node is placed
        self.isPlace = False
        # site (locX, locY)
        self.locX = -1
        self.locY = -1
        self.site = -1
        # real (realX, real Y)
        self.realX = -1
        self.realY = -1
    
    def IsMacro(self):
        # Whther 'BRAM' and 'DSP' is in the node name
        if 'RAMB' in self.celltype or 'DSP' in self.celltype:
            return True
        else:
            return False
        
    def IsLUT(self):
        if 'LUT' in self.celltype:
            return True
        else:
            return False
    
    def IsFF(self):
        if 'FF' in self.celltype:
            return True
        else:
            return False
    
    def IsBRAM(self):
        if 'RAMB' in self.celltype:
            return True
        else:
            return False
    
    def IsDSP(self):
        if 'DSP' in self.celltype:
            return True
        else:
            return False

    def IsIO(self):
        if 'IBUF' in self.celltype or ('OBUF' in self.celltype or 'BUFGCE' in self.celltype):
            return True
        else:
            return False
    
    def hasRegionConstr(self):
        return (self.regionconstr_type != -1)         
    
    def IsinRegionConstr(self, left_right_region_slack=0, up_down_region_slack=0):
        if self.regionconstr_type == -1:
            return True
        else:
            for constrid in range(len(self.regionconstr)):
                XLo = self.regionconstr[constrid][0]+left_right_region_slack
                YLo = self.regionconstr[constrid][1]+up_down_region_slack
                XHi = self.regionconstr[constrid][2]-left_right_region_slack
                YHi = self.regionconstr[constrid][3]-up_down_region_slack
                if (self.locX>=XLo and self.locX<=XHi) and (self.locY>=YLo and self.locY<=YHi):
                    return True
            return False

    # Set the location for the cells
    def SetPlaceLocation(self, locX, locY, realX, realY, site):
        self.ResetPlaceLocation()
        self.locX = locX
        self.locY = locY
        self.realX = realX
        self.realY = realY
        self.site = site
        self.isPlace = True

    # ReSet the location for the cells   
    def ResetPlaceLocation(self):
        self.locX = -1
        self.locY = -1
        self.realX = -1
        self.realY = -1
        self.site = -1
        self.isPlace = False
    
    def SetResourceType(self, resourcetype):
        self.resourcetype = resourcetype
    
    def addPin(self, pins):
        for id in range(len(pins)):
            if not pins[id].name in list(self.pinName2Id.keys()):
                self.pinName2Id[pins[id].name] = len(self.pins)
                self.pins.append(pins[id])
        self.pin_num = len(self.pins)

    def addNeighboringNets(self, netid):
        self.netIds.append(netid)
    
    def addNeighboringOutNets(self, outnetid):
        self.outnetIds.append(outnetid)
    
    def addNeighboringInNets(self, innetid):
        self.innetIds.append(innetid) 

    def getLocation(self):
        return self.locX, self.locY, self.realX, self.realY

    def getlocatedsiteid(self):
        return self.site  


