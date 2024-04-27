import numpy as np

class Net:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.pin_num = 0
        self.isclknet = False
        self.isctrlnet = False
        self.ishighdegree = False
        self.pins = []
        self.macropins = []
        self.cascademacropins = []
        self.IOpins = []
        self.is_in_cascade = False
        self.weight = 1
        if "clk_" in self.name:
            self.isclknet = True
        
    def addPin(self, pin):
        self.pins.append(pin)
        self.pin_num = self.pin_num + 1

    def addMacroPin(self, nodeid):
        self.macropins.append(nodeid)
    
    def addCascadeMacroPin(self, cascadeid):
        self.cascademacropins.appennd(cascade_id)

    def addIOPin(self, nodeid):
        self.IOpins.append(nodeid)
    
    def getclkPinNum(self):
        return self.clkpin_num
    
    def getPinNum(self):
        return self.pin_num
    
    def setHighDegreeNet(self):
        if len(self.pins) > 100:
            self.ishighdegree = True
    
    def setWeight(self, _weight):
        self.weight = _weight
        
