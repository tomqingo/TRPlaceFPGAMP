import numpy as np

class Pin:
    def __init__(self, name, id, celltype):
        self.name  = name
        self.id = id
        self.celltype = celltype
        self.clock_label = False
        self.ctrl_label = False
    
    def SetIO (self, IO_label):
        self.IO_label = IO_label #Input: 1, output: 0
    
    def SetClock (self):
        self.clock_label = True #True: clock Line, False: data Line

    def SetCtrl(self):
        self.ctrl_label = True #True: ctrl Line, False: data Line

class Cell:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.pinNameIdMap = {}
        self.pins = []
        self.pin_num = 0
        self.resourcename = None
    
    def addPin(self, pin_name, pin_IO, pin_clock=False, pin_ctrl=False):
        pin_id = len(self.pins)
        newpin = Pin(pin_name, pin_IO, self.name)
        newpin.SetIO(pin_IO)
        if pin_clock:
            newpin.SetClock()
        if pin_ctrl:
            newpin.SetCtrl()
        self.pins.append(newpin)
        self.pinNameIdMap[pin_name] = pin_id
        self.pin_num = self.pin_num + 1
