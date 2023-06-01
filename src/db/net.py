import numpy as np

class Net:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.pin_num = 0
        self.pins = []
        self.is_in_cascade = False
    
    def addPin(self, pin):
        self.pins.append(pin)
        self.pin_num = self.pin_num + 1