import numpy as np

class RegionConstrType:
    def __init__(self, id, num_boxes):
        self.id = id
        self.num_boxes = num_boxes
        self.constrcol = []
    
    def AddBox(self, xLo, yLo, xHi, yHi):
        self.constrcol.append([xLo, yLo, xHi, yHi])
