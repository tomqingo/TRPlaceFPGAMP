import numpy as np

class CascadeMacroType:
    def __init__(self, name, id, num_col, num_row):
        self.name = name
        self.id = id
        self.num_col = num_col
        self.num_row = num_row
        self.num_cells = self.num_col*self.num_row
        if "BRAM" in name:
            self.height = self.num_row * 5
        elif "DSP" in name:
            self.height = self.num_row * 2.5
        else:
            self.height = self.num_row
        self.width = self.num_col

    def getCellType(self, celltype):
        self.celltype = celltype

class CascadeMacro:
    def __init__(self, name, id, macrotype, num_col, num_row):
        self.name = name
        self.id = id
        self.macrotype = macrotype
        self.num_col = num_col
        self.num_row = num_row
        self.num_cells = self.num_col*self.num_row
        if "BRAM" in name:
            self.height = self.num_row * 5
        elif "DSP" in name:
            self.height = self.num_row * 2.5
        else:
            self.height = self.num_row
        self.width = self.num_col
        self.Macronodecol = []
        self.Nonmacronodecol = []
        # The conversion from the macro nodes to the 
        self.reference_node = -1

        # Location of the reference node
        self.locX = -1
        self.locY = -1
        self.realX = -1
        self.realY = -1

        # Degree of the cascademacros
        self.degree = 0  # total number of connections
        self.degree_IO  = 0  # total number of IO connections
        self.connectedPlacementNets = []
    
    def addNode(self, node, is_macro):
        if is_macro:
            self.Macronodecol.append(node)
            # Sort the node according to the name for 1,2,3...n
            if len(self.Macronodecol) > 1:
                ordered_list = self.Macronodecol[1:]
                if "BRAM" in self.macrotype:
                    ordered_list.sort(key=lambda x:int(x.name.split("/")[-1][13:]))
                else:
                    ordered_list.sort(key=lambda x:int(x.name.split("/")[1][18:]))
                self.Macronodecol[1:] = ordered_list                                      
        else:
            self.Nonmacronodecol.append(node)
    
    def SetReferenceNode(self, nodeid):
        self.reference_node = nodeid
    
    def SetCascadeMacroLoc(self, locX, locY):
        self.locX = locX
        self.locY = locY
        self.realX = self.locX
        self.realY = self.locY + self.height / 2


        
