import numpy as np

class CascadeMacroType:
    def __init__(self, name, id, width, height):
        self.name = name
        self.id = id
        self.width = width
        self.height = height
        self.num_cells = self.width*self.height
    
    def getCellType(self, celltype):
        self.celltype = celltype

class CascadeMacro:
    def __init__(self, name, id, macrotype, width, height):
        self.name = name
        self.id = id
        self.macrotype = macrotype
        self.width = width
        self.height = height
        self.Macronodecol = []
        self.Nonmacronodecol = []
    
    def addNode(self, node, is_macro):
        if is_macro:
            self.Macronodecol.append(node)
            # Sort the node according to the name for 1,2,3...n
            if len(self.Macronodecol) > 1:
                ordered_list = self.Macronodecol[1:]
                if "BRAM" in self.macrotype:
                    ordered_list.sort(key=lambda x:int(x.name.split("/")[-1][13:]))
                elif "URAM" in self.macrotype:
                    ordered_list.sort(key=lambda x:int(x.name.split("/")[-1][12:]))
                else:
                    ordered_list.sort(key=lambda x:int(x.name.split("/")[1][18:]))
                self.Macronodecol[1:] = ordered_list                                        
        else:
            self.Nonmacronodecol.append(node)
    
    def SetReferenceNode(self, nodeid):
        self.reference_node = nodeid
