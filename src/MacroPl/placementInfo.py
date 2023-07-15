import os

class PlacementUnit:   
    def __init__(self, id, macrotype, is_macro, is_cascade):
        self.id = id
        self.macrotype = macrotype #BRAM, URAM, or DSP
        self.is_macro = is_macro
        self.is_cascade = is_cascade
        self.num_cells = 0
        self.nodeidcol = []
        self.placementnetidcol = []
        self.internalnetnum = 0
        self.externalnetnum = 0
        self.connNonmacronum = 0
        #location
        self.locX = 0
        self.locY = 0
    
    def addNode(self, nodeid):
        self.nodeidcol.append(nodeid)
        self.num_cells = self.num_cells + 1
    
    def setLocation(self, locX, locY):
        self.locX = locX
        self.locY = locY
    
    def getNodeSet(self):
        return self.nodeidcol
    
    def isBRAM(self):
        return ("RAMB" in self.macrotype)
    
    def isDSP(self):
        return ("DSP" in self.macrotype)
    
    def isURAM(self):
        return ("URAM" in self.macrotype)

class PlacementNet:
    def __init__(self, id):
        self.id = id
        self.pinnum = 0
        self.placementunitidcol = []
    
    def addPin(self, placementunitid):
        self.placementunitidcol.append(placementunitid)
        self.pinnum = self.pinnum + 1

class PlacementInfo:
    def __init__(self, database):
        self.database = database
        self.placementunits = []
        self.placementnets = []
        self.nodeId2placementunitId = {}
        self.macroId2placementunitId = {}
    
    def ConvertNodes2PlacementUnits(self, logger):
        logger.info("Convert the node list to the placement units")
        placementunit_id = 0
        for id in range(len(self.database.nodes)):
            node = self.database.nodes[id]
            if node.id in list(self.nodeId2placementunitId.keys()):
                continue
            if node.is_macro:
                if node.cascade_id == -1:
                    placementunit_inst = PlacementUnit(placementunit_id, node.resourcetype, True, False)
                    placementunit_inst.addNode(node.id)
                    self.nodeId2placementunitId[node.id] = placementunit_id
                else:
                    macro_id = node.cascade_id
                    macroinst = self.database.cascademacros[macro_id]
                    NodeinMacrocol = macroinst.Macronodecol
                    ref_node = self.database.nodes[macroinst.reference_node]
                    placementunit_inst = PlacementUnit(placementunit_id, ref_node.resourcetype, True, True)
                    for id in range(len(NodeinMacrocol)):
                        placementunit_inst.addNode(NodeinMacrocol[id].id)
                        self.nodeId2placementunitId[NodeinMacrocol[id].id] = placementunit_id
                    self.macroId2placementunitId[macro_id] = placementunit_id
                self.placementunits.append(placementunit_inst)
                placementunit_id = placementunit_id + 1
        logger.info("Num of Placement Units: {}".format(placementunit_id))

    
    def ConvertNets2PlacementNets(self, logger):
        logger.info("Convert the net list to the placement nets")
        placementnet_id = 0
        for id in range(len(self.database.nets)):
            net = self.database.nets[id]
            placementnet_inst = PlacementNet(placementnet_id)
            flag_connto_nonmacro = False
            pindict = {}
            for pin_id in range(len(net.pins)):
                node_adj_id = net.pins[pin_id][0]
                if node_adj_id in list(self.nodeId2placementunitId.keys()):
                    placementunit_id = self.nodeId2placementunitId[node_adj_id]
                    if not placementunit_id in list(pindict.keys()):
                        pindict[placementunit_id] = 1
                        placementnet_inst.addPin(placementunit_id)
                else:
                    flag_connto_nonmacro = True
            if placementnet_inst.pinnum < 1:
                continue
            if placementnet_inst.pinnum == 1:
                if flag_connto_nonmacro:
                    self.placementunits[placementnet_inst.placementunitidcol[0]].connNonmacronum += 1
                    self.placementunits[placementnet_inst.placementunitidcol[0]].externalnetnum += 1
                else:
                    if len(net.pins) != 1:
                        self.placementunits[placementnet_inst.placementunitidcol[0]].internalnetnum += 1
                continue
            self.placementnets.append(placementnet_inst)
            for pinid in range(placementnet_inst.pinnum):
                placementunit_id = placementnet_inst.placementunitidcol[pinid]
                self.placementunits[placementunit_id].placementnetidcol.append(id)
                self.placementunits[placementunit_id].externalnetnum += 1
                if flag_connto_nonmacro:
                    self.placementunits[placementunit_id].connNonmacronum += 1
            placementnet_id += 1
        logger.info("Num of Placement Nets: {}".format(placementnet_id))

    def Convert2PlacementInfo(self, logger):
        self.ConvertNodes2PlacementUnits(logger)
        self.ConvertNets2PlacementNets(logger)




                

                        




