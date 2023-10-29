import os

class PlacementUnit:   
    def __init__(self, id, macrotype, is_macro, is_cascade, hasRegionConstr):

        self.id = id
        self.macrotype = macrotype # BRAM or DSP
        self.is_macro = is_macro # macro or not macro
        self.is_cascade = is_cascade # cascade macro or the non_cascade macro
        self.num_cells = 0 # the number of cells in the placement unit
        self.hasRegionConstr = hasRegionConstr # Whther the placement unit has the region constr requirement
        self.nodeidcol = [] # The nodes in the placement units
        self.placementnetidcol = [] # The placement nets attached to the placement units
        self.degree = 0 # The number of nodes (not the same cascaded macros) connected
        self.internalnetnum = 0 # The number of the nets in the cascaded macros
        self.connExternalMacronum = 0 # The number of macros (not the same cascaded macros) connected
        self.connBRAMnum = 0 # The BRAM macros directly connected
        self.connDSPnum = 0 # The DSP macros directly connected
        self.connIOnum = 0 # The number of IO ports connected
        self.connLUTFFnum = 0 # The number of LUT/FFs connected
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
    
    def updateDegree(self):
        self.degree = self.connExternalMacronum + self.connIOnum + self.connLUTFFnum

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
        self.nodeId2placementunitId = {} # node Id to the placement unit Id containing it
    
    def ConvertNodes2PlacementUnits(self, logger):
        logger.info("Convert the node list to the placement units")
        placementunit_id = 0 # placement unit id
        # Find all the nodes in the placement units (Simple Macros and Cascade Macros)
        for id in range(len(self.database.nodes)):
            node = self.database.nodes[id]
            # Judge whether the nodes are already in the nodeId2placementunitId
            if node.id in list(self.nodeId2placementunitId.keys()):
                continue
            if node.is_macro:
                # Simple Macro
                if node.cascade_id == -1:
                    placementunit_inst = PlacementUnit(placementunit_id, node.resourcetype, True, False, node.hasRegionConstr())
                    placementunit_inst.addNode(node.id)
                    self.nodeId2placementunitId[node.id] = placementunit_id
                else:
                # Cascade Macro (Assume there is no regional constraint for cascade macro)
                    macro_id = node.cascade_id
                    macroinst = self.database.cascademacros[macro_id]
                    NodeinMacrocol = macroinst.Macronodecol
                    ref_node = self.database.nodes[macroinst.reference_node]
                    placementunit_inst = PlacementUnit(placementunit_id, ref_node.resourcetype, True, True, False)
                    for id in range(len(NodeinMacrocol)):
                        placementunit_inst.addNode(NodeinMacrocol[id].id)
                        self.nodeId2placementunitId[NodeinMacrocol[id].id] = placementunit_id
                self.placementunits.append(placementunit_inst)
                placementunit_id = placementunit_id + 1
        logger.info("Num of Placement Units: {}".format(placementunit_id))

    
    def ConvertNets2PlacementNets(self, logger):
        logger.info("Convert the net list to the placement nets")
        placementnet_id = 0
        # trasverse all the nets in the benchmark
        for id in range(len(self.database.nets)):
            net = self.database.nets[id]
            placementnet_inst = PlacementNet(placementnet_id)
            num_ConnTo_LUTFF = 0 # the number of the LUTs and FFs connected
            num_ConnTo_IO = 0 # the number of the IO ports connected
            num_ConnTo_BRAM = 0 # the number of the BRAMs connected
            num_ConnTo_DSP = 0 # the number of the DSPs connected
            pindict = {}
            nodedict = {}
            for pin_id in range(len(net.pins)):
                node_id = net.pins[pin_id][0]
                # Judge whether the pin is Macro 
                if node_id in list(self.nodeId2placementunitId.keys()):
                    # Reduce the repeated pins connected to the same node for the net
                    if not node_id in list(nodedict.keys()):
                        placementunit_id = self.nodeId2placementunitId[node_id]
                        nodedict[node_id] = 1
                        # Add the macro pin to the pin set
                        if not placementunit_id in list(pindict.keys()):
                            pindict[placementunit_id] = 1
                            placementnet_inst.addPin(placementunit_id)
                            num_ConnTo_BRAM += int(self.placementunits[placementunit_id].isBRAM())
                            num_ConnTo_DSP += int(self.placementunits[placementunit_id].isDSP())                        
                        else:
                            pindict[placementunit_id] += 1
                else:
                    # Calculate the number of IO ports that the net connect
                    if self.database.nodes[node_id].IsIO():
                        num_ConnTo_IO += 1
                    else:
                    # Calculate the number of LUTs/FFs that the net connect
                        num_ConnTo_LUTFF += 1

            # The net connected to only the LUTs, FFs and IOs, skip
            
            if placementnet_inst.pinnum < 1:
                continue
            elif placementnet_inst.pinnum == 1:
                placementunit_id = placementnet_inst.placementunitidcol[0]
                if pindict[placementunit_id] > 1:
                    self.placementunits[placementunit_id].internalnetnum += 1
            # The net connected to more than two macros, consider as the placement net
            else:
                self.placementnets.append(placementnet_inst)
                placementnet_id += 1
                
            for pinid in range(placementnet_inst.pinnum):
                placementunit_id = placementnet_inst.placementunitidcol[pinid]
                if placementnet_inst.pinnum > 1:
                    self.placementunits[placementunit_id].placementnetidcol.append(id)
                self.placementunits[placementunit_id].connExternalMacronum += (placementnet_inst.pinnum - 1)
                self.placementunits[placementunit_id].connIOnum += num_ConnTo_IO
                self.placementunits[placementunit_id].connLUTFFnum += num_ConnTo_LUTFF
                self.placementunits[placementunit_id].connBRAMnum += (num_ConnTo_BRAM - int(self.placementunits[placementunit_id].isBRAM()))
                self.placementunits[placementunit_id].connDSPnum += (num_ConnTo_DSP - int(self.placementunits[placementunit_id].isDSP()))     
                self.placementunits[placementunit_id].updateDegree()
        logger.info("Num of Placement Nets: {}".format(placementnet_id))

    def ConvertDB2PlacementInfo(self, logger):
        self.ConvertNodes2PlacementUnits(logger)
        self.ConvertNets2PlacementNets(logger)




                

                        




