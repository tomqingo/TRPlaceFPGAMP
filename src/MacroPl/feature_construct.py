import os
import numpy as np

class FeatureExtractor:
    def __init__(self, placementinfo):
        self.placementinfo = placementinfo
        self.ObtainNodeFeature()
        self.ObtainNodelink()
    
    def ObtainNodeFeature(self):
        node_feature_col = []
        for id in range(len(self.placementinfo.placementunits)):
            placementunit = self.placementinfo.placementunits[id]
            node_feature = []
            # BRAM/DSP
            if placementunit.isBRAM():
                node_feature.extend([1,0])
            elif placementunit.isDSP():
                node_feature.extend([0,1])
            
            #Cascade/Single
            if placementunit.is_cascade:
                node_feature.extend([1,0])
            else:
                node_feature.extend([0,1])
            
            #Number of the cells in on placementunit
            node_feature.append(placementunit.num_cells)
            
            # degree
            degree = placementunit.degree
            # Other placement units
            connExternalMacronum = placementunit.connExternalMacronum
            # IO ports
            connIOnum = placementunit.connIOnum
            # LUTs/FFs
            connLUTFFnum = placementunit.connLUTFFnum
            # BRAMs
            connBRAMMacronum = placementunit.connBRAMnum
            # DSPs
            connDSPMacronum = placementunit.connDSPnum
            # internal nets for cascaded macros
            internalnetnum = placementunit.internalnetnum
            # regionconstr area
            regionconstrarea = placementunit.regionconstrarea
            # total area
            area = placementunit.area

            node_feature.extend([degree, connExternalMacronum, connIOnum, connLUTFFnum, connBRAMMacronum, connDSPMacronum, internalnetnum, regionconstrarea, area])            
            node_feature_col.append(node_feature)
        self.node_feature_map = np.array(node_feature_col)
    
    def ObtainNodelink(self):
        num_placementunit = len(self.placementinfo.placementunits)
        self.adj_matrix = np.zeros((num_placementunit, num_placementunit))
        for placementunit_i in range(num_placementunit):
            for placementunit_j in range(placementunit_i, num_placementunit):
                if placementunit_i == placementunit_j:
                    continue
                netidcol_i = self.placementinfo.placementunits[placementunit_i].placementnetidcol
                netidcol_j = self.placementinfo.placementunits[placementunit_j].placementnetidcol
                # Find the intersection of the connected net sets of these two nodes
                intersect_set = set(netidcol_i).intersection(set(netidcol_j))
                if len(intersect_set) > 0:
                    self.adj_matrix[placementunit_i][placementunit_j] = 1
                    self.adj_matrix[placementunit_j][placementunit_i] = 1                   

    def OutputNodeFeature(self, output_path):
        with open(output_path, "w") as f_nf:
            output_str = ""
            for placementunit_id in range(self.node_feature_map.shape[0]):
                output_str += str(placementunit_id)
                output_str += " "
                node_feature = self.node_feature_map[placementunit_id, :].reshape(-1)
                for feature_id in range(node_feature.shape[0]):
                    output_str += str(node_feature[feature_id])
                    if feature_id != node_feature.shape[0]-1:
                        output_str += ","
                output_str += "\n"
            f_nf.write(output_str)

    def OutputNodelink(self, output_path):
        with open(output_path, "w") as f_link:
            output_str = ""
            num_placementunit = len(self.placementinfo.placementunits)
            for placementunit_i in range(num_placementunit):
                for placementunit_j in range(placementunit_i, num_placementunit):
                    if self.adj_matrix[placementunit_i][placementunit_j] == 1:
                        output_str += str(placementunit_i)
                        output_str += " "
                        output_str += str(placementunit_j)
                        output_str += "\n"
            f_link.write(output_str)
    
    def OutputPlacementUnitNode(self, output_path):
        with open(output_path, "w") as f_unit:
            output_str = ""
            for id in range(len(self.placementinfo.placementunits)):
                placementunit = self.placementinfo.placementunits[id]
                nodidcol = placementunit.getNodeSet()
                nodes = self.placementinfo.database.nodes
                output_str += ("Placement Unit "+str(id)+" Begin")
                output_str += "\n"
                for nodeid in nodidcol:
                    nodeper = nodes[nodeid]
                    output_str += nodeper.name
                    output_str += "\n"
                output_str += ("Placement Unit "+str(id)+" End")
                output_str += "\n"
            
            f_unit.write(output_str)

    def OutputPUGraph(self, output_path_dir):
        if not os.path.exists(output_path_dir):
            os.makedirs(output_path_dir)
        self.OutputNodeFeature(os.path.join(output_path_dir, "PU_feature.txt"))
        self.OutputNodelink(os.path.join(output_path_dir, "PU_link.txt"))
        self.OutputPlacementUnitNode(os.path.join(output_path_dir, "PU_info.txt"))        



