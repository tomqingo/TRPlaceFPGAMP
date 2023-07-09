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
            # BRAM/DSP/URAM
            if placementunit.isBRAM():
                node_feature.extend([1,0,0])
            elif placementunit.isDSP():
                node_feature.extend([0,1,0])
            elif placementunit.isURAM():
                node_feature.extend([0,0,1])
            
            #Cascade/Single
            if placementunit.is_cascade:
                node_feature.extend([1,0])
            else:
                node_feature.extend([0,1])
            
            #Number of the cells in on placementunit
            node_feature.append(placementunit.num_cells)

            #Number of nets connecting between this macro and other cells (macros and LUT/FF)
            connmacronum = len(placementunit.placementnetidcol)
            externalconn = connmacronum + placementunit.connNonmacronum
            connNonmacronum = placementunit.connNonmacronum
            internalconn = placementunit.internalnetnum
            node_feature.extend([externalconn, internalconn, connmacronum, connNonmacronum])
            
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