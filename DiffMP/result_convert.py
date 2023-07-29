import os
import argparse

def get_option():
    parser = argparse.ArgumentParser("Convert the Result File")
    parser.add_argument("--PUInfoFile", type=str, default="/data/ssd/qluo/benchmark/mlcad2023_v2/Design_12/netlist_feature/PU_info.txt", help="the parent folder of dataset")
    parser.add_argument("--PULocFile", type=str, default="/data/ssd/qluo/docker_practice/Cumple/DiffMP/result/res_Design_12.txt", help="dataset name")
    parser.add_argument("--PULocConvertFile", type=str, default="/data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl", help="design name")

    args = parser.parse_args()

    #args.exp_id = "random"
    return args

def ConvertId2ReferenceName(PUInfoFile):
    Id2ReferenceNamedict = {}
    with open(PUInfoFile, "r") as f_puinfo:
        all_lines = f_puinfo.read().splitlines()
        unit_id = 0
        for line_id in range(len(all_lines)):
            cur_line = all_lines[line_id]
            print(cur_line)
            cur_line_col = cur_line.strip().split()
            if len(cur_line_col) > 1:
                if cur_line_col[0] == "Placement" and (cur_line_col[1] == "Unit" and cur_line_col[-1] == "Begin"):
                    Id2ReferenceNamedict[unit_id] = all_lines[line_id+1]
                    unit_id += 1
                else:
                    continue
            else:
                continue
    return Id2ReferenceNamedict

def OutputConvetFile(PULocFile, PUConvertLocFile, Id2ReferenceNamedict):
    output_str = ""
    with open(PULocFile, "r") as f_loc:
        all_lines = f_loc.read().splitlines()
        for line_id in range(len(all_lines)):
            cur_line = all_lines[line_id]
            cur_line_col = cur_line.strip().split()
            cur_macro_id = int(cur_line_col[0])
            cur_refnode_name = Id2ReferenceNamedict[cur_macro_id]
            cur_macro_loc = cur_line_col[1]
            cur_macro_locX = cur_line_col[1].strip().split(",")[0]
            cur_macro_locY = cur_line_col[1].strip().split(",")[1]
            bel_name = "0"
            output_str += cur_refnode_name
            output_str += " "
            output_str += cur_macro_locX
            output_str += " "
            output_str += cur_macro_locY
            output_str += " "
            output_str += bel_name
            output_str += "\n"
    
    with open(PUConvertLocFile, "w") as f_locconv:
        f_locconv.write(output_str)




if __name__ == "__main__":
    args = get_option()
    PUInfoFile = args.PUInfoFile
    PULocFile = args.PULocFile
    PULocConvertFile = args.PULocConvertFile
    Id2ReferenceNamedict = ConvertId2ReferenceName(PUInfoFile)
    OutputConvetFile(PULocFile, PULocConvertFile, Id2ReferenceNamedict)

