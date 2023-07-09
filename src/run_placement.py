from utils import *
from src import *
from src.MacroPl import *
import pandas as pd
import copy
import os


def run_placement_main(args, logger):
    # Load Dataset
    dataset = load_dataset(args, logger)
    # Convert the dataset to the placementinfo (integrating simple and cascade macros)
    placementinfo = PlacementInfo(dataset)
    placementinfo.Convert2PlacementInfo(logger)
    # Extract the initial feature for each placement unit
    feature_extractor = FeatureExtractor(placementinfo)
    output_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, args.design_name, "out1_node_feature_label.txt")
    feature_extractor.OutputNodeFeature(output_path)
    output_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, args.design_name, "out1_graph_edges.txt")
    feature_extractor.OutputNodelink(output_path)    
    # Run Macro Placement
    if args.random_place:
        dataset.RandomCordGenerate(logger)
    else:
        dataset.readSamplePl(logger)
    # Check the legality of the Placement Result
    is_legal = dataset.CheckLegality(logger)
    # Output the placement result
    output_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, args.design_name, "solution.pl")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    dataset.OutputSolutionpl(output_path)
    # draw the placement result by inherent visulization tool
    totalMacroHPWL = dataset.calMacroHPWL()
    logger.info("Macro HPWL: {}".format(totalMacroHPWL))
    draw_macro_placement_result(args, dataset, logger)
    design_info = (dataset.num_nodes, dataset.num_nets, dataset.num_macro, dataset.num_basic_macro, dataset.num_cascade_macro, dataset.num_fix, dataset.num_region_constr, dataset.num_region_constr_node)
    resource = ["LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "URAM288", "IO"]
    design_resource_info = []
    for res_id, res_name in enumerate(resource):
        design_resource_info.append(dataset.num_resource_demand[res_name])
    design_resource_info.append(dataset.num_nodes)
    design_resource_info = tuple(design_resource_info)
    FPGA_resource_info = []
    for res_id, res_name in enumerate(resource):
        FPGA_resource_info.append(dataset.num_resource_supply[res_name])
    FPGA_resource_info.append(dataset.num_avail_site)    
    FPGA_resource_info = tuple(FPGA_resource_info)
    return design_info, design_resource_info, FPGA_resource_info
    

def run_placement_single(args, logger):
    logger.info("=================")
    logger.info("Start place %s/%s" % (args.dataset , args.design_name))
    res = run_placement_main(args, logger)
    return res

def run_placement_all(args, logger):
    logger.info("Run all designs in dataset %s." % args.dataset)
    design_info_col = pd.DataFrame(columns=["design","#Nodes", "#Nets", "#Macros", "#BasicMacros", "#CascadeMacros", "FixedNode", "#RegionConstraints","#RegionConstraintNodes"])
    design_resource_col = pd.DataFrame(columns=["design", "LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "URAM288", "IO", "#Nodes"])
    FPGA_resource_col = pd.DataFrame(columns=["design", "LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "URAM288", "IO", "#Sites"])
    mul_params = get_multiple_design_params(args.dataset_root, args.dataset)
    for i, params in enumerate(mul_params):
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        design_info, design_resource_info, FPGA_resource_info = run_placement_single(cur_args, logger)       
        design_info_col.loc[i] = [cur_args.design_name, *design_info]
        design_resource_col.loc[i] = [cur_args.design_name, *design_resource_info]
        FPGA_resource_col.loc[i] = [cur_args.design_name, *FPGA_resource_info]        
    design_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "design_info.csv")
    design_info_col.to_csv(design_csv_path)
    design_res_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "design_res_info.csv")
    design_resource_col.to_csv(design_res_csv_path)
    FPGA_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "FPGA_info.csv")
    FPGA_resource_col.to_csv(FPGA_csv_path)
    print(design_info_col)
    print(design_resource_col)
    print(FPGA_resource_col)