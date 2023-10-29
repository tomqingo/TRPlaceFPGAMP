from utils import *
from src import *
from src.MacroPl import *
import pandas as pd
import copy
import os
from .check_legality import CheckLegality


def run_placement_main(args, logger):
    # Load Dataset
    dataset = load_dataset(args, logger)
    # Read the macro placement result or run Macro Placement
    logger.info("====Macro Placement====")
    if args.random_place:
         dataset.RandomCordGenerate(logger)
    else:
        solution_file_path = os.path.join(args.solution_dir, args.design_name, "place_results", "macroplacement.pl")
        if os.path.exists(solution_file_path):
           dataset.readSamplePl(solution_file_path, logger)
    # Check the legality of the Placement Result
    log_dir = os.path.join(args.result_dir, args.exp_id, args.log_dir, args.design_name)
    is_legal = CheckLegality(dataset, log_dir, logger)
    totalMacroHPWL = dataset.calMacroHPWL()
    logger.info("Macro HPWL: {}".format(totalMacroHPWL))

    # Whether to output the placement results
    if args.output_pl:
        if args.output_dir == "":
            output_path = os.path.join(log_dir, "place_results", "macroplacement.pl")
        else:
            output_path = os.path.join(args.output_dir, args.design_name, "place_results", "macroplacement.pl")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        dataset.OutputSolutionpl(output_path)
    # draw the placement result by inherent visulization tool
    if args.visualize:
        draw_macro_placement_result(args, dataset, logger)
    # output the design statistics
    design_info = (dataset.num_nodes, dataset.num_nets, dataset.num_macro, dataset.num_basic_macro, dataset.num_cascade_macro, dataset.num_cascade_node, dataset.num_fix, dataset.num_clk_nets, dataset.num_high_degree_nets, dataset.min_region_constr, dataset.max_region_constr, dataset.avg_region_constr, dataset.num_region_constr, dataset.num_region_constr_node, dataset.num_region_constr_macronode, dataset.num_region_constr_cascademacronode)
    resource = ["LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "IO"]
    design_resource_info = []
    for res_id, res_name in enumerate(resource):
        design_resource_info.append(dataset.num_resource_demand[res_name])
    design_resource_info.append(dataset.num_nodes)
    design_resource_info = tuple(design_resource_info)
    FPGA_resource_info = []
    for res_id, res_name in enumerate(resource):
        FPGA_resource_info.append(dataset.num_resource_supply[res_name])
    FPGA_resource_info.append(dataset.num_bel)
    site = ["SLICE", "DSP", "BRAM", "IO"]
    for site_id, site_name in enumerate(site):
        FPGA_resource_info.append(dataset.num_site_dict[site_name])
    FPGA_resource_info.append(dataset.num_site)    
    FPGA_resource_info = tuple(FPGA_resource_info)
    if is_legal:
        place_info = (int(is_legal), totalMacroHPWL)
    else:
        place_info = (-1, -1)
    return design_info, design_resource_info, FPGA_resource_info, place_info
    

def run_placement_single(args, logger):
    logger.info("=================")
    logger.info("Start place %s/%s" % (args.dataset , args.design_name))
    res = run_placement_main(args, logger)
    return res

def run_placement_all(args, logger):
    logger.info("Run all designs in dataset %s." % args.dataset)
    design_info_col = pd.DataFrame(columns=["design","#Nodes", "#Nets", "#Macros", "#BasicMacros", "#CascadeMacros", "#NodesInCascadeMacros", "FixedNode", "#CLK_nets", "#HighFanout_nets", "#MinRegionConstrArea", "#MaxRegionConstrArea", "#AvgRegionConstrArea", "#RegionConstraints","#RegionConstraintNodes","#RegionConstraintMacroNodes","#RegionConstraintCascadeMacroNodes"])
    design_resource_col = pd.DataFrame(columns=["design", "LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "IO", "#Nodes"])
    FPGA_resource_col = pd.DataFrame(columns=["design", "LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "IO", "#BELs", "SLICE", "DSP", "BRAM", "IO", "#Sites"])
    place_col = pd.DataFrame(columns=["design", "is_legal", "Macro_HPWL"])
    mul_params = get_multiple_design_params(args.dataset_root, args.dataset)
    for i, params in enumerate(mul_params):
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        design_info, design_resource_info, FPGA_resource_info, place_info = run_placement_single(cur_args, logger)       
        design_info_col.loc[i] = [cur_args.design_name, *design_info]
        design_resource_col.loc[i] = [cur_args.design_name, *design_resource_info]
        FPGA_resource_col.loc[i] = [cur_args.design_name, *FPGA_resource_info]
        place_col.loc[i] = [cur_args.design_name, *place_info]        
    design_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "design_info.csv")
    design_info_col.to_csv(design_csv_path)
    design_res_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "design_res_info.csv")
    design_resource_col.to_csv(design_res_csv_path)
    FPGA_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "FPGA_info.csv")
    FPGA_resource_col.to_csv(FPGA_csv_path)
    place_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "place.csv")
    place_col.to_csv(place_path)
    print(design_info_col)
    print(design_resource_col)
    print(FPGA_resource_col)
    print(place_col)
