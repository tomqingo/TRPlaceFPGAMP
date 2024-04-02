from utils import *
from src import *
from src.MacroPl import *
import pandas as pd
import copy
import time
import os
from .check_legality import CheckLegality
import multiprocessing


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

def load_placement_calMacroHPWL(args, logger, dataset, solution_col_dir, solution_slice, MacroHPWL_q):
    MacroHPWL_list = []
    for solution_name in solution_slice:
        solution_dir = os.path.join(solution_col_dir, solution_name)
        solution_path = os.path.join(solution_col_dir, solution_name, "macroplacement.pl")
        if os.path.exists(solution_path):
            logger.info("calculate the MacroWirelength for case: "+solution_path)
            dataset_aug = copy.deepcopy(dataset)
            dataset_aug.readSamplePl(solution_path, logger)
            error_path = os.path.join(solution_dir, "PlaceError.log")
            if CheckLegality(dataset_aug, error_path, logger):
                totalMacroHPWL = dataset_aug.calMacroHPWL()
                logger.info("Macro HPWL "+str(totalMacroHPWL))
                MacroHPWL_list.append([solution_dir, totalMacroHPWL])
    MacroHPWL_q.put(MacroHPWL_list)


def load_placement_all_parallel(args, logger):
    logger.info("Run all designs in dataset %s in parallel." % args.dataset)
    mul_params = get_multiple_design_params(args.dataset_root, args.dataset)
    for i, params in enumerate(mul_params):
        logger.info("Augment "+params["design_name"])
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        # load the design information
        dataset = load_dataset(cur_args, logger)
        ## add solution path (25 designs)
        solution_col_dir = os.path.join(args.solution_dir, params["design_name"], "AMF_solution")
        solution_name_col = os.listdir(solution_col_dir)
        ## split the solution names
        solution_slice = splitfilecol2job(solution_name_col, cur_args.runs)

        jobs = []
        MacroHPWL_q = multiprocessing.Queue()
        for j in range(cur_args.runs):
            p = multiprocessing.Process(target=load_placement_calMacroHPWL, args=(cur_args, logger, dataset, solution_col_dir, solution_slice[j], MacroHPWL_q))
            jobs.append(p)
            p.start()
            time.sleep(2)
        for proc in jobs:
            proc.join()
        aug_MacroHPWL_col = []
        for job_id in jobs:
            aug_MacroHPWL_col.extend(MacroHPWL_q.get())
        aug_MacroHPWL_col = sorted(aug_MacroHPWL_col, key=(lambda x: x[0][9:]))
        logger.info("Finish augmenting "+params["design_name"])
        writehpwl_placement(args, logger, solution_col_dir, aug_MacroHPWL_col)

def writehpwl_placement(args, logger, base_dir_path, hpwl_col):
    macroHPWL_path = os.path.join(base_dir_path, "MacroHPWL.txt")
    f_hpwl = open(macroHPWL_path, "w")
    hpwl_str = ""
    for id in range(len(hpwl_col)):
        hpwl_str += hpwl_col[id][0]
        hpwl_str += " "
        hpwl_str += str(hpwl_col[id][1])
        hpwl_str += "\n"
    f_hpwl.write(hpwl_str)
    f_hpwl.close()

        










