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
    # logger.info("Generate the netlist feature for:"+args.design_name)
    #placementinfo = PlacementInfo(dataset)
    #placementinfo.Convert2PlacementInfo(logger)
    # Extract the initial feature for each placement unit
    #feature_extractor = FeatureExtractor(placementinfo)
    #output_path = os.path.join(dataset.params["sample_dir"], "PU_feature.txt")
    #feature_extractor.OutputNodeFeature(output_path)
    #output_path = os.path.join(dataset.params["sample_dir"], "PU_link.txt")
    #feature_extractor.OutputNodelink(output_path)    
    #output_path = os.path.join(dataset.params["sample_dir"], "PU_info.txt")
    #feature_extractor.OutputPlacementUnitNode(output_path)
    # Run Macro Placement
    if args.random_place:
        dataset.RandomCordGenerate(logger)
    else:
        #if os.path.exists(dataset.params["sample"]):
        dataset.readSamplePl(logger)
    # Check the legality of the Placement Result
    is_legal = dataset.CheckLegality(logger)
    totalMacroHPWL = dataset.calMacroHPWL()
    logger.info("Macro HPWL: {}".format(totalMacroHPWL))
    # Output the placement result
    output_path = os.path.join(args.dataset_root, args.design_name, "macroplacement.pl")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    dataset.OutputSolutionpl(output_path)
    # draw the placement result by inherent visulization tool
    # draw_macro_placement_result(args, dataset, logger)
    design_info = (dataset.num_nodes, dataset.num_nets, dataset.num_macro, dataset.num_basic_macro, dataset.num_cascade_macro, dataset.num_cascade_node, dataset.num_fix, dataset.num_region_constr, dataset.num_region_constr_node, dataset.num_region_constr_maceonode, dataset.num_region_constr_cascademaceonode)
    resource = ["LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "URAM288", "IO"]
    design_resource_info = []
    for res_id, res_name in enumerate(resource):
        design_resource_info.append(dataset.num_resource_demand[res_name])
    design_resource_info.append(dataset.num_nodes)
    design_resource_info = tuple(design_resource_info)
    FPGA_resource_info = []
    for res_id, res_name in enumerate(resource):
        FPGA_resource_info.append(dataset.num_resource_supply[res_name])
    FPGA_resource_info.append(dataset.num_bel)
    site = ["SLICE", "DSP", "BRAM", "URAM", "IO"]
    for site_id, site_name in enumerate(site):
        FPGA_resource_info.append(dataset.num_avail_site_dict[site_name])
    FPGA_resource_info.append(dataset.num_avail_site)    
    FPGA_resource_info = tuple(FPGA_resource_info)
    if os.path.exists(dataset.params["sample"]):
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
    design_info_col = pd.DataFrame(columns=["design","#Nodes", "#Nets", "#Macros", "#BasicMacros", "#CascadeMacros", "#NodesInCascadeMacros", "FixedNode", "#RegionConstraints","#RegionConstraintNodes","#RegionConstraintMacroNodes","#RegionConstraintCascadeMacroNodes"])
    design_resource_col = pd.DataFrame(columns=["design", "LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "URAM288", "IO", "#Nodes"])
    FPGA_resource_col = pd.DataFrame(columns=["design", "LUT", "FF", "CARRY8", "DSP48E2", "RAMB36E2", "URAM288", "IO", "#BELs", "SLICE", "DSP", "BRAM", "URAM", "IO", "#Sites"])
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