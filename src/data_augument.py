from src import *
from utils import *
import copy

def augment_single_data(args, logger):
    dataset = load_dataset(args, logger)
    dataset.readSamplePl(logger)
    input_path = dataset.params["sample_dir"]
    macroHPWL_path = input_path + "MacroHPWL.txt"
    f_hpwl = open(macroHPWL_path, "w")
    hpwl_str = ""
    totalMacroHPWL = dataset.calMacroHPWL()
    logger.info("Macro HPWL for vivado case:"+str(totalMacroHPWL))
    hpwl_str += dataset.params["sample"]
    hpwl_str += " "
    hpwl_str += str(totalMacroHPWL)
    hpwl_str += "\n"
    sample_cnt = 0
    for id in range(args.augment_pos_num):
        logger.info("generating the augmented benchmark"+str(sample_cnt))
        dataset_aug = copy.deepcopy(dataset)
        dataset_aug.RandomAugment(args.augment_small_range, args.augment_small_per, True, logger)
        output_path = input_path + "solution_gt_"+str(sample_cnt)+".pl"
        #print(output_path)
        if dataset_aug.CheckLegality(logger):
            totalMacroHPWL = dataset_aug.calMacroHPWL()
            logger.info("Macro HPWL"+str(totalMacroHPWL))
            dataset_aug.OutputSolutionpl(output_path)
            hpwl_str += output_path
            hpwl_str += " "
            hpwl_str += str(totalMacroHPWL)
            hpwl_str += "\n"
            sample_cnt = sample_cnt + 1
    
    for id in range(args.augment_neg_num):
        logger.info("generating the augmented benchmark"+str(sample_cnt))
        dataset_aug = copy.deepcopy(dataset)
        dataset_aug.RandomAugment(args.augment_large_range, args.augment_large_per, False, logger)
        output_path = input_path + "solution_gt_"+str(sample_cnt)+".pl"
        if dataset_aug.CheckLegality(logger):
            totalMacroHPWL = dataset_aug.calMacroHPWL()
            logger.info("Macro HPWL"+str(totalMacroHPWL))      
            dataset_aug.OutputSolutionpl(output_path)
            hpwl_str += output_path
            hpwl_str += " "
            hpwl_str += str(totalMacroHPWL)
            hpwl_str += "\n"
            sample_cnt = sample_cnt + 1
    
    f_hpwl.write(hpwl_str)
    f_hpwl.close()
    

