from src import *
from utils import *
import copy
import math
import time
import multiprocessing


def augment_data(args, logger, dataset, augment_pos_list, augment_neg_list, MacroHPWL_q):
    input_path = dataset.params["sample_dir"]
    MacroHPWL_list = []
    for sample_cnt in augment_pos_list:
        logger.info("generating the augmented benchmark"+str(sample_cnt))
        dataset_aug = copy.deepcopy(dataset)
        dataset_aug.RandomAugment(args.augment_small_range, args.augment_small_per, logger)
        output_path = input_path + "solution_gt_"+str(sample_cnt)+".pl"
        #print(output_path)
        if dataset_aug.CheckLegality(logger):
            totalMacroHPWL = dataset_aug.calMacroHPWL()
            logger.info("Macro HPWL "+str(totalMacroHPWL))
            dataset_aug.OutputSolutionpl(output_path)
            MacroHPWL_list.append([sample_cnt, totalMacroHPWL])
    
    for sample_cnt in augment_neg_list:
        logger.info("generating the augmented benchmark"+str(sample_cnt))
        dataset_aug = copy.deepcopy(dataset)
        dataset_aug.RandomAugment(args.augment_large_range, args.augment_large_per, logger)
        output_path = input_path + "solution_gt_"+str(sample_cnt)+".pl"
        if dataset_aug.CheckLegality(logger):
            totalMacroHPWL = dataset_aug.calMacroHPWL()
            logger.info("Macro HPWL "+str(totalMacroHPWL))      
            dataset_aug.OutputSolutionpl(output_path)
            MacroHPWL_list.append([sample_cnt, totalMacroHPWL])
    
    MacroHPWL_q.put(MacroHPWL_list)

def augment_single_data(args, logger):
    logger.info("Augment all designs in dataset %s." % args.dataset)
    dataset = load_dataset(args, logger)
    dataset.readSamplePl(logger)
    totalMacroHPWL_gt = dataset.calMacroHPWL()
    logger.info("Macro HPWL for vivado case:"+str(totalMacroHPWL_gt))
    aug_file_num = list(range(0, args.augment_pos_num + args.augment_neg_num))
    aug_pos_file_num = aug_file_num[0:args.augment_pos_num]
    aug_neg_file_num = aug_file_num[args.augment_pos_num:]
    MacroHPWL_q = multiprocessing.Queue()
    augment_data(args, logger, dataset, aug_pos_file_num, aug_neg_file_num, MacroHPWL_q)
    aug_MacroHPWL_col = []
    aug_MacroHPWL_col.extend(MacroHPWL_q.get())
    aug_MacroHPWL_col = sorted(aug_MacroHPWL_col, key=(lambda x: x[0]))
    logger.info("Finish augment design"+dataset.params["design_name"])
    input_path = dataset.params["sample_dir"]
    writehpwl(args, logger, input_path, aug_MacroHPWL_col, totalMacroHPWL_gt)


def writehpwl(args, logger, base_dir_path, hpwl_col, hpwl_gt):
    macroHPWL_path = base_dir_path + "MacroHPWL.txt"
    f_hpwl = open(macroHPWL_path, "w")
    hpwl_str = base_dir_path + "solution_gt "
    hpwl_str += str(hpwl_gt)
    hpwl_str += "\n"
    for id in range(len(hpwl_col)):
        hpwl_str += (base_dir_path + "solution_gt_" + str(hpwl_col[id][0]))
        hpwl_str += " "
        hpwl_str += str(hpwl_col[id][1])
        hpwl_str += "\n"
    f_hpwl.write(hpwl_str)
    f_hpwl.close()

def splitfilecol2job(aug_file_num, njobs):
    aug_file_num_slice = []
    num_file_per_slice = math.ceil(len(aug_file_num)*1.0/njobs)
    for id in range(njobs):
        if id == njobs-1:
            aug_file_num_slice.append(aug_file_num[id*num_file_per_slice:])
        else:
            aug_file_num_slice.append(aug_file_num[id*num_file_per_slice:(id+1)*num_file_per_slice])
    return aug_file_num_slice

def augment_all_data_parallel(args, logger):
    logger.info("Augment all designs in dataset %s." % args.dataset)
    mul_params = get_multiple_design_params(args.dataset_root, args.dataset)
    for i,params in enumerate(mul_params):
        logger.info("Augment"+params["design_name"])
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        dataset = load_dataset(cur_args, logger)
        dataset.readSamplePl(logger)
        totalMacroHPWL_gt = dataset.calMacroHPWL()
        logger.info("Macro HPWL for vivado case:"+str(totalMacroHPWL_gt))
        aug_file_num = list(range(0, args.augment_pos_num + args.augment_neg_num))        
        aug_pos_file_num = aug_file_num[0:args.augment_pos_num]
        aug_neg_file_num = aug_file_num[args.augment_pos_num:]
        aug_pos_slice = splitfilecol2job(aug_pos_file_num, cur_args.runs)
        aug_neg_slice = splitfilecol2job(aug_neg_file_num, cur_args.runs)
        jobs = []
        MacroHPWL_q = multiprocessing.Queue()
        for j in range(cur_args.runs):
            p = multiprocessing.Process(target=augment_data, args=(cur_args, logger, dataset, aug_pos_slice[j], aug_neg_slice[j], MacroHPWL_q))
            jobs.append(p)
            p.start()
            time.sleep(2)
        for proc in jobs:
            proc.join()
        aug_MacroHPWL_col = []
        for job_id in jobs:
            aug_MacroHPWL_col.extend(MacroHPWL_q.get())
        aug_MacroHPWL_col = sorted(aug_MacroHPWL_col, key=(lambda x: x[0]))
        logger.info("Finish augment design"+params["design_name"])
        input_path = dataset.params["sample_dir"]
        writehpwl(cur_args, logger, input_path, aug_MacroHPWL_col, totalMacroHPWL_gt)
        


    

