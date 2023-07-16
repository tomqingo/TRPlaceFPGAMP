from utils import *
from src import run_placement_single, run_placement_all, augment_single_data, augment_all_data_parallel
import argparse
import datetime
import sys
import os

def get_option():
    parser = argparse.ArgumentParser("Cumple")
    parser.add_argument("--dataset_root", type=str, default="/data/ssd/qluo/benchmark/mlcad2023_v2", help="the parent folder of dataset")
    parser.add_argument("--dataset", type=str, default="mlcad2023", help="dataset name")
    parser.add_argument("--design_name", type=str, default="Design_2", help="design name")
    parser.add_argument("--custom_path", type=str, default="", help="custom design path, set it astoken1:path1,token2:path2 e.g. nodes:data/test.nodes,nets:data/test.nets,design_name:mydesign,benchmark:mybenchmark")
    parser.add_argument('--run_all', type=str2bool, default=False, help='If True, run/augment all designs in the given dataset. If False, run the given design only.')
    parser.add_argument('--runs', type=int, default=8, help='The number of the threads used in the program.')

    parser.add_argument("--log_freq", type=int, default=100) 
    parser.add_argument("--result_dir", type=str, default="result", help="log/model root directory") 
    parser.add_argument("--exp_id", type=str, default="", help="experiment id") 
    parser.add_argument("--log_dir", type=str, default="log", help="log directory") 
    parser.add_argument("--log_name", type=str, default="test.log", help="log file name") 
    parser.add_argument("--eval_dir", type=str, default="eval", help="visualization directory")

    parser.add_argument("--random_place", type=str2bool, default=True, help="If True, randomly place macros, or place them according to sample.pl.")
    parser.add_argument("--augument", type=str2bool, default=False, help="If True, randomly augment the benchmarks from solution_gt.pl.")
    parser.add_argument("--augment_pos_num", type=int, default=20, help="Augment positive sample number")
    parser.add_argument("--augment_neg_num", type=int, default=37, help="Augment negative sample number")
    parser.add_argument("--augment_small_range", type=int, default=50, help="The small range for augumentation")
    parser.add_argument("--augment_large_range", type=int, default=100, help="The large range for augmentation")
    parser.add_argument("--augment_small_per", type=int, default=0.5, help="The small range for augumentation")
    parser.add_argument("--augment_large_per", type=int, default=0.7, help="The small range for augumentation")

    args = parser.parse_args()

    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
    #args.exp_id = "augment_3"
    return args

def main():
    args = get_option()
    logger = setup_logger(args,sys.argv)

    logger.info("=================")
    logger.info("   CUMPLE Tool   ")
    logger.info("=================")

    if args.augument:
        if args.run_all:
            #augment_all_data(args, logger)
            augment_all_data_parallel(args, logger)
        else:
            augment_single_data(args, logger)
    else:
        if args.run_all:
            run_placement_all(args,logger)
        else:
            run_placement_single(args,logger)

if __name__ == "__main__":
    main()

