from utils import *
from src import run_placement_single, run_placement_all
import argparse
import datetime
import sys

def get_option():
    parser = argparse.ArgumentParser("Cumple")
    parser.add_argument("--dataset_root", type=str, default="/data/ssd/qluo/benchmark", help="the parent folder of dataset")
    parser.add_argument("--dataset", type=str, default="mlcad2023", help="dataset name")
    parser.add_argument("--design_name", type=str, default="Design_105", help="design name")
    parser.add_argument("--custom_path", type=str, default="", help="custom design path, set it astoken1:path1,token2:path2 e.g. nodes:data/test.nodes,nets:data/test.nets,design_name:mydesign,benchmark:mybenchmark")
    parser.add_argument('--run_all', type=str2bool, default=False, help='If True, run all designs in the given dataset. If False, run the given design only.')
    
    parser.add_argument("--log_freq", type=int, default=100) 
    parser.add_argument("--result_dir", type=str, default="result", help="log/model root directory") 
    parser.add_argument("--exp_id", type=str, default="", help="experiment id") 
    parser.add_argument("--log_dir", type=str, default="log", help="log directory") 
    parser.add_argument("--log_name", type=str, default="test.log", help="log file name") 
    parser.add_argument("--eval_dir", type=str, default="eval", help="visualization directory")

    parser.add_argument('--random_place', type=str2bool, default=False, help='If True, randomly place macros, or place them according to sample.pl.')

    args = parser.parse_args()

    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
    #args.exp_id = "1"
    return args

def main():
    args = get_option()
    logger = setup_logger(args,sys.argv)

    if args.run_all:
        run_placement_all(args,logger)
    else:
        run_placement_single(args,logger)

if __name__ == "__main__":
    main()

