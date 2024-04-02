from utils import *
from src import run_placement_single, run_placement_all, augment_single_data, augment_all_data_parallel, load_placement_all_parallel
from src import train_model
import argparse
import datetime
import sys
import os


def get_option():
    parser = argparse.ArgumentParser("Cumple")
    parser.add_argument("--dataset_root", type=str, default="benchmarks/mlcad2023_v2", help="the parent folder of dataset")
    parser.add_argument("--dataset", type=str, default="mlcad2023", help="dataset name")
    parser.add_argument("--design_name", type=str, default="Design_2", help="design name")
    parser.add_argument("--custom_path", type=str, default="", help="custom design path, set it astoken1:path1,token2:path2 e.g. nodes:data/test.nodes,nets:data/test.nets,design_name:mydesign,benchmark:mybenchmark")
    parser.add_argument('--run_all', type=str2bool, default=False, help='If True, run/augment all designs in the given dataset. If False, run the given design only.')
    parser.add_argument('--runs', type=int, default=1, help='The number of the threads used in the program.')

    parser.add_argument("--log_freq", type=int, default=100) 
    parser.add_argument("--result_dir", type=str, default="result", help="log/model root directory") 
    parser.add_argument("--exp_id", type=str, default="", help="experiment id") 
    parser.add_argument("--log_dir", type=str, default="", help="log directory") 
    parser.add_argument("--log_name", type=str, default="test.log", help="log file name") 

    # Placement or argument 
    parser.add_argument("--random_place", type=str2bool, default=False, help="If True, randomly place macros, or otherwise place them according to macroplacement.pl file.")
    parser.add_argument('--solution_dir', type=str, default="/uac/gds/qluo22/disk/AMFPlacer-MLCAD/run/debug", help='The dir to the placement solution read from')    
    parser.add_argument("--augument", type=str2bool, default=False, help="If True, randomly augment the benchmarks from solution_gt.pl.")
    parser.add_argument("--feature_extract", type=str2bool, default=False, help="Whether to extract the initial feature from the netlist")

    parser.add_argument("--augment_pos_num", type=int, default=20, help="Augment positive sample number")
    parser.add_argument("--augment_neg_num", type=int, default=37, help="Augment negative sample number")
    parser.add_argument("--augment_small_range", type=int, default=50, help="The small range for augumentation")
    parser.add_argument("--augment_large_range", type=int, default=100, help="The large range for augmentation")
    parser.add_argument("--augment_small_per", type=int, default=0.5, help="The small range for augumentation")
    parser.add_argument("--augment_large_per", type=int, default=0.7, help="The small range for augumentation")

    # visualize
    parser.add_argument("--visualize", type=str2bool, default=False, help="Whether we need to visualize")
    parser.add_argument("--eval_dir", type=str, default="eval", help="visualization directory")

    # output placement result
    parser.add_argument("--output_pl", type=str2bool, default=False, help="Whether we need to output the placement result")
    parser.add_argument("--output_dir", type=str, default="", help="The folder to save the placement result")

    # Training mode
    parser.add_argument("--is_training", type=str2bool, default=False, help="Whether we would initialize the training")
    parser.add_argument(
    '--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42, metavar='N', help='random seed (default: 0)')
    parser.add_argument('--disable_tqdm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2.5e-3)
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='interval between training status logs (default: 10)')
    parser.add_argument('--pnm', type=int, default=128)
    # default benchmark
    parser.add_argument('--soft_coefficient', type=float, default = 1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--is_test',  type=str2bool, default=False)
    parser.add_argument('--save_fig', type=str2bool, default=False)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--epochs', type=int, default=30000)

    args = parser.parse_args()

    return args

def main():
    args = get_option()
    logger = setup_logger(args,sys.argv)

    logger.info("=================")
    logger.info("   CUMPLE Tool   ")
    logger.info("=================")

    if args.is_training:
        train_model(args, logger)
    else:
        if args.augument:
            if args.run_all:
                augment_all_data_parallel(args, logger)
            else:
                augment_single_data(args, logger)
        else:
            if args.run_all:
                if args.runs == 1:
                    run_placement_all(args,logger)
                else:
                    load_placement_all_parallel(args, logger)
            else:
                run_placement_single(args,logger)


if __name__ == "__main__":
    main()

