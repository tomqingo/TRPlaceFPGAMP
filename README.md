CUMPLE: FPGA Macro Placement Tool

# Input file format

design.nodes <br>
design.nets <br>
design.lib <br>
design.scl <br>
design.pl <br>
design.cascade_shape <br>
design.cascade_shape_instances <br>
design.regions <br>
sample.pl

# Macro Placement Output file format
solution.pl <br>

# Overall Flow

1. Read the benchmark (Two modes: All designs in the benchmark or one design)

2. Macro Placement (Implement the macro placement using the online Reinforce Learning)

3. Validate the legality of the placement result (The resource, coordinate, macro shape, and regional requirements)

4. Output the solution.pl file

# Argument Settings
~~~
options:
--dataset_root  [optional]      string	the relative path or absolute path to the parent folder of the benchmark
--dataset       [optional]		string 	dataset name
--design_name	[optional]		string	design name
--custom_path   [optional]      string  custom design path, set it as token1:path1,token2:path2 e.g. nodes:data/test.nodes,nets:data/test.nets,design_name:mydesign,benchmark:mybenchmark
--run_all		[optional]	    str2bool  If True, run all designs in the given dataset. If False, run the given design only.
--log_freq		[optional]      int     the logging frequency	    
--result_dir    [optional]	    string  log/model root directory
--exp_id        [optional]      string  experiment id
--log_dir       [optional]      string  log directory
--log_name      [optional]      string  log file name
--eval_dir      [optional]      string  visualization directory
--random_place  [optional]      str2bool If True, randomly place macros, or place them according to sample.pl.
--is_training   [optional]      str2bool Whether we train/test a RL model for the macro placement
--batch_size    [optional]      int      Online RL training batch size
--epochs        [optional]      int      The training iterations
--lr            [optional]      float    Learning rate
--is_test       [optional]      str2bool Whether we test the RL Model
~~~

# Directory in the repo

result: the macro placement results, placement error and the log file

preprocessing: some preprocessing tcl and python file (like extract the macro and convert the macro placement solution)

DiffMP: diffusion model for macro placement

src: source code
- db  Some data structures used to represent the nodes, nets, macros, and sitemaps.
- MacroPl: To do, intended for the final selected framework
- model: Policy and Value Network
- place_env: placement enironment for online RL

thirdparty: Some extra tools we would use in the macro placement

utils: Some extra function like logger, get the params of the benchmark

# How to run
~~~
$ python3 main.py --dataset_root <benchmark_root> --dataset <dataset_name> --design_name <design_name> --solution <pathToSolution> --run_all <run_all_designs_flag> --random_place <run_randomplace_or_samplePL_flag>
~~~
If we want to run the Design_12 in mlcad2023_v2 using the macro placement solution in /data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl
~~~
$ python3 main.py --dataset_root /data/ssd/qluo/benchmark/mlcad2023_v2 --dataset mlcad2023 --design_name Design_12 --solution /data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl --run_all False --random_place False
~~~
If we want to run all the cases in mlcad2023_v2 using randomly placement
~~~
$ python3 main.py --dataset_root /data/ssd/qluo/benchmark/mlcad_v2 --dataset mlcad2023 --run_all True --random_place True
~~~
If we want to train the online RL model on Design_2
~~~
$  python3 main.py --is_training True --is_test False --epoch 30000 --design_name Design_2
~~~
If we want to test the online RL model after training on Design_2
~~~
$ python3 main.py --is_training True --is_test True --epoch 1 --design_name Design_2 --checkpoint_path save_models/Design_2/net_dict-Design_2-2320-2024-04-01-03-53-38-2788503.pkl
~~~

### Dependencies

* Python (version 3)
* Pandas>=1.12.0
* Numpy>=1.19.2
* Gym>=0.18.0