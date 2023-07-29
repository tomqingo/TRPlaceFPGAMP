Modified from [*__*DiffuSeq*__: Sequence to Sequence Text Generation With Diffusion Models*].
[github](https://github.com/Shark-NLP/DiffuSeq)
[arxiv](https://arxiv.org/abs/2210.08933)

See the original repo for dependencies.

# Dataset
```
python dataset_parser.py
```

# Training
```
cd ./scripts
bash train.sh
```

# Convert res_Design_12.txt to Design_12_solution.pl
```
python3 result_convert.py --PUInfoFile /data/ssd/qluo/benchmark/mlcad2023_v2/Design_12/netlist_feature/PU_info.txt --PULocFile /data/ssd/qluo/docker_practice/Cumple/DiffMP/result/res_Design_12.txt --PULocConvertFile /data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl
```

# Check the Legality and calculate the macro hpwl
```
python3 ../main.py --dataset_root /data/ssd/qluo/benchmark/mlcad2023_v2 --dataset mlcad2023 --design_name Design_12 --solution /data/ssd/qluo/docker_practice/Cumple/DiffMP/result/Design_12_solution.pl --run_all False --random_place False
```
# Run Vivado
If the initial check passes, we could run the vivado with the the Converted Placement Result Design_12_solution.pl
```
mkdir run_vivado
cd run_vivado
vivado -mode tcl -so ../preprocessing/placeandroute.tcl
```
