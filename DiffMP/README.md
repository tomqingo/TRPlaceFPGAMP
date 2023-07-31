The Diffusion Model is modified from [*__*DiffuSeq: Sequence to Sequence Text Generation With Diffusion Models*__*].
[github](https://github.com/Shark-NLP/DiffuSeq)
[arxiv](https://arxiv.org/abs/2210.08933)

The Transformer is modified from [*__*linear-attention-transformer*__*].
[github](https://github.com/lucidrains/linear-attention-transformer)

The Graph Neural Network is modified from [*__*Multi-task Self-supervised Graph Neural Network Enable Stronger Task Generalization*__*].
[github](https://github.com/jumxglhf/ParetoGNN)
[openreview](https://openreview.net/forum?id=1tHAZRqftM)

See the original repos for dependencies.

# Dataset preprocessing
The processed data will be saved in ./dataset.
```
cd DiffMP
python dataset_parser.py
```

# GNN pretraining
Different version of DGL is involved because the original repo uses an old version, but that doesn't work on our CUDA. The hetero graphs will be saved in ParetoGNN/MP_hetero_graphs, the links and labels will be saved in ParetoGNN/MP_links and ParetoGNN/MP_pretrain_labels. The pretrained checkpoints will be saved in ParetoGNN/scripts.
```
cd ParetoGNN

pip uninstall dgl
pip install DGL==0.9.0
python hetero_graph_gen.py
python link_gen.py

pip uninstall dgl
pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
cd scripts
bash ssnc_MP.sh 0
```

# Diffusion model training
The model checkpoints will be saved in ./diffusion_models.
```
cd ../../scripts
bash train.sh
```

# Inference
The generated results will be saved in ./generated_results
```
cd ..
cd scripts
bash run_decode.sh
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
