import sys
sys.path.append('/research/d1/gds/qluo22/Cumple/DiffMP/ParetoGNN/')

from src.options import Options
import os
import torch
import dgl
from pathlib import Path
import numpy as np
import src.utils
from src.model import PretrainModule, BigModel, GCN
import json
import src.data
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import src.min_norm_solvers
from src.data_train import Graph_Dataset, Universal_Collator




# def get_node_emb(rank, world_size, opt):

#     def forward(model, checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         model.load_state_dict(checkpoint)
#         model.eval()
#         with torch.no_grad():
#             if opt.no_self_loop:
#                 use_g = dgl.add_self_loop(g)
#             else:
#                 use_g = g
#             if opt.is_distributed:
#                 X = model.module.compute_representation(use_g.to(opt.device), g.ndata['feat'].to(opt.device))
#             else:
#                 X = model.compute_representation(use_g.to(opt.device), g.ndata['feat'].to(opt.device))

#             print(X)
#             print(X.shape)

#     if opt.is_distributed:
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '12355'
#         opt.local_rank = rank
#         if opt.local_rank == 0:
#             opt.is_main = True
#         else:
#             opt.is_main = False
#         opt.device = "cuda:{}".format(opt.local_rank)
#         opt.world_size = world_size
#         torch.distributed.init_process_group(backend="nccl", world_size=opt.world_size, rank=opt.local_rank)
#         torch.cuda.set_device(opt.local_rank)
#     else:
#         opt.device = "cuda:0"
#         opt.is_main = True
#         opt.local_rank = 0
#         opt.world_size = 1

#     if opt.wandb and opt.is_main:
#         import wandb
#         name = '{}_{}_{}_{}_{}_{}_{}_{}'.format(opt.dataset, str(opt.tasks), str(opt.hid_dim), str(opt.n_layer), \
#              str(opt.total_steps), 'saint' if opt.use_saint else 'k-order', \
#              str(opt.lr), str(opt.weight_decay))
#         if opt.mask_edge:
#             name += '_{}'.format('mask_edge')
#         wandb.init(project="ParetoGNN")
#         wandb.config = opt
        
#     np.random.seed(opt.seed+opt.local_rank)
#     dgl.seed(opt.seed+opt.local_rank)
#     torch.manual_seed(opt.seed+opt.local_rank)
#     checkpoint_path = Path(opt.checkpoint_dir)/opt.name
#     checkpoint_path.mkdir(parents=True, exist_ok=True)
#     logger = src.utils.init_logger(
#         opt.is_main,
#         opt.is_distributed, # is_distributed=
#         checkpoint_path / 'run.log'
#     )
#     opt.checkpoint_path = checkpoint_path

#     logger.info(f"Initializing Data..")

#     g = src.data.load_data(opt.dataset, opt.pretrain_label_dir, opt.mask_edge, opt.tvt_addr, opt.split, hetero_graph_path=opt.hetero_graph_path)

#     logger.info(f"Initializing Model..")
        
#     node_module = GCN(g.ndata['feat'].shape[1], opt.hid_dim, opt.dropout, opt.norm, opt.use_prelu)

#     bigM = BigModel(node_module, None, opt.inter_dim)
#     ParetoGNN = PretrainModule(bigM, opt.predictor_dim).to(opt.device)
#     ParetoGNN_config = {'input_dim':g.ndata['feat'].shape[1], 'hid_dim':opt.hid_dim, 
#                 'n_layer':len(opt.hid_dim), 'inter_dim':opt.inter_dim, 'dropout':opt.dropout}
#     opt.ParetoGNN_config = ParetoGNN_config
#     logger.info("ParetoGNN CONFIG: "+json.dumps(ParetoGNN_config, indent=2))
#     model = ParetoGNN.to(opt.device)

#     if opt.is_distributed:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, static_graph=True)

#     checkpoint_path = '/research/d4/gds/qjwang21/DiffuSeq_MP/ParetoGNN/scripts/chameleon/experiment_name/checkpoint/step-10000_ssnc/model.pth.tar'
#     print('Loading model from: ', checkpoint_path)
#     logger.info("Start Farward Pass")
#     forward(model, checkpoint_path)

# def Get_Node_Emb():
#     print(torch.cuda.is_available())
#     options = Options()
#     options.add_ParetoGNN_options()
#     options.add_optim_options()
#     opt = options.parse()
#     print('opt:', opt)
#     world_size = opt.world_size
#     if opt.is_distributed:
#         mp.spawn(
#             get_node_emb,
#             args=(world_size, opt),
#             nprocs=world_size,
#             start_method='spawn',
#             join=True
#         )
#     else:
#         get_node_emb(0, 1, opt)


# if __name__ == '__main__':
#     print(torch.cuda.is_available())
#     options = Options()
#     options.add_ParetoGNN_options()
#     options.add_optim_options()
#     opt = options.parse()
#     print('opt:', opt)
#     world_size = opt.world_size
#     if opt.is_distributed:
#         mp.spawn(
#             get_node_emb,
#             args=(world_size, opt),
#             nprocs=world_size,
#             start_method='spawn',
#             join=True
#         )
#     else:
#         get_node_emb(0, 1, opt)


def get_node_emb(checkpoint_path, design_name='MacroPlacement'):

    # print("Loading Netlist Data...")
    pretrain_label_dir = 'ParetoGNN/MP_pretrain_labels'
    mask_edge = False
    tvt_addr = None
    split = 'random'
    hetero_graph_path = 'ParetoGNN/MP_hetero_graphs'
    # get the embedding
    g = src.data.load_data(design_name, 
        pretrain_label_dir = pretrain_label_dir, 
        mask_edge = mask_edge, 
        tvt_addr = tvt_addr, 
        split = split, 
        hetero_graph_path = hetero_graph_path)
    
    hid_dim = [512, 256]
    dropout = 0.
    norm = 'batch'
    use_prelu = True
    node_module = GCN(g.ndata['feat'].shape[1], hid_dim, dropout, norm, use_prelu)

    inter_dim = 0
    bigM = BigModel(node_module, None, inter_dim)

    predictor_dim = 512 ### No use
    device = 'cuda:0'
    ParetoGNN = PretrainModule(bigM, predictor_dim).to(device)

    model = ParetoGNN

    checkpoint_path = checkpoint_path
    # print('Loading parameter from: ', checkpoint_path)
    # print("Start Farward Pass...")
    node_emb = forward(g, model, checkpoint_path, device)

    return node_emb

    
def forward(g, model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        use_g = g
        X = model.compute_representation(use_g.to(device), g.ndata['feat'].to(device))

        # print(X)
        # print(X.shape)

        return X

if __name__ == '__main__':
    checkpoint_path = './scripts/chameleon/experiment_name/checkpoint/step-10000_ssnc/model.pth.tar'
    exp_name = 'chameleon'
    get_node_emb(checkpoint_path, exp_name)

    # save_emb = True
    # if save_emb:
