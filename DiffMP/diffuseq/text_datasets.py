from ParetoGNN.only_forward import get_node_emb
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append('..')
sys.path.append('../ParetoGNN')


def load_data_text(
    batch_size,
    split='train',
    loop=True,
    metric_enc=None,  ### encode metric (1->hidden size)
    place_enc=None,  ### encode solution (1->hidden size)
    region_enc=None,  ### encode constraint region (4->hidden size)
    device=None,
):

    if split == 'train':
        print('#'*30, '\nLoading data for training...')

        train_data, val_data = get_corpus(split=split)

        train_dataset = TextDataset(
            train_data,
            metric_enc=metric_enc,
            place_enc=place_enc,
            region_enc=region_enc,
            device=device,
        )

        val_dataset = TextDataset(
            val_data,
            metric_enc=metric_enc,
            place_enc=place_enc,
            region_enc=region_enc,
            device=device,
        )

        print('#Data in the training set:', train_dataset.__len__())
        print('#Data in the validation set:', val_dataset.__len__())

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_data_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        if loop:
            return infinite_loader(train_data_loader), infinite_loader(val_data_loader)
        else:
            return iter(train_data_loader), iter(val_data_loader)
    
    else:
        print('#'*30, '\nLoading data for testing...')

        test_data = get_corpus(split=split)
        test_dataset = TextDataset(
            test_data,
            metric_enc=metric_enc,
            place_enc=place_enc,
            region_enc=region_enc,
            device=device,
        )
        print('#Data in the test set:', test_dataset.__len__())
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        if loop:
            return infinite_loader(test_data_loader)
        else:
            return iter(test_data_loader)


def infinite_loader(data_loader):
    while True:
        yield from data_loader


def get_corpus(split='train'):

    with open('./dataset/all_metrics.txt', 'r') as f_metrics:
        with open('./dataset/all_constraints.txt', 'r') as f_constraints:
            with open('./dataset/all_solutions.txt', 'r') as f_solutions:
                with open('./dataset/all_netlists.txt', 'r') as f_netlists:
                    with open('./dataset/all_ordering.txt', 'r') as f_ordering:

                        f_metrics = f_metrics.readlines()
                        f_constraints = f_constraints.readlines()
                        f_solutions = f_solutions.readlines()
                        f_netlists = f_netlists.readlines()
                        f_ordering = f_ordering.readlines()

                        design_name_data = []
                        metric_data = []
                        constraint_data = []
                        solution_data = []
                        netlist_data = []
                        ordering_data = []

                        # for i in range(len(f_metrics)):  ### Formal run
                        for i in range(10):  ### Debug

                            ### metrics
                            metrics = f_metrics[i].strip().split('|')[1:]
                            # R_wirelength = metrics[0].strip()
                            # R_congestion = metrics[1].strip()
                            # metrics = np.array([[R_wirelength]], dtype=np.float32)
                            R = metrics[0].strip()
                            if R == 'none':
                                continue
                            else:
                                metrics = np.array([[R]], dtype=np.float32)
                                metric_data.append(metrics)

                                ### Design name
                                design_name = f_metrics[i].strip().split('|')[0].strip()
                                design_name_data.append(design_name)

                                ### solutions
                                solutions = f_solutions[i].strip().split('|')[1].strip().split(' ')
                                solutions = np.array(solutions, dtype=np.float32)
                                solution_data.append(solutions)

                                ### constraints
                                constraints = f_constraints[i].strip().split('|')[1].strip().split(' ')
                                constraints = [i.split(',') for i in constraints]
                                constraints = np.array(constraints, dtype=np.float32)
                                constraint_data.append(constraints)

                                # netlists
                                base_design_name = f_netlists[i].strip().split('|')[-1].strip()
                                netlist_data.append(base_design_name)

                                # ordering
                                ordering = f_ordering[i].strip().split('|')[1].strip().split(',')
                                ordering = np.array(ordering, dtype=np.int32)
                                ordering_data.append(ordering)

    if split == 'train':  ### Provide real reward in training
        metric_data = metric_data
    else:  ### Maximize reward in reference
        metric_data = np.ones_like(metric_data, dtype=np.float32)

    train_design_name_data, val_design_name_data, train_metric_data, val_metric_data, train_solution_data, val_solution_data, train_constraint_data, val_constraint_data, train_netlist_data, val_netlist_data, train_ordering_data, val_ordering_data = train_test_split(
        design_name_data,
        metric_data,
        solution_data,
        constraint_data,
        netlist_data,
        ordering_data,
        test_size=0.1,
        random_state=1020)

    if split == 'train':
        print('Loading training set')
        train_dataset = {'design_name': train_design_name_data, 'metric': train_metric_data, 'solution': train_solution_data,
                         'constraint': train_constraint_data, 'netlist': train_netlist_data, 'ordering': train_ordering_data}
        val_dataset = {'design_name': val_design_name_data, 'metric': val_metric_data, 'solution': val_solution_data,
                       'constraint': val_constraint_data, 'netlist': val_netlist_data, 'ordering': val_ordering_data}
        return train_dataset, val_dataset
    else:
        print('Loading validation set')
        val_dataset = {'design_name': val_design_name_data, 'metric': val_metric_data, 'solution': val_solution_data,
                       'constraint': val_constraint_data, 'netlist': val_netlist_data, 'ordering': val_ordering_data}
        return val_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, metric_enc, place_enc, region_enc, device, split='train'):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['metric'])
        self.metric_enc = metric_enc
        ### Total size of vocab = 2282 = 2280 (#site: 0~2279) + 1 (stop token: 2280) + 1 (padding token: 2281)
        self.place_enc = place_enc
        self.region_enc = region_enc
        self.device = device
        self.hidden_size = self.metric_enc.bias.shape[0]*2
        self.avgpool = torch.nn.AvgPool1d(1)
        self.gnn_ckpt = './ParetoGNN/scripts/Design_2/experiment_name/checkpoint/step-10000_ssnc/model.pth.tar'
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        design_name = self.text_datasets['design_name'][idx]
        metric = self.text_datasets['metric'][idx]
        solution = self.text_datasets['solution'][idx]
        constraint = self.text_datasets['constraint'][idx]
        netlist = self.text_datasets['netlist'][idx]
        ordering = self.text_datasets['ordering'][idx]

        ### Stop token: 2280
        stop_token = np.array([2280])
        stop_emb = self.place_enc(torch.tensor(
            stop_token, dtype=torch.float32, requires_grad=True).to(torch.int32).to(self.device))
        ### Padding token: 2281
        pad_token = np.array([2281])
        pad_emb = self.place_enc(torch.tensor(
            pad_token, dtype=torch.float32, requires_grad=True).to(torch.int32).to(self.device))

        ### Generate solution embedding
        raw_input = []
        final_emb = []
        for s in range(len(solution)):
            cur_sol = solution[s]
            raw_input.append([cur_sol])
            cur_sol = torch.tensor(cur_sol, dtype=torch.float32, requires_grad=True).to(
                torch.int32).to(self.device)
            cur_sol_emb = self.place_enc(cur_sol)
            final_emb.append(cur_sol_emb)
        final_emb = torch.stack(final_emb)  ### (sequence_length, width/2)
        # print('final_emb.shape (solution):', final_emb.shape)
        solution_dim = final_emb.shape[-1]


        ### Generate constraint embedding
        constraint_emb = []
        for c in range(len(constraint)):
            cur_con = constraint[c]  ### [lo_x lo_y hi_x hi_y]
            cur_con = torch.tensor(
                cur_con, dtype=torch.float32, requires_grad=True).to(self.device)
            cur_con_emb = self.region_enc(cur_con)
            constraint_emb.append(cur_con_emb)
        constraint_emb = torch.stack(constraint_emb)  ### (sequence_length, width/4)
        # print('constraint_emb.shape:', constraint_emb.shape)
        final_emb = torch.cat((final_emb, constraint_emb), dim=1)  ### -> (sequence_length, width*3/4)
        # print('final_emb.shape (solution & constraint):', final_emb.shape)

        netlist_emb = get_node_emb(self.gnn_ckpt, netlist)
        netlist_emb = netlist_emb[-len(final_emb):]
        netlist_emb_proj = torch.nn.Linear(
            netlist_emb.shape[1], constraint_emb.shape[1], device=self.device, dtype=torch.float32)
        netlist_emb = netlist_emb_proj(netlist_emb)  ### (sequence_length, width/4)
        final_emb = torch.cat((final_emb, netlist_emb), dim=1)  ### -> (sequence_length, width)
        # print('final_emb.shape (solution & constraint & netlist):', final_emb.shape)

        ### Ordering the sequence according to macro size and degree
        final_emb = final_emb[torch.tensor(ordering, dtype=torch.long)]

        ### Generate metric embedding
        metric_emb = torch.tensor(
            metric, dtype=torch.float32, requires_grad=True).to(self.device)
        metric_emb = self.metric_enc(metric_emb)
        final_emb = torch.cat((metric_emb, final_emb))
        # print('final_emb.shape (metric @ solution & constraint & netlist):', final_emb.shape)
        raw_input.insert(0, [1])

        ### Generate mask
        mask = [0] + [1]*(len(solution)+1)
        mask_decoder = [0] + [1]*(len(solution)+1)

        ### Stop and padding
        # print('stop_emb.shape (before pad):', stop_emb.shape)
        # print('pad_emb.shape (before pad):', pad_emb.shape)
        stop_emb = torch.nn.functional.pad(
            stop_emb, (0, final_emb.shape[-1]-stop_emb.shape[-1]))
        pad_emb = torch.nn.functional.pad(
            pad_emb, (0, final_emb.shape[-1]-pad_emb.shape[-1]))
        # print('stop_emb.shape (after pad):', stop_emb.shape)
        # print('pad_emb.shape (after pad):', pad_emb.shape)

        final_emb = torch.cat((final_emb, stop_emb))
        # print('final_emb.shape (metric @ solution & constraint & netlist @ stop):', final_emb.shape)
        raw_input.append(stop_token)

        # SEQ_LEN = 2282 - 1 ### Max length of placement solution is 2280, add 1 stop emb and leave 1 for time emb
        SEQ_LEN = 2304 - 1  ### Sequence length = 2304 = 128 * 18 is divisible by 128, it is required by the new transformer model
        if len(final_emb) <= SEQ_LEN:
            pad_len = SEQ_LEN - len(final_emb)
            final_emb = torch.cat((final_emb, pad_emb.repeat(pad_len, 1)))
            raw_input += [pad_token] * pad_len
            mask += [1] * pad_len
            mask_decoder += [0] * pad_len
        else:
            raise NotImplementedError(
                "Something wrong with the sequence length")
        # print('final_emb.shape (metric @ solution & constraint & netlist @ stop @ pad):', final_emb.shape)

        out_kwargs = {}
        out_kwargs['emb'] = final_emb
        out_kwargs['mask'] = torch.tensor(mask, dtype=torch.float32)
        out_kwargs['raw_input'] = torch.tensor(
            np.array(raw_input, dtype=np.float32), dtype=torch.float32)

        if out_kwargs['mask'].shape[0] != SEQ_LEN:
            raise NotImplementedError

        mask = torch.unsqueeze(out_kwargs['mask'], -1)
        emb_mask = torch.where(
            mask==0, 
            torch.zeros((1,final_emb.shape[-1])), 
            torch.cat(
                (
                    torch.ones((1,solution_dim)), 
                    torch.zeros((1,final_emb.shape[-1]-solution_dim)),
                ), dim=1
            )
        )
        out_kwargs['emb_mask'] = emb_mask

        if self.split == 'train':
            return out_kwargs['emb'], out_kwargs
        else:
            out_kwargs['design_name'] = design_name 
            out_kwargs['ordering'] = ordering
            out_kwargs['mask_decoder'] = torch.tensor(mask_decoder, dtype=torch.float32)
            return out_kwargs['emb'], out_kwargs
