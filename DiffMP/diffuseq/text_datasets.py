from ParetoGNN.only_forward import get_node_emb
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
import sys
import os
import pdb

sys.path.append('..')
sys.path.append('../ParetoGNN')


def load_data_text(
    batch_size,
    split='train',
    loop=True,        ### loop
    metric_enc=None,  ### encode metric (1->hidden size)
    dsp_place_enc=None,  ### encode solution (1->hidden size)
    bram_place_enc=None,
    region_enc=None, ### encode constraint region (4->hidden size)
    device=None,
):

    if split == 'train':
        print('#'*30, '\nLoading data for training...')

        train_data, val_data = get_corpus(split=split)

        # train dataset
        # input1 : train_data (the data used for the training)
        # input2-5: all kinds of encodings
        train_dataset = TextDataset(
            train_data,
            metric_enc=metric_enc,
            dsp_place_enc=dsp_place_enc,
            bram_place_enc=bram_place_enc,
            region_enc=region_enc,
            device=device,
        )

        # validation dataset
        val_dataset = TextDataset(
            val_data,
            metric_enc=metric_enc,
            dsp_place_enc=dsp_place_enc,
            bram_place_enc=bram_place_enc,
            region_enc=region_enc,
            device=device,
        )

        print('#Data in the training set:', train_dataset.__len__())
        print('#Data in the validation set:', val_dataset.__len__())

        #train dataloader
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # validation dataloader
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
        # test dataset
        test_dataset = TextDataset(
            test_data,
            metric_enc=metric_enc,
            dsp_place_enc=dsp_place_enc,
            bram_place_enc=bram_place_enc,
            region_enc=region_enc,
            device=device,
        )

        print('#Data in the test set:', test_dataset.__len__())

        # test data loader
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

                        # all kinds of data
                        design_name_data = []
                        metric_data = []
                        BRAM_constraint_data = []
                        DSP_constraint_data = []
                        BRAM_solution_data = []
                        DSP_solution_data = []
                        netlist_data = []
                        BRAM_ordering_data = []
                        DSP_ordering_data = []

                        for i in range(len(f_metrics)):  ### Formal run (all the solutions)
                        #for i in range(10):  ### used for debugging

                            ### metrics (Macrowirelength)
                            metrics = f_metrics[i].strip().split('|')[1:]
                            # R_wirelength = metrics[0].strip()
                            # R_congestion = metrics[1].strip()
                            # metrics = np.array([[R_wirelength]], dtype=np.float32)
                            R = metrics[0].strip()
                            ###
                            if R == 'none':
                                continue
                            else:
                                # metrics
                                metrics = np.array([[R]], dtype=np.float32)
                                metric_data.append(metrics)

                                ### Design name
                                design_name = f_metrics[i].strip().split('|')[0].strip()
                                design_name_data.append(design_name)

                                ### solutions for BRAMs and DSPs (the DSP or BRAM would be put at which site)
                                # DSPs solutions
                                DSP_solutions = f_solutions[i].strip().split('|')[1].strip().split(' ')
                                # place BRAMs
                                BRAM_solutions = f_solutions[i].strip().split('|')[2].strip().split(' ')
                                
                                BRAM_solutions = np.array(BRAM_solutions, dtype=np.float32)
                                DSP_solutions = np.array(DSP_solutions, dtype=np.float32)

                                # all the solutions
                                BRAM_solution_data.append(BRAM_solutions)
                                DSP_solution_data.append(DSP_solutions)

                                ### constraints
                                ## the methods to deal with the region constraints are very strange
                                ## not only do the DSPs have the regional constraints, so do the BRAMs
                                ## how to indicate which BRAM or DSP have the regional constraints
                                ## Temporally place DSP

                                DSP_constraints = f_constraints[i].strip().split('|')[1].strip().split(' ')
                                DSP_constraints = [i.split(',') for i in DSP_constraints]

                                BRAM_constraints = f_constraints[i].strip().split('|')[2].strip().split(' ')
                                BRAM_constraints = [i.split(',') for i in BRAM_constraints]

                                # constraint_data add the constraints together
                                BRAM_constraint_data.append(BRAM_constraints)
                                DSP_constraint_data.append(DSP_constraints)

                                # netlists
                                # which design for the solution belongs to
                                base_design_name = f_netlists[i].strip().split('|')[-1].strip()
                                netlist_data.append(base_design_name)

                                # ordering
                                DSP_ordering = f_ordering[i].strip().split('|')[1].strip().split(',')
                                BRAM_ordering = f_ordering[i].strip().split('|')[2].strip().split(',')

                                DSP_ordering = np.array(DSP_ordering, dtype=np.int32)
                                BRAM_ordering = np.array(BRAM_ordering, dtype=np.int32)
                                
                                BRAM_ordering_data.append(BRAM_ordering)
                                DSP_ordering_data.append(DSP_ordering)

    if split == 'train':  ### Provide real reward in training
        metric_data = metric_data
    else:  ### Maximize reward in reference
        metric_data = np.ones_like(metric_data, dtype=np.float32)

    # all the files are split into the training set and validation sets
    # this split would allow for the many cases in one design
    # use one design for training
    # To do: use better training/val splitting method
    train_design_name_data, val_design_name_data, train_metric_data, val_metric_data, train_BRAM_solution_data, val_BRAM_solution_data, \
        train_DSP_solution_data, val_DSP_solution_data, train_BRAM_constraint_data, val_BRAM_constraint_data, train_DSP_constraint_data, val_DSP_constraint_data, \
        train_netlist_data, val_netlist_data, train_BRAM_ordering_data, val_BRAM_ordering_data, train_DSP_ordering_data, val_DSP_ordering_data = train_test_split(
        design_name_data,
        metric_data,
        BRAM_solution_data,
        DSP_solution_data,
        BRAM_constraint_data,
        DSP_constraint_data,
        netlist_data,
        BRAM_ordering_data,
        DSP_ordering_data,
        test_size=0.1,
        random_state=1020)

    # Training set and validation set
    if split == 'train':
        print('Loading training set')
        train_dataset = {'design_name': train_design_name_data, 'metric': train_metric_data, 'BRAM_solution_data': train_BRAM_solution_data,
                         'DSP_solution_data': train_DSP_solution_data, 'BRAM_constraint_data': train_BRAM_constraint_data, 
                         'DSP_constraint_data': train_DSP_constraint_data, 'netlist': train_netlist_data, 'BRAM_ordering': train_BRAM_ordering_data,
                         'DSP_ordering': train_DSP_ordering_data}
        val_dataset = {'design_name': val_design_name_data, 'metric': val_metric_data, 'BRAM_solution_data': val_BRAM_solution_data,
                         'DSP_solution_data': val_DSP_solution_data, 'BRAM_constraint_data': val_BRAM_constraint_data, 
                         'DSP_constraint_data': val_DSP_constraint_data, 'netlist': val_netlist_data, 'BRAM_ordering': val_BRAM_ordering_data,
                         'DSP_ordering': val_DSP_ordering_data}
        return train_dataset, val_dataset
    else:
        print('Loading validation set')
        val_dataset = {'design_name': val_design_name_data, 'metric': val_metric_data, 'BRAM_solution_data': val_BRAM_solution_data,
                         'DSP_solution_data': val_DSP_solution_data, 'BRAM_constraint_data': val_BRAM_constraint_data, 
                         'DSP_constraint_data': val_DSP_constraint_data, 'netlist': val_netlist_data, 'BRAM_ordering': val_BRAM_ordering_data,
                         'DSP_ordering': val_DSP_ordering_data}
        return val_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, metric_enc, dsp_place_enc, bram_place_enc, region_enc, device, split='train'):
        super().__init__()
        
        self.text_datasets = text_datasets

        self.length = len(self.text_datasets['metric'])
        self.metric_enc = metric_enc
        ### Total size of vocab = 3002 = 2280+720 (#site: 0~2999) + 1 (stop token: 3000) + 1 (padding token: 3001)
        self.dsp_place_enc = dsp_place_enc
        self.bram_place_enc = bram_place_enc

        self.region_enc = region_enc

        self.device = device
        self.hidden_size = self.metric_enc.bias.shape[0]*2

        self.avgpool = torch.nn.AvgPool1d(1)
        ## checkpoint name
        self.gnn_ckpt = './ParetoGNN/scripts/Design_2/experiment_name/checkpoint/step-50000_ssnc/model.pth.tar'
        self.split = split

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        design_name = self.text_datasets['design_name'][idx]
        metric = self.text_datasets['metric'][idx]
        BRAM_solution = self.text_datasets['BRAM_solution_data'][idx]
        DSP_solution = self.text_datasets['DSP_solution_data'][idx]
        BRAM_constraint = self.text_datasets['BRAM_constraint_data'][idx]
        DSP_constraint = self.text_datasets['DSP_constraint_data'][idx]

        netlist = self.text_datasets['netlist'][idx]
        BRAM_ordering = self.text_datasets['BRAM_ordering'][idx]
        DSP_ordering = self.text_datasets['DSP_ordering'][idx]

        ### Stop token: 3000
        stop_token = np.array([0])
        stop_emb = self.dsp_place_enc(torch.tensor(
            stop_token, dtype=torch.float32, requires_grad=True).to(torch.int32).to(self.device))
        
        ### Padding token: 3001
        pad_token = np.array([720])
        pad_emb = self.bram_place_enc(torch.tensor(
            pad_token, dtype=torch.float32, requires_grad=True).to(torch.int32).to(self.device))

        ### Generate solution embedding
        raw_input = []
        final_emb = []

        # Generate Embedding for the DSP solutions
        for s in range(len(DSP_solution)):
            DSP_solution_per = DSP_solution[s]
            raw_input.append([DSP_solution_per])
            DSP_solution_per = torch.tensor(DSP_solution_per, dtype=torch.float32, requires_grad=True).to(torch.int32).to(self.device)
            dsp_sol_emb = self.dsp_place_enc(DSP_solution_per)
            final_emb.append(dsp_sol_emb)

        # Generate Embedding for BRAM solutions
        for s in range(len(BRAM_solution)):
            BRAM_solution_per = BRAM_solution[s]
            raw_input.append([BRAM_solution_per])
            BRAM_solution_per = torch.tensor(BRAM_solution_per, dtype=torch.float32, requires_grad=True).to(torch.int32).to(self.device)
            bram_sol_emb = self.bram_place_enc(BRAM_solution_per)
            final_emb.append(bram_sol_emb)
                
        final_emb = torch.stack(final_emb)  ### (sequence_length, width/2)
        # print('final_emb.shape (solution):', final_emb.shape)
        solution_dim = final_emb.shape[-1]

        constraint_emb = []
        ### Generate constraint embedding for DSP solutions
        for c in range(len(DSP_constraint)):
            dsp_con_org = DSP_constraint[c]  ### [lo_x lo_y hi_x hi_y]
            dsp_con = [float(i) for i in dsp_con_org]
            dsp_con = torch.tensor(dsp_con, dtype=torch.float32, requires_grad=True).to(self.device)
            dsp_con_emb = self.region_enc(dsp_con)
            constraint_emb.append(dsp_con_emb)

        ### Generate constraint embedding for BRAM solutions
        for c in range(len(BRAM_constraint)):
            bram_con_org = BRAM_constraint[c]  ### [lo_x lo_y hi_x hi_y]
            bram_con = [float(i) for i in bram_con_org]
            bram_con = torch.tensor(bram_con, dtype=torch.float32, requires_grad=True).to(self.device)
            bram_con_emb = self.region_enc(bram_con)
            constraint_emb.append(bram_con_emb)        

        constraint_emb = torch.stack(constraint_emb)  ### (sequence_length, width/4)
        # print('constraint_emb.shape:', constraint_emb.shape)

        final_emb = torch.cat((final_emb, constraint_emb), dim=1)  ### -> (sequence_length, width*3/4)
        # print('final_emb.shape (solution & constraint):', final_emb.shape)

        ### Netlist embedding (pretrain)
        netlist_emb = get_node_emb(self.gnn_ckpt, netlist)

        ### Why the order of the embedding is like this?
        ### Whether the BRAM is firstly read and then the DSP? (correspondence)
        bram_netlist_emb = netlist_emb[:len(BRAM_constraint)]
        dsp_netlist_emb = netlist_emb[-len(DSP_constraint):]
        netlist_emb = torch.cat((dsp_netlist_emb, bram_netlist_emb), axis=0)

        ### (sequence_length, width/4)
        netlist_emb_proj = torch.nn.Linear(
            netlist_emb.shape[1], constraint_emb.shape[1], device=self.device, dtype=torch.float32)
        
        netlist_emb = netlist_emb_proj(netlist_emb)  ### (sequence_length, width/4)
        final_emb = torch.cat((final_emb, netlist_emb), dim=1)  ### -> (sequence_length, width)
        # print('final_emb.shape (solution & constraint & netlist):', final_emb.shape)

        ### Ordering the sequence according to macro size and degree (ordering would meet some problems)
        # final_emb = final_emb[torch.tensor(ordering, dtype=torch.long)]
        final_dsp_emb = final_emb[:len(DSP_constraint)]
        final_bram_emb = final_emb[-len(BRAM_constraint):]
        final_dsp_emb = final_dsp_emb[torch.tensor(DSP_ordering, dtype=torch.long)]
        final_bram_emb = final_bram_emb[torch.tensor(BRAM_ordering, dtype=torch.long)]
        final_emb = torch.cat((final_dsp_emb, final_bram_emb), axis=0)

        ### Generate metric embedding
        metric_emb = torch.tensor(
            metric, dtype=torch.float32, requires_grad=True).to(self.device)
        ### evaluation metric as the embedding
        ### how to get the metric (for different case)
        metric_emb = self.metric_enc(metric_emb)
        final_emb = torch.cat((metric_emb, final_emb))
        # print('final_emb.shape (metric @ solution & constraint & netlist):', final_emb.shape)
        
        
        raw_input.insert(0, [1])

        ### Generate mask
        ### mask would let the nodes know where it could place the macros (would the mask be changed)
        ### stop signature
        solution_length = len(DSP_solution) + len(BRAM_solution)
        mask = [0] + [1]*(solution_length+1)
        mask_decoder = [0] + [1]*(solution_length+1)

        ### Stop and padding
        # print('stop_emb.shape (before pad):', stop_emb.shape)
        # print('pad_emb.shape (before pad):', pad_emb.shape)
        # stop token, padding token
        stop_emb = torch.nn.functional.pad(
            stop_emb, (0, final_emb.shape[-1]-stop_emb.shape[-1]))
        pad_emb = torch.nn.functional.pad(
            pad_emb, (0, final_emb.shape[-1]-pad_emb.shape[-1]))
        # print('stop_emb.shape (after pad):', stop_emb.shape)
        # print('pad_emb.shape (after pad):', pad_emb.shape)

        final_emb = torch.cat((final_emb, stop_emb))
        # print('final_emb.shape (metric @ solution & constraint & netlist @ stop):', final_emb.shape)
        raw_input.append(stop_token)

        # SEQ_LEN = 2282 - 1 ### Max length of placement solution is 2280, add 1 stop emb and leave 1 for time emb (embedding)
        # SEQ_LEN = 2304 - 1  ### Sequence length = 2304 = 128 * 18 is divisible by 128, it is required by the new transformer model
        SEQ_LEN =  3072 - 1   ### Sequence length = 3072 = 128 * 24 is divisible by 128, it is required by the new transformer model
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

        # whether to change it into the mask for the regional constraints
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
            #  BRAM/DSP ordering
            out_kwargs['BRAM_ordering'] = BRAM_ordering
            out_kwargs['DSP_ordering'] = DSP_ordering
            out_kwargs['mask_decoder'] = torch.tensor(mask_decoder, dtype=torch.float32)
            return out_kwargs['emb'], out_kwargs
