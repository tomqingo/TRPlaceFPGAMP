import argparse
import pickle
from collections import namedtuple

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np

import gym

from src import *
from src.model import *
from src.place_env import *

import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torch
import torchvision
import torch.optim as optim
from torch.distributions import Categorical

from .check_legality import CheckLegality
from .plot_macro_placement import draw_macro_placement_result

import pdb

# Convert the string to bool value
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

device = torch.device('cuda')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

writer = SummaryWriter('./tb_log')


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10

    def __init__(self, args, grid_width, grid_height):
        super(PPO, self).__init__()
        self.gcn = None
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.cnn = MyCNN().to(device)
        self.cnn_coarse = MyCNNCoarse(self.resnet, device).to(device)
        self.actor_net = Actor(cnn = self.cnn, gcn = self.gcn, cnn_coarse = self.cnn_coarse, grid_width = grid_width, grid_height= grid_height).float().to(device)
        self.critic_net = Critic(cnn = self.cnn, gcn = self.gcn, cnn_coarse = None, res_net=self.resnet).float().to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        # ACTOR-CRITIC
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), args.lr)
        # hyperparameters in training
        self.placed_num_macro = args.pnm
        self.soft_coefficient = args.soft_coefficient
        self.gamma = args.gamma
        self.disable_tqdm = args.disable_tqdm
        self.batch_size = args.batch_size
        self.benchmark = args.design_name
        self.exp_id = args.exp_id
        # buffer capacity = 10 * placed_num_macro
        if self.placed_num_macro:
            self.buffer_capacity = 10 * (self.placed_num_macro)
        else:
            self.buffer_capacity = 5120
    
    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state, column_ids):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state, self.soft_coefficient, column_ids)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, running_reward):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("save_models"):
            os.mkdir("save_models")
        # -13166 reward
        benchmark_save_model_path = os.path.join("save_models", self.benchmark, self.exp_id)
        if not os.path.exists(benchmark_save_model_path):
            os.mkdir(benchmark_save_model_path)
        
        torch.save({"actor_net_dict": self.actor_net.state_dict(),
                    "critic_net_dict": self.critic_net.state_dict()},

                    os.path.join(benchmark_save_model_path, "net_dict-{}-{}-".format(self.benchmark, self.placed_num_macro)+strftime+"{}".format(int(running_reward))+".pkl"))        

    def store_transition(self, transition):
        # add the transition state
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0
    
    def update(self, is_adjust_lr):
        # state
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        # action
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        # reward
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        # column id
        column_ids = torch.tensor(np.array([t.col_ids for t in self.buffer]), dtype=torch.int).view(-1, 1).to(device)

        # log action
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        
        target_list = []
        target = 0
        # adjust the learning rate
        if is_adjust_lr:
            self.adjust_learning_rate(self.actor_optimizer, 2.5e-2)

        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= self.placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
       
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = self.disable_tqdm):
                self.training_step +=1
                
                # action probability
                action_probs, _, _ = self.actor_net(state[index].to(device), self.soft_coefficient, column_ids[index])
                dist = Categorical(action_probs)
                # action log probability
                action_log_prob = dist.log_prob(action[index].squeeze())
                # According to the probability to update
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]
                # critic_net_output                
                critic_net_output = self.critic_net(state[index].to(device))
                # actor-critic framework (target_v - baseline)
                advantage = (target_v - critic_net_output).detach()

                # actor loss < 0
                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze()
                #pdb.set_trace()
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                self.actor_optimizer.zero_grad()
                action_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index].to(device)), target_v)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                writer.add_scalar('action_loss', action_loss, self.training_step)
                writer.add_scalar('value_loss', value_loss, self.training_step)
    
    # adjust the learning rate for the optimizers
    def adjust_learning_rate(self, optimizer, adjust_lr):
        for params in optimizer.param_groups:
            params["lr"] = adjust_lr

# Reinforce Learning of the cell distribution in columns
class PPO_col():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10

    def __init__(self, args, grid_width):
        super(PPO_col, self).__init__()
        # grid columns
        self.actorcol_net = ActorCol(grid_col = grid_width).float().to(device)
        self.criticcol_net = CriticCol().float().to(device)
        #pdb.set_trace()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.grid_width = grid_width
        # ACTOR-CRITIC
        self.actorcol_optimizer = optim.Adam(self.actorcol_net.parameters(), args.collr)
        self.criticcol_net_optimizer = optim.Adam(self.criticcol_net.parameters(), args.collr)
        # hyperparameters in training
        # The number of the placed macros
        self.placed_num_macro = args.pnm
        self.soft_coefficient = args.soft_coefficient
        self.gamma = args.gamma
        self.disable_tqdm = args.disable_tqdm
        self.batch_size = args.batch_size
        self.benchmark = args.design_name
        self.exp_id = args.exp_id
        self.k_col = args.k_col
        # buffer capacity = 10 * placed_num_macro
        if self.placed_num_macro:
            self.buffer_capacity = 10 * (self.placed_num_macro)
        else:
            self.buffer_capacity = 5120
    
    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actorcol_net.load_state_dict(checkpoint['actorcol_net_dict'])
        self.criticcol_net.load_state_dict(checkpoint['criticcol_net_dict'])
    
    # Selet the col id (without the overflow and X-displacement)
    def select_action(self, state):
        # pdb.set_trace()
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _= self.actorcol_net(state, self.soft_coefficient)
        dist = Categorical(action_probs)
        action = dist.sample()
        #pdb.set_trace()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob.item()
    
    def select_topk_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs,_ = self.actorcol_net(state, self.soft_coefficient)
        dist = Categorical(action_probs)
        # k = 3  # k = 3
        # take the k-largest
        values, indices = dist.probs.topk(self.k_col, dim=1)
        return indices

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.criticcol_net(state)
        return value.item()

    def save_param(self, running_reward):
        strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("save_models_col"):
            os.mkdir("save_models_col")
        # -13166 reward
        benchmark_save_model_path = os.path.join("save_models_col", self.benchmark, self.exp_id)
        if not os.path.exists(benchmark_save_model_path):
            os.mkdir(benchmark_save_model_path)
        

        torch.save({"actorcol_net_dict": self.actorcol_net.state_dict(),
                    "criticcol_net_dict": self.criticcol_net.state_dict()},

                    os.path.join(benchmark_save_model_path, "net_dict-{}-{}-".format(self.benchmark, self.placed_num_macro)+strftime+"{}".format(int(running_reward))+".pkl"))        

    def store_transition(self, transition):
        # add the transition state
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0
    
    def update(self):
        # state
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float)
        # action
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        # reward
        reward = torch.tensor(np.array([t.reward for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        # log action
        old_action_log_prob = torch.tensor(np.array([t.a_log_prob for t in self.buffer]), dtype=torch.float).view(-1, 1).to(device)
        del self.buffer[:]
        
        target_list = []
        target = 0
        # self.placed_num_macro (place every macro there would be a reward)
        for i in range(reward.shape[0]-1, -1, -1):
            if state[i, 0] >= self.placed_num_macro - 1:
                target = 0
            r = reward[i, 0].item()
            target = r + self.gamma * target
            target_list.append(target)
        # calculate the culmulative rewards from i to the end macrkxss
        target_list.reverse()
        target_v_all = torch.tensor(np.array([t for t in target_list]), dtype=torch.float).view(-1, 1).to(device)
       
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in tqdm(BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True),
                disable = self.disable_tqdm):
                self.training_step +=1
                
                # action probability
                action_probs, _ = self.actorcol_net(state[index].to(device), self.soft_coefficient)
                #pdb.set_trace()
                dist = Categorical(action_probs)
                action_log_prob = dist.log_prob(action[index].squeeze())
                # According to the probability to update
                ratio = torch.exp(action_log_prob - old_action_log_prob[index].squeeze())
                target_v = target_v_all[index]
                # critic_net_output                
                critic_net_output = self.criticcol_net(state[index].to(device))
                # actor-critic framework (target_v - baseline)
                advantage = (target_v - critic_net_output).detach()

                # actor loss < 0
                L1 = ratio * advantage.squeeze() 
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage.squeeze()
                #pdb.set_trace()
                actioncol_loss = -torch.min(L1, L2).mean() # MAX->MIN desent

                # print(self.actor_optimizer.state_dict()['param_groups'][0]['lr'])
                # print(self.critic_net_optimizer.state_dict()['param_groups'][0]['lr'])

                self.actorcol_optimizer.zero_grad()
                actioncol_loss.backward()

                nn.utils.clip_grad_norm_(self.actorcol_net.parameters(), self.max_grad_norm)
                self.actorcol_optimizer.step()

                valuecol_loss = F.smooth_l1_loss(self.criticcol_net(state[index].to(device)), target_v)
                self.criticcol_net_optimizer.zero_grad()
                valuecol_loss.backward()
                nn.utils.clip_grad_norm_(self.criticcol_net.parameters(), self.max_grad_norm)
                self.criticcol_net_optimizer.step()

                writer.add_scalar('action_loss', actioncol_loss, self.training_step)
                writer.add_scalar('value_loss', valuecol_loss, self.training_step)
    
    # adjust the learning rate for the optimizers
    def adjust_learning_rate(self, optimizer, adjust_lr):
        for params in optimizer.param_groups:
            params["lr"] = adjust_lr


def train_model(args, logger):
    dataset = load_dataset(args, logger)
    # if args.pnm > dataset.num_macro:
    # the number of the placement units
    args.pnm = dataset.num_macro
    placed_num_macro = args.pnm

    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    Transition_col = namedtuple('Transition_col', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic', 'col_ids'])
    
    envcol = gym.make("place_colenv-v1", database = dataset, grid_col = dataset.grid_width, grid_col_capacity=dataset.gridcolnumsites, placed_num_macro = placed_num_macro)
    env = gym.make("place_env-v0", database = dataset, grid_width = dataset.grid_width, grid_height = dataset.grid_height, placed_num_macro = placed_num_macro)

    TrainingRecord = namedtuple('TrainRecord', ['episode', 'reward'])
    agent = PPO(args, dataset.grid_width, dataset.grid_height)
    load_model_path = args.checkpoint_path
    if load_model_path:
       agent.load_param(load_model_path)

    # Setup the model to predict the columns
    if args.colfirst:
        TrainingRecord_col = namedtuple('TrainRecord', ['episode', 'reward'])
        agent_col = PPO_col(args, dataset.grid_width)
        load_colmodel_path = args.checkpoint_path_col
        if load_colmodel_path:
            agent_col.load_param(load_colmodel_path)

    running_reward = -1000000
    training_records = []

    if not args.is_test:
        log_file_name = "logs/log_"+ args.design_name + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
        # logs_col and logs
        if args.traincol:
            logcol_file_name = "logs_col/log_"+ args.design_name + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
            if not os.path.exists("logs_col"):
                os.mkdir("logs_col")
            fwrite_col = open(logcol_file_name, "w")  
        if not os.path.exists("logs"):
            os.mkdir("logs")
        # output the results
        fwrite = open(log_file_name, "w")
        # The largest reward
        best_reward = running_reward
    
    # epochs for the training of the first model
    warm_up_epochs = args.warm_up_epochs
    total_epochs = args.epochs

    if args.is_test:
        torch.inference_mode()
        total_epochs = 1
    
    #pdb.set_trace()

    # epochs for training the column prediction network
    print("========Phase 1 Training=========")
    col2cascadeid = {} # column to cascade id

    if args.traincol:
        start = time.time()
        for i_epoch in range(warm_up_epochs):
            score = 0
            raw_score = 0
            state = envcol.reset()
            # pdb.set_trace()
            done = False
            macro_cnt = 0
            state_final = None
            while done is False:
                state_tmp = state.copy()
                action, action_log_prob = agent_col.select_action(state_tmp)
                # macro = dataset.macros[dataset.nodeidlist[macro_cnt]]
                # print(macro.name, action)
                # pdb.set_trace()
                next_state, reward, done, info = envcol.step(action)
                # to see the calculation of the reward
                # update the action and transition
                reward_intrinsic = 0
                if not args.is_test:
                    trans = Transition_col(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
                # agent_col.store_transition
                if not args.is_test and agent_col.store_transition(trans):
                    assert done == True
                    agent_col.update()
                score += reward
                raw_score += info["raw_reward"]
                state = next_state
                cascade_id = dataset.nodeidlist[macro_cnt]
                # col2cascadeid (action)
                if action not in col2cascadeid.keys():
                    col2cascadeid[action] = [cascade_id]
                else:
                    col2cascadeid[action].append(cascade_id)
            
                macro_cnt += 1

                if done:
                    state_final = state_tmp
            
            #pdb.set_trace()
            
            end = time.time()
            print("Endtime of stage1", end)

            if i_epoch == 0:
                running_reward = score
            running_reward = running_reward * 0.9 + score * 0.1
            print("Phase 1 Training: score : {}, raw score : {}".format(score, raw_score))

            if not args.is_test and (i_epoch % 100 == 0):
                agent_col.save_param(running_reward)

            if not args.is_test and i_epoch % 1 ==0:
                print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
                fwrite_col.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent_col.training_step))
                fwrite_col.flush()
    
        
    print("========Phase 2 Training=========")
    # 10000 epochs
    start = time.time()
    for i_epoch in range(total_epochs):
        score = 0
        raw_score = 0
        # start time for the macro placement
        # The same images?
        state = env.reset()
        if args.colfirst:
            state_col = envcol.reset()
        done = False
        macro_cnt = 0
        xscore = 0
        while done is False:
            state_tmp = state.copy()
            # Obtain the top-3 columns for the training
            is_adjust_lr = False
            if args.colfirst and (i_epoch < 50):
                state_col_tmp = state_col.copy()
                column_ids = agent_col.select_topk_action(state_col)
                # change the state according to the culumn ids
            else:
                # all the columns could be selected
                column_ids = np.array([list(range(0, dataset.grid_width))])
                column_ids = torch.Tensor(column_ids).to(device)
                if args.colfirst and (i_epoch < 100):
                    is_adjust_lr = True
            
            action, action_log_prob = agent.select_action(state, column_ids[0])
            next_state, reward, done, info = env.step(action)

            if args.colfirst:
                action_col = round(action // dataset.grid_height)
                next_state_col, reward_col, done, info_col = envcol.step(action_col)

            # to see the calculation of the reward
            # update the action
            reward_intrinsic = 0
            # Why is the reward devided by 200?
            if not args.is_test:
                trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic, column_ids[0].cpu().numpy())
            # pdb.set_trace()
            # store
            if not args.is_test and agent.store_transition(trans):                
                assert done == True
                agent.update(is_adjust_lr)
            
            # scores
            score += reward
            xscore += info["x_reward"]
            raw_score += info["raw_reward"]
            state = next_state
            if args.colfirst:
                state_col = next_state_col

            # update the Xcorr and Ycorr for each macro
            x = round(action // dataset.grid_height)
            real_x = dataset.gridcolid2colid[x]
            y = round(action % dataset.grid_height)
            cascade_id = dataset.nodeidlist[macro_cnt]
            macro = dataset.macros[cascade_id]
            submacro_cnt = 0
            base_siteid = 0

            for node in macro.Macronodecol:
                if submacro_cnt == 0:
                    dataset.macros[cascade_id].SetCascadeMacroLoc(real_x, y)
                    base_siteid = dataset.sitemaps[real_x][y]
                nodeid = node.id
                dataset.nodes[nodeid].locX = real_x
                dataset.nodes[nodeid].locY = dataset.sites[int(base_siteid + submacro_cnt)].locY
                dataset.nodes[nodeid].site = int(base_siteid + submacro_cnt)
                # pdb.set_trace()
                submacro_cnt += 1
            macro_cnt += 1

        # end time for the macro placement
        end = time.time()

        if i_epoch == 0:
            running_reward = score
        # Average award
        running_reward = running_reward * 0.9 + score * 0.1
        print("score = {}, raw_score = {}".format(score, raw_score))
        print("xscore = {}".format(xscore))
        
        # one policy as the initialize to train, not good
        # if running_reward > best_reward * 0.975:
        #     best_reward = running_reward
        if not args.is_test and i_epoch % 100 == 0:
            agent.save_param(running_reward)
            if args.save_fig:
                strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                if not os.path.exists("figures"):
                    os.mkdir("figures")
                env.save_fig("./figures/{}{}.png".format(strftime_now,int(raw_score)))
                print("save_figure: figures/{}{}.png".format(strftime_now,int(raw_score)))
            # try:
            #     print("start try")
            #     # cost is the routing estimation based on the MST algorithm
            #     hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)
            #     print("hpwl = {:.2f}\tcost = {:.2f}".format(hpwl, cost))
            # except:
            #     assert False
        
        # Store the result

        # show the coordinates for all the macros
        # for macro in dataset.macros:
        #     for node in macro.Macronodecol:
        #         nodeid = node.id
        #         print(dataset.nodes[nodeid].name, dataset.nodes[nodeid].locX, dataset.nodes[nodeid].locY)
        # pdb.set_trace()

        if args.is_test:
            print("save node_pos")
            macrohpwl = dataset.calMacroHPWL()
            log_dir = os.path.join(args.result_dir, args.exp_id, args.log_dir, args.design_name)
            legal_flag = CheckLegality(dataset, log_dir, logger)
            logger.info("macrohpwl = {:.2f}".format(macrohpwl))
            print("Check legal flag of the macro placement")
            if legal_flag:
                print("The macro placement is legal!!")
            else:
                print("The macro placement is not legal!!")
            print("time = {}s".format(end-start))
            benchmark = args.design_name
            # save_placement(pl_file_path, env.node_pos, env.ratio)
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            pl_folder = 'place_result'
            pl_path = os.path.join(pl_folder, 'd'+benchmark.split("_")[-1], 'macroplacement.pl')
            if not os.path.exists(pl_folder):
                os.makedirs(pl_folder)
            fwrite_pl = open(pl_path, 'w')
            # #for nodeid in env.node_pos:
            # #    x, y, size_x, size_y = env.node_pos[nodeid]
            #     node_name = dataset.macros[nodeid].name
            #     # x = int(x)
            #     realx = dataset.gridcolid2colid[x]
            #     y = int(y)
            #     fwrite_pl.write("{} {} {} {}\n".format(node_name, realx, y, 0))
            for cascadeid in env.node_pos:
                macro = dataset.macros[cascadeid]
                for node in macro.Macronodecol:
                    x = dataset.nodes[node.id].locX
                    y = dataset.nodes[node.id].locY
                    node_name = dataset.nodes[node.id].name
                    fwrite_pl.write("{} {} {} {}\n".format(node_name, x, y, 0))
            fwrite_pl.close()

            # draw the macro placement
            draw_macro_placement_result(args, dataset, logger)
            
        
        training_records.append(TrainingRecord(i_epoch, running_reward))
        if not args.is_test and i_epoch % 1 ==0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
            fwrite.write("{},{},{:.2f},{}\n".format(i_epoch, score, running_reward, agent.training_step))
            fwrite.flush()
        writer.add_scalar('reward', running_reward, i_epoch)
        if running_reward > -100:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break
        # if i_epoch % 100 == 0:
        #     if placed_num_macro is None:
        #         gl_folder = "./gl"
        #         if not os.path.exists(gl_folder):
        #             os.makedirs(gl_folder)
        #         env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))
        #     fig, ax1 = plt.subplots()



    


