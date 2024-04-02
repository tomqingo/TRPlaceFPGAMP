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
        # buffer capacity = 10 * placed_num_macro
        if self.placed_num_macro:
            self.buffer_capacity = 10 * (self.placed_num_macro)
        else:
            self.buffer_capacity = 5120
    
    def load_param(self, path):
        checkpoint = torch.load(path, map_location=torch.device(device))
        self.actor_net.load_state_dict(checkpoint['actor_net_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_dict'])
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _, _ = self.actor_net(state, self.soft_coefficient)
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
        benchmark_save_model_path = os.path.join("save_models", self.benchmark)
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
                action_probs, _, _ = self.actor_net(state[index].to(device), self.soft_coefficient)
                dist = Categorical(action_probs)
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

def train_model(args, logger):
    dataset = load_dataset(args, logger)
    # if args.pnm > dataset.num_macro:
    # the number of the placement units
    args.pnm = dataset.num_macro
    placed_num_macro = args.pnm

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'reward_intrinsic'])
    env = gym.make("place_env-v0", database = dataset, grid_width = dataset.grid_width, grid_height = dataset.grid_height, placed_num_macro = placed_num_macro)

    TrainingRecord = namedtuple('TrainRecord',['episode', 'reward'])
    agent = PPO(args, dataset.grid_width, dataset.grid_height)

    strftime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    running_reward = -1000000
    training_records = []

    if not args.is_test:
        log_file_name = "logs/log_"+ args.design_name + "_" + strftime + "_seed_"+ str(args.seed) + "_pnm_" + str(args.pnm) + ".csv"
        if not os.path.exists("logs"):
            os.mkdir("logs")
        fwrite = open(log_file_name, "w")
        # The largest reward
        best_reward = running_reward

    load_model_path = args.checkpoint_path
    if load_model_path:
       agent.load_param(load_model_path)
    

    total_epochs = args.epochs
    if args.is_test:
        torch.inference_mode()
        total_epochs = 1
    
    #pdb.set_trace()

    # 10000 epochs
    for i_epoch in range(total_epochs):
        score = 0
        raw_score = 0
        # start time for the macro placement
        start = time.time()
        # The same images?
        state = env.reset()
        done = False
        macro_cnt = 0
        while done is False:
            state_tmp = state.copy()
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # to see the calculation of the reward
            # update the action
            reward_intrinsic = 0
            # Why is the reward devided by 200?
            if not args.is_test:
                trans = Transition(state_tmp, action, reward / 200.0, action_log_prob, next_state, reward_intrinsic)
            # pdb.set_trace()
            # store
            if not args.is_test and agent.store_transition(trans):                
                assert done == True
                agent.update()
            
            # scores
            score += reward
            raw_score += info["raw_reward"]
            state = next_state

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
            pl_file_path = "{}-{}-{}.pl".format(benchmark, int(macrohpwl), time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            # save_placement(pl_file_path, env.node_pos, env.ratio)
            strftime_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            pl_folder = 'gg_place_new'
            pl_path = os.path.join(pl_folder, '{}-{}-{}.pl'.format(benchmark, strftime_now, int(macrohpwl)))
            if not os.path.exists(pl_folder):
                os.makedirs(pl_folder)
            fwrite_pl = open(pl_path, 'w')
            for nodeid in env.node_pos:
                x, y, size_x, size_y = env.node_pos[nodeid]
                node_name = dataset.macros[nodeid].name
                x = x
                y = y
                fwrite_pl.write("{}\t{:.4f}\t{:.4f}\n".format(node_name, x, y))
            fwrite_pl.close()
        
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
        if i_epoch % 100 == 0:
            if placed_num_macro is None:
                gl_folder = "./gl"
                if not os.path.exists(gl_folder):
                    os.makedirs(gl_folder)
                env.write_gl_file("./gl/{}{}.gl".format(strftime, int(score)))
            fig, ax1 = plt.subplots()



    


