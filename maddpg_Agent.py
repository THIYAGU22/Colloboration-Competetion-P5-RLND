import numpy as np
import torch 
import random
import copy
from collections import namedtuple,deque
from Buffered_memory import ReplayBuffer

from actor_critic import Actor,Critic
from OrnsteinUhlenbeck_noise import OUNoise
import torch.nn.functional as F
import torch.optim as optim




BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
LEARN_NUM = 5
UPDATE_EVERY = 1
EPS_START = 5
EPS_END = 400
TAU = 7e-2
GAMMA = 0.99
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPGAGENT:
    def __init__(self,state_size,action_size,agents,random_seed):
        
        self.seed = random.seed(random_seed)
        self.state_size = state_size
        self.action_size = action_size
        self.agents = agents
        self.epsilon = EPS_START
        self.timestep = 0
        self.eps_decay = 1 / (EPS_END * LEARN_NUM) 
        
        #--- actor -----#
        
        self.actor_local = Actor(state_size,action_size,random_seed).to(device)
        self.actor_target = Actor(state_size,action_size,random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr=1e-3)
        
        #---- critic -----#
        
        self.critic_local = Critic(state_size,action_size,random_seed).to(device)
        self.critic_target = Critic(state_size,action_size,random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),lr=1e-3,weight_decay=0)
        
        
        self.noise = OUNoise((agents,action_size),random_seed)
        
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,random_seed)
        
        
        
        
    def step(self,state,action,reward,next_state,done,agent_num):
        self.timestep = self.timestep + 1
        self.memory.add_experience(state,action,reward,next_state,done)
        
         
        
        if len(self.memory) > BATCH_SIZE and self.timestep%UPDATE_EVERY == 0:
            for _ in range(LEARN_NUM):
                xp = self.memory.sample()
                self.learn(xp,GAMMA,agent_num)#GAMMA VALUE 0.99
                
    
    def act(self,state,noise_accumulate=True):
        state = torch.from_numpy(state).float().to(device)
        actions = np.zeros((self.agents,self.action_size))
        self.actor_local.eval()
        
        with torch.no_grad():
            ### fetching actions for agents
            for agent_actions , state_fetch in enumerate(state):
                action =self.actor_local(state_fetch).cpu().data.numpy()
                actions[agent_actions, : ] = action
                
            
        self.actor_local.train()
        
        #Epsilon greedy selection
        if noise_accumulate:
            actions += self.epsilon * self.noise.sample()
        return np.clip(actions , -1 , 1)
    
        
        
    def reset(self):
        self.noise.reset_internal_state()
        
    def learn(self,xp,gamma,agent_num):
        states,actions,rewards,next_states,dones = xp
        
        #---configuring critic and computation of loss with help of MSE
        
        actions_nxt = self.actor_target(next_states)
        
        if agent_num == 0:
            actions_nxt = torch.cat((actions_nxt,actions[:,2:]),dim=1)
        else:
            actions_nxt = torch.cat((actions[:,:2],actions_nxt),dim=1)

        q_target_next = self.critic_target(next_states,actions_nxt)
        
        q_target = rewards + ( gamma * q_target_next * (1 - dones))
        
        q_expected = self.critic_local(states,actions)
       
        #MSE LOSS
        critic_loss = F.mse_loss(q_expected,q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clips gradient norm of an iterable of parameters
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),1)
        self.critic_optimizer.step()
        
        
        #---configuring actor and computation of loss with help of MSE
        actor_predicted = self.actor_local(states)
        
        if agent_num == 0:
            actor_predicted = torch.cat((actor_predicted,actions[:,2:]),dim=1)
        else:
            actor_predicted = torch.cat((actions[:,:2],actor_predicted),dim=1)
        
        actor_loss = -self.critic_local(states,actor_predicted).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        self.soft_update(self.critic_local,self.critic_target,TAU)
        self.soft_update(self.actor_local,self.actor_target,TAU)
        
        self.epsilon -=  self.eps_decay
        self.epsilon = max(self.epsilon,0) ## 0 --> assuming eps_final would be 0 
        self.noise.reset_internal_state()
    
    def soft_update(self,local_model,target_model,tau):
        for target_param,local_param in zip(target_model.parameters(),
                                            local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 - tau) * target_param.data)
                
                

        
        
        
     
        
        
        
        