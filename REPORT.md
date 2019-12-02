# PROJECT SPECIFICATIONS:

### LEARNING ALGORITHM:
As a same spirit after continuing the P2 : Continous Control  I have proceeded the project with Multi Deep Determnisitic Policy Gradients(MADDPG)

As MADDPG indicates itself multiple agents ! I have implemented separate 2 agents the agents which colloborates together and compete with each other accumulates many experiences by progression updating the policies responsively.

In Actor-Critic Model , it is proceeded like agent's critic is always trained using the observations and gathered actions across the agents , whereas in agents actor is trained by its own observations alone.


### ALGO:
<img src = "/maddpg.png" width="75%" align="center" alt="maddpg_algo" title="maddpg_algo"/>

### Network Architecture:

* Actor 
  * Hidden_Layer1 = state_size*2,256-> ReLu activation 
  * Hidden_Layer2 = 256,128-> ReLu activation 
  * Output layer  = 128,2(action_size) --> TanH activation 
  
* Critic
  * Hidden_Layer1 = state_size*2,256 -> ReLu activation 
  * Hidden_Layer2 = 256+(action_size*2),128 -> ReLu activation 
  * Output layer  = 128,1(action_size --> Linear


Hyperparameters      | Fine Value
-------------------  | -------------
Replay Batch Size | 128
Replay Buffer Size | 1e5
Actor LR | 3e-3
Critic LR| 3e-3
TAU | 7e-2
GAMMA | 0.99
EPS_START | 5
EPS_END | 400
EPS_DECAY | 1 / (EPS_END * LEARN_NUM) 
LEARN_NUM | 5
UPDATE_EVERY | 1
OU_theta | 0.15
OU_sigma | 0.2
OU_MU | 0.0
Max_episodes | 6000



### Plot Of Rewards:

After training the model attaining the average score of +0.5 , by proving it as episodic task plotting the rewards graph for Episode Vs Score 

<img src = "/rewards_plot.png" width="75%" align="center" alt="robotic_arms" title="REWARD_PLOT"/>

Once it gets finished it will throw out 
```Environment solved in 1679 episodes!	Average Score: 0.502```

### Ideas for future work :
* Would love to go with PPO Proximal Policy Optimizatoin (PPO ) 
* prioritized Buffered Memory (replay buffer) -- Selecting the action based on priority respective to the actions (that has been experienced while exploring the action space)
* Controlling the Exploration vs Exploitation by tuning sigma , theta parameters for better exploration in the state space
* Tuning the hyperparmaters to attain maximum efficiency ( I have spent lot of time but i would love to) 

