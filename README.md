# Colloboration-Competetion-P5-RLND

# Description:
In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

<img src = "/trained_agent.gif" width="75%" align="center" alt="robotic_arms" title="Robotic Arms"/>

### Exploring Unity Agents:

``` Unity Academy name: Academy
        Number of Brains: 1
        Number of External Brains : 1
        Lesson number : 0
        Reset Parameters :
Unity brain name: TennisBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 8
        Number of stacked Vector Observation: 3
        Vector Action space type: continuous
        Vector Action space size (per agent): 2
        Vector Action descriptions: ,
```
        
# Goal:

Training a equi-level RL agents to play a tennis as comparable as humans over the nets.If the trained agent hits a ball successfully over the net it will be rewarded by +0.1 , whereas if it misses the ball , it wil be rewarded by -0.01 

The task is performed in the environment is an episodic and The goal is to get the average score of +0.5 over consecutive episodes!

### Project Info - Getting started :

*Step 1*: Download the Unity Agents that suits your OS Platform
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

After downloading place the environment file path in Tennis.ipynb notebook file


*Step 2*: Running all the cells in the notebook will start training the model by the way you can train an RL agent to play tennis

# Instructions:

### Project Archi:


Files        | Description
------------ | -------------
Tennis.ipynb | Jupyter Notebook (for training)
maddpg_Agent.py |  MADDPG Agent class
actor_critic.py |  Actor and Critic network archi
Buffered_memory.py |  replay buffer class
OrnsteinUhlenbeck_noise.py |  OUInoise class
ckpt_critic1.pth | critic trained model for agent 1 
ckpt_actor1.pth | actor trained model for agent 1
ckpt_critic2.pth | critic trained model for agent 2 
ckpt_actor2.pth | actor trained model for agent 2
Report.md |  Description abt project implementation 

