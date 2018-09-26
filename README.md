# Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning
This repository contains the codes for our ICRA 2018 submission.

For more details, please refer to the [paper](https://arxiv.org/abs/1809.08835). 

If you find the codes or paper useful for your research, please cite our paper:
```
@misc{1809.08835,
Author = {Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
Title = {Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
Year = {2018},
Eprint = {arXiv:1809.08835},
}
```

## Abstract
Mobility in an effective and socially-compliant manner is an essential yet challenging task for robots operating in crowded spaces. 
Recent works have shown the power of deep reinforcement learning techniques to learn socially cooperative policies. 
However, their cooperation ability deteriorates as the crowd grows since they typically relax the problem as a one-way Human-Robot interaction problem. 
In this work, we want to go beyond first-order Human-Robot interaction and more explicitly model Crowd-Robot Interaction (CRI). 
We propose to (i) rethink pairwise interactions with a self-attention mechanism, and 
(ii) jointly model Human-Robot as well as Human-Human interactions in the deep reinforcement learning framework. 
Our model captures the Human-Human interactions occurring in dense crowds that indirectly affects the robot's anticipation capability. 
Our proposed attentive pooling mechanism learns the collective importance of neighboring humans with respect to their future states. 
Various experiments demonstrate that our model can anticipate human dynamics and navigate in crowds with time efficiency, 
outperforming state-of-the-art methods.


## Method Overview
<img src="https://i.imgur.com/YOPHXD1.png" width="600" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```
3. Install all the dependencies listed in requirements.txt
```
pip install -r requirements.txt
```


## Getting started
This repository are organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. All the commands 
below should be run inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy om-sarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy om-sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy om-sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training curve
```
python utils/plot.py data/output/output.log
```

## Simulation Videos
Qualitative comparison between CADRL, LSTM-RL, SARL and OM-SARL


<img src="https://i.imgur.com/f8ES7Lb.gif" height="300" />
<img src="https://i.imgur.com/6Pe3Xlp.gif" height="300" />
<img src="https://i.imgur.com/ktRTdiD.gif" height="300" />
<img src="https://i.imgur.com/JKW1wl8.gif" height="300" />


## Simulation Framework
### Environment
The environment contains n+1 agents. N of them are humans controlled by certain unknown
policy. The other is robot and it's controlled by one known policy.
The environment is built on top of OpenAI gym library, and has implemented two abstract methods.
* reset(): the environment will reset positions for all the agents and return observation 
for robot. Observation for one agent is the observable states of all other agents.
* step(action): taking action of the robot as input, the environment computes observation
for each agent and call agent.act(observation) to get actions of agents. Then environment detects
whether there is a collision between agents. If not, the states of agents will be updated. Then 
observation, reward, done will be returned.


### Agent
Agent is a base class, and has two derived class of human and robot. Agent class holds
all the physical properties of an agent, including position, velocity, orientation, policy and etc.
* visibility: humans are always visible, but robot can be set to be visible or invisible
* sensor: can be either visual input or coordinate input
* kinematics: can be either holonomic (move in any direction) or unicycle (has rotation constraints)
* act(observation): transform observation to state and pass it to policy


### Policy
Policy takes state as input and output an action. Current available policies:
* ORCA: compute collision-free velocity under the reciprocal assumption
* CADRL: learn a value network to predict the value of a state and during inference it predicts action for the most important human
* LSTM-RL: use lstm to encode the human states into one fixed-length vector
* SARL: use pairwise interaction module to model human-robot interaction and use self-attention to aggregate humans' information
* OM-SARL: extend SARL by encoding intra-human interaction with a local map


### State
There are multiple definition of states in different cases. The state of an agent representing all
the knowledge of environments is defined as JointState, and it's different from the state of the whole environment.
* ObservableState: position, velocity, radius of one agent
* FullState: position, velocity, radius, goal position, preferred velocity, rotation
* DualState: concatenation of one agent's full state and one another agent's observable state
* JoinState: concatenation of one agent's full state and all other agents' observable states 


### Action
There are two types of actions depending on what kinematics constraint the agent has.
* ActionXY: (vx, vy) if kinematics == 'holonomic'
* ActionRot: (velocity, rotation angle) if kinematics == 'unicycle'

