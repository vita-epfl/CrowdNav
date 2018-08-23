# DyNav
Robot navigation in an unknown and dynamic environment


## Installation
1. Set up [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install environment and dynav into pip
```
pip install -e .
```
3. Install all the dependencies as listed in requirements.txt
```
pip install -r requirements.txt
```
4. Install trajnettools and place trajnet data (train/ and val/) under gym_crowd/envs/data/trajnet

## Getting started
The codes are organized in two parts: gym_crowd/ folder contains the environment and
dynav/ folder contains codes for training and testing the algorithms. All the commands 
below should be run inside the dynav/ folder.


1. Train a trainable policy.
```
python train.py --policy cadrl
```
2. Test policies in VAL or TEST test cases. Specify il flag for test imitation learning trained model.
```
python test.py --policy orca --phase val
python test.py --policy orca --phase test
python test.py --policy cadrl --model_dir data/output --phase val
python test.py --policy cadrl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase val --visualize
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy cadrl --model_dir data/output --phase val --visualize
python test.py --policy cadrl --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training log
```
python utils/plot.py data/output/output.log
```


## Framework Overview
### Environment
The environment contains n+1 agents. N of them are pedestrians controlled by certain unknown
but fixed policy. The other is navigator and it's controlled by one known policy.
The environment is derived from OpenAI gym library, and has implemented two abstract methods.
* reset(): the environment will reset positions for all the agents and return observation 
for navigator. Observation for one agent is the observable states of all other agents.
* step(action): taking action of the navigator as input, the environment computes observation
for each agent and call agent.act(observation) to get actions of agents. Then environment detects
whether there is a collision between agents. If not, the states of agents will be updated. Then 
observation, reward, done will be returned.


### Agent
Agent is a base class, and has two derived class of pedestrian and navigator. Agent class holds
all the physical properties of an agent, including position, velocity, rotation, policy and etc.
* visibility: pedestrians are always visible, but navigator is only visible if the policy of 
pedestrian is not controlled by trajnet
* sensor: can be either visual input or coordinate input
* kinematics: can be either holonomic (travel in any direction) or unicycle (has rotation constraints)
* act(observation): transform observation to state and pass it to policy

### Policy
Policy takes state as input and output an action. Current available policies:
* potential field(TODO): head straight to goal until collision happens or goal is reached
* ORCA: compute collision-free velocity under the assumption each agent will take half responsibility
* CADRL: learn a value network to predict the value of a state and during inference,
the action with maximum one step lookahead value will be chosen.
* SRL: do social pooling for surrounding pedestrians and use that as state representation

### State
There are multiple definition of states in different cases. The state of an agent representing all
the knowledge of environments is defined as JointState, and it's different from the state of the whole environment.
* ObservableState: position, velocity, radius of one agent
* FullState: position, velocity, radius, goal position, preferred velocity, rotation
* DualState: concatenation of one agent's full state and one another agent's observable state
* JoinState: concatenation of one agent's full state and all other agents' observable states 

### Action
There are two types of actions depending on what kinematics constraint the agent has.
* ActionXY: (x-axis velocity, y-axis velocity) if kinematics == 'holonomic'
* ActionRot: (velocity, rotation angle) if kinematics == 'unicycle'

### Phase
Environment has different setup for different phases and the behavior of policy also 
depends what phase it is in.
#### Simulated data
* Train: environment randomly initialized the position and goal for pedestrians (on a circle) and RL policy
uses epsilon-greedy to stabilize the training.
* Val: environment is the same as train, but RL policy doesn't use epsilon-greedy.
* Test: environment has some fixed test cases and RL policy doesn't use epsilon-greed. 
#### Trajnet data
* Train: environment sets the pedestrian positions according to the train split of one dataset
* Val: environment sets the pedestrian positions according to the val split of the same dataset
* Test: environment sets the pedestrian positions according to the val split of another dataset

### Evaluation
The success rate, collision rate and extra time to reach goal are used to measure
the methods.
