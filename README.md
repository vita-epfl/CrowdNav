# DyNav
Robot navigation in an unknown and dynamic environment


## Installation
1. Set up [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install gym environment and dynav
```
pip install -e .
```
3. Train RL algorithm
```
cd dynav/ & python train.py
```
3. Test and visualize
```
cd dynav/ & python test.py --visualize
```


## Details
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
* linear: head straight to goal until collision happens or goal is reached
* orca: compute collision-free velocity under the assumption each agent will take half responsibility
* value network: learn a value network to predict the value of a state and during inference,
the action with maximum one step lookahead value will be chosen.

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

### Evaluation
The success rate, collision rate and extra time to reach goal are used to measure
the methods.
