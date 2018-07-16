# DyNav
Robot navigation in an unknown and dynamic environment


## Installation
1. Set up [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install gym environment and dynav
```
pip install -e .
```
3. Test algorithm
```
cd dynav/ & python test.py
```


## Details
### Environment
There are n pedestrians in the scene with unknown policies A1, A2, ...,
An and one agent with known policy B. Unknown polices can be linear, random,
ORCA or following the exact trajectories in trajnet)

### Agent
* Agent is controlled by policy B, which is the policy we want to learn
and optimize. Policy could be hand-crafted rule like ORCA or a learned
policy like RL.
* Agent can have either the visual perception or the coordinate input.
* Agent is invisible if the pedestrian policies are following trajnet.


### Evaluation
The success rate, collision rate and extra time to reach goal are used to measure
the methods.
