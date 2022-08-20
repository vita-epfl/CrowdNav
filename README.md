# CrowdNav
## Abstract
We envision a future with safe, interactive robots which can co-exist with people. Therefore, we choose the topic "Social Navigation". Some previous works have been done by Chen et. al. [[1]](https://arxiv.org/pdf/1809.08835.pdf) , including the benchmark environment and their method SARL. Our works focus mostly on making the environment more realistic and enabling the agent to perform navigation in such a complicated environment. 


## Graph Attention Network for Social Navigation (GAT4SN)
![image info](./pictures/deep_v_network.png)

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy gat4sn
```
2. Test policies with 500 test cases.
```
python test.py --policy gat4sn --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy gat4sn --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
GAT4SN             | SARL
:-------------------------:|:-------------------------:
![image info](./pictures/gat4sn_test_case_40_H20.gif)|![image info](./pictures/sarl_test_case_40_H20.gif)
![image info](./pictures/test_183_gat4sn.gif)|![image info](./pictures/test_183_sarl.gif)
![image info](./pictures/test_178_gat4sn.gif)|![image info](./pictures/test_178_sarl.gif)

## More Complicated Environment?
Only Humans             | With Static Obstacles
:-------------------------:|:-------------------------:
![image info](./pictures/H30O0_0.gif)|![image info](./pictures/H10O10_1.gif)
![image info](./pictures/gat4sn_test_case_40_H20.gif)|![image info](./pictures/H10O10_0.gif)
