import logging
import torch
import copy


class Explorer(object):
    def __init__(self, env, navigator, memory, gamma, device):
        self.env = env
        self.navigator = navigator
        self.memory = memory
        self.gamma = gamma
        self.stabilized_model = None
        self.device = device

    def update_stabilized_model(self, stabilized_model):
        self.stabilized_model = copy.deepcopy(stabilized_model)

    def run_k_episodes(self, k, phase, episode, update_memory=True, imitation_learning=False):
        self.navigator.policy.set_phase(phase)
        times = []
        succ = 0
        failure = 0
        for _ in range(k):
            # run one episode
            ob = self.env.reset(phase)
            timer = 0
            done = False
            states = []
            rewards = []
            while not done:
                action = self.navigator.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.navigator.policy.last_state)
                rewards.append(reward)
                timer += 1

            if update_memory:
                self.update_memory(states, rewards, imitation_learning)

            if info == 'reach goal':
                succ += 1
            elif info == 'collision':
                failure += 1
            times.append(timer)
        if len(times) == 0:
            average_time = 0
        else:
            average_time = sum(times) / len(times)
        logging.info('{} in episode {} has success rate: {:.2f}, failure rate: {:.2f}, '
                     'average time to reach goal: {:.0f}'.format(phase, episode, succ / k, failure / k, average_time))

    def update_memory(self, states, rewards, imitation_learning):
        steps = len(states)
        for i in range(steps-1):
            state = states[i]
            next_state = states[i]
            reward = rewards[i]

            if imitation_learning:
                # In imitation learning, the value of state is defined based on the time to reach the goal
                value = pow(self.gamma, (steps - 1 - i) * self.navigator.v_pref)
            else:
                value = reward + self.gamma * self.stabilized_model(torch.Tensor(next_state), self.device).data.item()
            state = torch.Tensor(state).to(self.device).squeeze()
            value = torch.Tensor([value]).to(self.device)
            self.memory.push((state, value))
