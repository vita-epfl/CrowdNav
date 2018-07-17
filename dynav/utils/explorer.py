import logging
import torch
import copy


class Explorer(object):
    def __init__(self, env, navigator, device, memory=None, gamma=None):
        self.env = env
        self.navigator = navigator
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.stabilized_model = None

    def update_stabilized_model(self, stabilized_model):
        self.stabilized_model = copy.deepcopy(stabilized_model)

    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None):
        self.navigator.policy.set_phase(phase)
        times = []
        success = 0
        collision = 0
        for _ in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            rewards = []
            while not done:
                action = self.navigator.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.navigator.policy.last_state)
                rewards.append(reward)

            if update_memory:
                self.update_memory(states, rewards, imitation_learning)

            if info == 'reach goal':
                success += 1
                times.append(self.env.timer)
            elif info == 'collision':
                collision += 1

        success_rate = success / k
        collision_rate = collision / k
        if len(times) == 0:
            average_time = 0
        else:
            average_time = sum(times) / len(times)

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{} {}has success rate: {:.2f}, collision rate: {:.2f}, average time to reach goal: {:.0f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, average_time))

    def update_memory(self, states, rewards, imitation_learning):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

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
