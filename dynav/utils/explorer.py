import logging
import torch
import copy


class Explorer(object):
    def __init__(self, env, navigator, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.navigator = navigator
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.stabilized_model = None

    def update_stabilized_model(self, stabilized_model):
        self.stabilized_model = copy.deepcopy(stabilized_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, print_failure=False):
        self.navigator.policy.set_phase(phase)
        times = []
        success = 0
        collision = 0
        timeout = 0
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            while not done:
                action = self.navigator.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.navigator.policy.last_state)
                actions.append(action)
                rewards.append(reward)

            if update_memory:
                self.update_memory(states, actions, rewards, imitation_learning)

            if info == 'reach goal':
                success += 1
                times.append(self.env.timer)
            elif info == 'collision':
                collision += 1
                collision_cases.append(i)
            elif info == 'timeout':
                timeout += 1
                timeout_cases.append(i)
            else:
                raise ValueError('Invalid info from environment')

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        if len(times) == 0:
            average_time = 0
        else:
            average_time = sum(times) / len(times)

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, average time to reach goal: {:.2f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, average_time))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        steps = len(states)
        for i in range(steps-1):
            state = states[i]
            next_state = states[i]
            reward = rewards[i]

            if imitation_learning:
                # in imitation learning, the value of state is defined based on the time to reach the goal
                state = self.target_policy.transform(state)
                value = pow(self.gamma, (steps - 1 - i) * self.navigator.time_step * self.navigator.v_pref)
            else:
                gamma_bar = pow(self.gamma, self.navigator.time_step * self.navigator.v_pref)
                value = reward + gamma_bar * self.stabilized_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            self.memory.push((state, value))
