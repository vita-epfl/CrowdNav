import logging
import torch
import copy
from gym_crowd.envs.utils.info import *


class Explorer(object):
    def __init__(self, env, navigator, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.navigator = navigator
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.navigator.policy.set_phase(phase)
        navigator_times = []
        all_nav_times = []
        ped_times = []
        all_ped_times = []
        last_ped_time = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
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

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                navigator_times.append(self.env.global_time)
                all_nav_times.append(self.env.global_time)
                if self.navigator.visible and phase in ['val', 'test']:
                    times = self.env.get_ped_times()
                    ped_times.append(average(times))
                    last_ped_time.append(max(times))
                else:
                    all_ped_times.append(self.env.time_limit)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                all_nav_times.append(self.env.time_limit)
                all_ped_times.append(self.env.time_limit)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                all_nav_times.append(self.env.time_limit)
                all_ped_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.navigator.time_step * self.navigator.v_pref) * reward
                                          for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(navigator_times) / len(navigator_times) if len(navigator_times) != 0 else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if self.navigator.visible and phase in ['val', 'test']:
            logging.info('Average time for peds to reach goal: {:.2f}'.format(average(ped_times)))
            logging.info('Average time for last ped to reach goal: {:.2f}'.format(average(last_ped_time)))
            logging.info('Frequency of being in danger: {:.2f} and average min separate distance in danger: {:.2f}'.
                         format(too_close/sum(navigator_times)*self.navigator.time_step, average(min_dist)))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        return all_nav_times, all_ped_times, cumulative_rewards

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i in range(len(states)):
            state = states[i]
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.navigator.time_step * self.navigator.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.navigator.time_step * self.navigator.v_pref) * reward
                             for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.navigator.time_step * self.navigator.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            self.memory.push((state, value))


def average(li):
    if len(li) == 0:
        return 0
    else:
        return sum(li) / len(li)
