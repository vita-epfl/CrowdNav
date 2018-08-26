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
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False):
        self.navigator.policy.set_phase(phase)
        navigator_times = []
        ped_times = []
        last_ped_time = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
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

                if info == 'too close':
                    too_close += 1

            if info == 'reach goal':
                success += 1
                navigator_times.append(self.env.global_time)
                if self.navigator.visible and phase in ['val', 'test']:
                    times = self.env.get_ped_times()
                    ped_times += times
                    last_ped_time.append(max(times))
            elif info == 'collision':
                collision += 1
                collision_cases.append(i)
            elif info == 'timeout':
                timeout += 1
                timeout_cases.append(i)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if (imitation_learning and info == 'reach goal') or \
                   (not imitation_learning and info in ['reach goal', 'collision']):
                    # only provide successful demonstrations in imitation learning
                    # only add positive(success) or negative(collision) experience in reinforcement learning
                    self.update_memory(states, actions, rewards, imitation_learning)

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(navigator_times) / len(navigator_times) if len(navigator_times) != 0 else 0

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, average time to reach goal: {:.2f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time))
        if self.navigator.visible and phase in ['val', 'test']:
            logging.info('Average time for peds to reach goal: {:.2f}'.format(average(ped_times)))
            logging.info('Average time for last ped to reach goal: {:.2f}'.format(average(last_ped_time)))

        if phase in ['test']:
            logging.info('Average times of navigator getting too close to peds per second: {:.2f}'.
                         format(too_close/sum(navigator_times) if len(navigator_times) != 0 else 0))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        return navigator_times, ped_times

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i in range(len(states)):
            state = states[i]
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # in imitation learning, the value of state is defined based on the time to reach the goal
                state = self.target_policy.transform(state)
                value = pow(self.gamma, (len(states) - 1 - i) * self.navigator.time_step * self.navigator.v_pref)
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.navigator.time_step * self.navigator.v_pref)
                    value = reward + gamma_bar * self.stabilized_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            self.memory.push((state, value))


def average(li):
    if len(li) == 0:
        return 0
    else:
        return sum(li) / len(li)
