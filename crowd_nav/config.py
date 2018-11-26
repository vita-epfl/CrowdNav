class Config(object):
    pass


class EnvConfig(object):
    env = Config()
    env.time_limit = 30
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500
    env.randomize_attributes = False
    env.robot_sensor_range = 5

    reward = Config()
    reward.success_reward = 1
    reward.collision_penalty = -0.25
    reward.discomfort_dist = 0.2
    reward.discomfort_penalty_factor = 0.5

    sim = Config()
    sim.train_val_scenario = 'square_crossing'
    sim.test_scenario = 'square_crossing'
    sim.square_width = 20
    sim.circle_radius = 5
    sim.human_num = 20
    sim.centralized_planning = True

    humans = Config()
    humans.visible = True
    humans.policy = 'socialforce'
    humans.radius = 0.3
    humans.v_pref = 1
    humans.sensor = 'coordinates'

    robot = Config()
    robot.visible = False
    robot.policy = 'none'
    robot.radius = 0.3
    robot.v_pref = 1
    robot.sensor = 'coordinates'

    def __init__(self, debug=False):
        if debug:
            self.env.val_size = 1
            self.env.test_size = 1


class PolicyConfig(object):
    rl = Config()
    rl.gamma = 0.9

    om = Config()
    om.cell_num = 4
    om.cell_size = 1
    om.om_channel_size = 3

    action_space = Config()
    action_space.kinematics = 'unicycle'
    action_space.speed_samples = 5
    action_space.rotation_samples = 7
    action_space.sampling = 'exponential'
    action_space.query_env = True

    cadrl = Config()
    cadrl.mlp_dims = [150, 100, 100, 1]
    cadrl.multiagent_training = False

    lstm_rl = Config()
    lstm_rl.global_state_dim = 50
    lstm_rl.mlp1_dims = [150, 100, 100, 50]
    lstm_rl.mlp2_dims = [150, 100, 100, 1]
    lstm_rl.multiagent_training = True
    lstm_rl.with_om = False
    lstm_rl.with_interaction_module = True

    srl = Config()
    srl.mlp1_dims = [150, 100, 100, 50]
    srl.mlp2_dims = [150, 100, 100, 1]
    srl.multiagent_training = True
    srl.with_om = False

    sarl = Config()
    sarl.mlp1_dims = [150, 100]
    sarl.mlp2_dims = [100, 50]
    sarl.attention_dims = [100, 100, 1]
    sarl.mlp3_dims = [150, 100, 100, 1]
    sarl.multiagent_training = True
    sarl.with_om = False
    sarl.with_global_state = False

    def __init__(self, debug=False):
        pass


class TrainConfig(object):
    trainer = Config()
    trainer.batch_size = 100

    imitation_learning = Config()
    imitation_learning.il_episodes = 2000
    imitation_learning.il_policy = 'orca'
    imitation_learning.il_epochs = 50
    imitation_learning.il_learning_rate = 0.01
    imitation_learning.safety_space = 0.15

    train = Config()
    train.rl_learning_rate = 0.001
    # number of batches to train at the end of training episode
    train.train_batches = 100
    # training episodes in outer loop
    train.train_episodes = 10000
    # number of episodes sampled in one training episode
    train.sample_episodes = 1
    train.target_update_interval = 50
    train.evaluation_interval = 1000
    # the memory pool can roughly store 2K episodes, total size = episodes * 50
    train.capacity = 100000
    train.epsilon_start = 0.5
    train.epsilon_end = 0.1
    train.epsilon_decay = 4000
    train.checkpoint_interval = 1000

    def __init__(self, debug=False):
        if debug:
            self.imitation_learning.il_episodes = 20
            self.train.train_episodes = 1
            self.train.checkpoint_interval = self.train.train_episodes
