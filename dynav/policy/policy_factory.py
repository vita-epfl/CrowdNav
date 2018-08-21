from gym_crowd.envs.policy.policy_factory import policy_factory
from dynav.policy.cadrl import CADRL
from dynav.policy.lstm_rl import LstmRL
from dynav.policy.srl import SRL
from dynav.policy.sarl import SARL
from dynav.policy.om_sarl import OmSarl

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['srl'] = SRL
policy_factory['sarl'] = SARL
policy_factory['om_sarl'] = OmSarl
