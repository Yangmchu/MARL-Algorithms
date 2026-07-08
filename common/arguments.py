"""集中定义环境、训练流程及各 MARL 算法的超参数。"""

import argparse

"""
Here are the param for the training

"""


def add_env_args(parser):
    """Add SMAC environment arguments to the parser."""
    parser.add_argument('--continuing_episode', type=bool, default=False, help='whether to continue after episode limit')
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default=None, help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--move_amount', type=int, default=2, help='movement amount for the environment')
    parser.add_argument('--obs_all_health', type=bool, default=True, help='whether observations include all unit health')
    parser.add_argument('--obs_instead_of_state', type=bool, default=False, help='whether to use observations instead of state')
    parser.add_argument('--obs_last_action', type=bool, default=False, help='whether observations include last actions')
    parser.add_argument('--obs_own_health', type=bool, default=True, help='whether observations include own health')
    parser.add_argument('--obs_pathing_grid', type=bool, default=False, help='whether observations include the pathing grid')
    parser.add_argument('--obs_terrain_height', type=bool, default=False, help='whether observations include terrain height')
    parser.add_argument('--obs_timestep_number', type=bool, default=False, help='whether observations include timestep number')
    parser.add_argument('--reward_death_value', type=float, default=10, help='reward for killing a unit')
    parser.add_argument('--reward_defeat', type=float, default=0, help='reward for defeat')
    parser.add_argument('--reward_negative_scale', type=float, default=0.5, help='scale for negative rewards')
    parser.add_argument('--reward_only_positive', type=bool, default=True, help='whether rewards are only positive')
    parser.add_argument('--reward_scale', type=bool, default=True, help='whether to scale rewards')
    parser.add_argument('--reward_scale_rate', type=float, default=20, help='reward scaling rate')
    parser.add_argument('--reward_sparse', type=bool, default=False, help='whether to use sparse rewards')
    parser.add_argument('--reward_win', type=float, default=200, help='reward for winning')
    parser.add_argument('--replay_prefix', type=str, default='', help='prefix for saved replays')
    parser.add_argument('--state_last_action', type=bool, default=True, help='whether state includes last actions')
    parser.add_argument('--state_timestep_number', type=bool, default=False, help='whether state includes timestep number')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--heuristic_ai', type=bool, default=False, help='whether to use heuristic AI')
    parser.add_argument('--debug', type=bool, default=False, help='whether to enable environment debug mode')
    return parser


def get_common_args():
    """解析所有算法共享的命令行参数。"""
    parser = argparse.ArgumentParser()
    # the environment setting
    parser = add_env_args(parser)
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=1000000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--save_run', type=int, default=None, help='run id used when saving results and models')
    parser.add_argument('--load_run', type=int, default=None, help='run id used when loading models; default loads the latest run')
    parser.add_argument('--load_checkpoint', type=int, default=None, help='checkpoint id used when loading models; default loads the latest checkpoint')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args


def get_env_args(args):
    """Build keyword arguments for the SMAC environment."""
    return {
        'map_name': args.map,
        'step_mul': args.step_mul,
        'difficulty': args.difficulty,
        'game_version': args.game_version,
        'replay_dir': args.replay_dir,
        'continuing_episode': args.continuing_episode,
        'move_amount': args.move_amount,
        'obs_all_health': args.obs_all_health,
        'obs_instead_of_state': args.obs_instead_of_state,
        'obs_last_action': args.obs_last_action,
        'obs_own_health': args.obs_own_health,
        'obs_pathing_grid': args.obs_pathing_grid,
        'obs_terrain_height': args.obs_terrain_height,
        'obs_timestep_number': args.obs_timestep_number,
        'reward_death_value': args.reward_death_value,
        'reward_defeat': args.reward_defeat,
        'reward_negative_scale': args.reward_negative_scale,
        'reward_only_positive': args.reward_only_positive,
        'reward_scale': args.reward_scale,
        'reward_scale_rate': args.reward_scale_rate,
        'reward_sparse': args.reward_sparse,
        'reward_win': args.reward_win,
        'replay_prefix': args.replay_prefix,
        'state_last_action': args.state_last_action,
        'state_timestep_number': args.state_timestep_number,
        'heuristic_ai': args.heuristic_ai,
        'debug': args.debug,
    }


# arguments of coma
def get_coma_args(args):
    """补充 COMA 的 Actor-Critic、TD(lambda) 与探索参数。"""
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    """补充 IQL/VDN/QMIX/QTRAN/MAVEN 共用的值分解训练参数。"""
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args


# arguments of central_v
def get_centralv_args(args):
    """补充 Central-V 的 Actor-Critic 参数。"""
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    """补充 REINFORCE 的策略网络与探索参数。"""
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    """根据地图规模设置 CommNet 的通信轮数。"""
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    """设置 G2ANet 的注意力维度及是否启用硬注意力。"""
    args.attention_dim = 32
    args.hard = True
    return args

