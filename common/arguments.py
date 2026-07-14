"""集中定义环境、训练流程及各 MARL 算法的超参数。"""

import argparse

"""
Here are the param for the training

"""


def str2bool(value):
    """Parse boolean values from command line strings."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _set_default(args, name, value):
    """Set an argument default only when it was not parsed from the command line."""
    if not hasattr(args, name):
        setattr(args, name, value)


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
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    parser.add_argument('--heuristic_ai', type=bool, default=False, help='whether to use heuristic AI')
    parser.add_argument('--debug', type=bool, default=False, help='whether to enable environment debug mode')
    return parser


def add_mixer_args(parser):
    """Add command line arguments shared by value-decomposition algorithms."""
    # network
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='hidden dimension of agent RNN')
    parser.add_argument('--qmix_hidden_dim', type=int, default=32, help='hidden dimension of QMIX mixer')
    parser.add_argument('--two_hyper_layers', type=str2bool, default=False, help='whether QMIX uses two hypernet layers')
    parser.add_argument('--hyper_hidden_dim', type=int, default=64, help='hidden dimension of QMIX hypernet')
    parser.add_argument('--qtran_hidden_dim', type=int, default=64, help='hidden dimension of QTRAN networks')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for value-based policies')

    # epsilon greedy
    parser.add_argument('--epsilon', type=float, default=1, help='initial epsilon for epsilon-greedy')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='minimum epsilon for epsilon-greedy')
    parser.add_argument('--anneal_steps', type=int, default=50000, help='number of steps used to anneal epsilon')
    parser.add_argument('--anneal_epsilon', type=float, default=None, help='epsilon decay per step; default is computed')
    parser.add_argument('--epsilon_anneal_scale', type=str, default='step', help='epsilon anneal scale')

    # training and replay
    parser.add_argument('--train_steps', type=int, default=1, help='number of train steps per epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size sampled from replay buffer')
    parser.add_argument('--buffer_size', type=int, default=int(5e3), help='replay buffer size')
    parser.add_argument('--save_cycle', type=int, default=5000, help='how often to save the model')
    parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update target networks')
    parser.add_argument('--grad_norm_clip', type=float, default=10, help='gradient clipping norm')

    # QTRAN
    parser.add_argument('--lambda_opt', type=float, default=1, help='QTRAN optimality loss coefficient')
    parser.add_argument('--lambda_nopt', type=float, default=1, help='QTRAN non-optimality loss coefficient')

    # MAVEN
    parser.add_argument('--noise_dim', type=int, default=16, help='MAVEN noise dimension')
    parser.add_argument('--lambda_mi', type=float, default=0.001, help='MAVEN mutual information coefficient')
    parser.add_argument('--lambda_ql', type=float, default=1, help='MAVEN Q-learning loss coefficient')
    parser.add_argument('--entropy_coefficient', type=float, default=0.001, help='MAVEN entropy coefficient')
    return parser


def add_qplex_args(parser):
    """Add QPLEX-specific command line arguments."""
    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='QPLEX mixer embedding dimension')
    parser.add_argument('--hypernet_embed', type=int, default=64, help='QPLEX transformation hypernet dimension')
    parser.add_argument('--adv_hypernet_layers', type=int, default=2, help='number of QPLEX advantage hypernet layers')
    parser.add_argument('--adv_hypernet_embed', type=int, default=32, help='QPLEX advantage hypernet dimension')
    parser.add_argument('--num_kernel', type=int, default=4, help='number of QPLEX advantage weight kernels')
    parser.add_argument('--is_minus_one', type=str2bool, default=True, help='whether to use lambda_i - 1 in advantage')
    parser.add_argument('--is_stop_gradient', type=str2bool, default=False, help='whether to detach QPLEX advantage values')
    parser.add_argument('--weighted_head', type=str2bool, default=True, help='whether to use QPLEX transformation network')
    parser.add_argument('--double_q', type=str2bool, default=True, help='whether QPLEX uses double Q targets')
    return parser


def get_common_args():
    """解析所有算法共享的命令行参数。"""
    parser = argparse.ArgumentParser()
    # the environment setting
    parser = add_env_args(parser)
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--alg', type=str, default='qplex', help='the algorithm to train the agent')
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
    parser.add_argument('--device', type=str, default=None, help='torch device, e.g. cpu, cuda:0, cuda:1')
    known_args, _ = parser.parse_known_args()
    mixer_algs = ['vdn', 'iql', 'qmix', 'qtran_base', 'qtran_alt', 'maven', 'qplex']
    if known_args.alg in mixer_algs:
        parser = add_mixer_args(parser)
    if known_args.alg == 'qplex':
        parser = add_qplex_args(parser)
    args = parser.parse_args()
    return args


def get_env_args(args):
    """Build keyword arguments for the SMAC environment."""
    return {
        'map_name': args.map,
        'seed': args.seed,
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
    _set_default(args, 'rnn_hidden_dim', 64)
    _set_default(args, 'qmix_hidden_dim', 32)
    _set_default(args, 'two_hyper_layers', False)
    _set_default(args, 'hyper_hidden_dim', 64)
    _set_default(args, 'qtran_hidden_dim', 64)
    _set_default(args, 'lr', 5e-4)
    if args.alg == 'qplex':
        _set_default(args, 'mixing_embed_dim', 32)
        _set_default(args, 'hypernet_embed', 64)
        _set_default(args, 'adv_hypernet_layers', 2)
        _set_default(args, 'adv_hypernet_embed', 32)
        _set_default(args, 'num_kernel', 4)
        _set_default(args, 'is_minus_one', True)
        _set_default(args, 'is_stop_gradient', False)
        _set_default(args, 'weighted_head', True)
        _set_default(args, 'double_q', True)

    # epsilon greedy
    _set_default(args, 'epsilon', 1)
    _set_default(args, 'min_epsilon', 0.05)
    _set_default(args, 'anneal_steps', 50000)
    if not hasattr(args, 'anneal_epsilon') or args.anneal_epsilon is None:
        args.anneal_epsilon = (args.epsilon - args.min_epsilon) / args.anneal_steps
    _set_default(args, 'epsilon_anneal_scale', 'step')

    # the number of the train steps in one epoch
    _set_default(args, 'train_steps', 1)

    # experience replay
    _set_default(args, 'batch_size', 32)
    _set_default(args, 'buffer_size', int(5e3))

    # how often to save the model
    _set_default(args, 'save_cycle', 5000)

    # how often to update the target_net
    _set_default(args, 'target_update_cycle', 200)

    # QTRAN lambda
    _set_default(args, 'lambda_opt', 1)
    _set_default(args, 'lambda_nopt', 1)

    # prevent gradient explosion
    _set_default(args, 'grad_norm_clip', 10)

    # MAVEN
    _set_default(args, 'noise_dim', 16)
    _set_default(args, 'lambda_mi', 0.001)
    _set_default(args, 'lambda_ql', 1)
    _set_default(args, 'entropy_coefficient', 0.001)
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

