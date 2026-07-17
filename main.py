"""Project entry point: parse args, create SMAC envs, and launch runs."""

import random

import numpy as np
import torch

from common.arguments import (
    get_centralv_args,
    get_coma_args,
    get_commnet_args,
    get_common_args,
    get_env_args,
    get_g2anet_args,
    get_mixer_args,
    get_reinforce_args,
)
from common.utils import reserve_next_run
from runner import Runner
from smac.env import StarCraft2Env


def set_random_seed(seed):
    """Set random seeds used by Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_args():
    """Parse CLI args and add algorithm-specific defaults."""
    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)
    return args


if __name__ == '__main__':
    for i in range(5):
        args = build_args()

        if not args.evaluate:
            run_num = reserve_next_run(args)
        else:
            args.run_num = i
            run_num = args.run_num

        args.seed += run_num
        set_random_seed(args.seed)

        env = StarCraft2Env(**get_env_args(args))
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]

        runner = Runner(env, args)
        if not args.evaluate:
            print('Start {} on {} with run_{}'.format(args.alg, args.map, run_num))
            runner.run(run_num)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
