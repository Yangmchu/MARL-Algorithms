"""Utilities for collecting SMAC episodes in independent worker processes."""

import copy
import random

import numpy as np
import torch
from torch import nn

from agent.agent import Agents, CommAgents
from common.arguments import get_env_args
from common.rollout import RolloutWorker, CommRolloutWorker
from smac.env import StarCraft2Env


def _module_state_dicts(policy):
    """Return CPU state_dict snapshots for all torch modules on a policy."""
    states = {}
    for name, value in policy.__dict__.items():
        if isinstance(value, nn.Module):
            states[name] = {
                key: tensor.detach().cpu().clone()
                for key, tensor in value.state_dict().items()
            }
    return states


def get_policy_state(policy):
    """Build a pickle-friendly policy snapshot for rollout workers."""
    return _module_state_dicts(policy)


def load_policy_state(policy, policy_state):
    """Load a policy snapshot into matching modules in a worker policy."""
    for name, state_dict in policy_state.items():
        module = getattr(policy, name, None)
        if isinstance(module, nn.Module):
            module.load_state_dict(state_dict)
            module.eval()


def _set_worker_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _get_worker_seed(args, worker_id):
    run_num = int(getattr(args, 'run_num', 0))
    return int(args.seed) + run_num * 10000 + int(worker_id) + 1


def rollout_worker(worker_id, args, task_pipe, result_pipe):
    """Persistent rollout worker controlled by the main process through pipes."""
    worker_args = copy.deepcopy(args)
    worker_args.cuda = False
    worker_args.device = 'cpu'
    worker_args.load_model = False
    worker_args.evaluate = False
    worker_args.replay_dir = ''
    worker_args.seed = _get_worker_seed(args, worker_id)

    _set_worker_seed(worker_args.seed)
    env = StarCraft2Env(**get_env_args(worker_args))
    try:
        if worker_args.alg.find('commnet') > -1 or worker_args.alg.find('g2anet') > -1:
            agents = CommAgents(worker_args)
            rollout = CommRolloutWorker(env, agents, worker_args)
        else:
            agents = Agents(worker_args)
            rollout = RolloutWorker(env, agents, worker_args)

        while True:
            task = task_pipe.recv()
            if task == 'TERMINATE':
                break
            policy_state, epsilon, episode_idx = task
            load_policy_state(agents.policy, policy_state)
            rollout.epsilon = epsilon
            with torch.no_grad():
                result = rollout.generate_episode(episode_idx, evaluate=False)
            result_pipe.send(result)
    finally:
        env.close()
