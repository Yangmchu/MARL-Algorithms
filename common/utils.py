"""通用辅助函数：构造器参数保存装饰器和 TD(lambda) 目标计算。"""

import inspect
import functools
import os
import torch


def get_run_tag(args):
    """根据关键实验参数生成保存目录标签，便于区分不同配置的结果。"""
    tags = [
        'last_action_{}'.format(int(args.last_action)),
        'reuse_network_{}'.format(int(args.reuse_network))
    ]
    if hasattr(args, 'two_hyper_layers'):
        tags.append('two_hyper_layers_{}'.format(int(args.two_hyper_layers)))
    return '-'.join(tags)


def get_run_name(args):
    """返回当前独立实验的目录名，例如 run_0。"""
    run_num = getattr(args, 'save_run', None)
    if run_num is None:
        run_num = getattr(args, 'run_num', 0)
    return 'run_{}'.format(run_num)


def get_save_dir(root_dir, args):
    """生成带参数标签和独立实验编号的保存目录。"""
    return root_dir + '/' + args.alg + '/' + args.map + '/' + get_run_tag(args) + '/' + get_run_name(args)


def _get_run_base_dir(root_dir, args):
    return root_dir + '/' + args.alg + '/' + args.map + '/' + get_run_tag(args)


def _list_run_nums(base_dir):
    run_nums = []
    if not os.path.exists(base_dir):
        return run_nums
    for name in os.listdir(base_dir):
        path = base_dir + '/' + name
        if os.path.isdir(path) and name.startswith('run_'):
            try:
                run_nums.append(int(name.split('_')[-1]))
            except ValueError:
                pass
    return run_nums


def reserve_next_run(args):
    """Reserve the next run id atomically so concurrent jobs do not overwrite."""
    if getattr(args, 'save_run', None) is not None:
        args.run_num = args.save_run
        return args.run_num

    result_base_dir = _get_run_base_dir(args.result_dir, args)
    model_base_dir = _get_run_base_dir(args.model_dir, args)
    os.makedirs(result_base_dir, exist_ok=True)
    os.makedirs(model_base_dir, exist_ok=True)

    while True:
        used_run_nums = _list_run_nums(result_base_dir) + _list_run_nums(model_base_dir)
        next_run = max(used_run_nums) + 1 if used_run_nums else 0
        result_run_dir = result_base_dir + '/run_{}'.format(next_run)
        model_run_dir = model_base_dir + '/run_{}'.format(next_run)
        try:
            os.mkdir(result_run_dir)
        except FileExistsError:
            continue
        try:
            os.mkdir(model_run_dir)
        except FileExistsError:
            continue
        args.run_num = next_run
        return next_run


def get_load_dir(root_dir, args):
    """返回模型加载目录；未指定 load_run 时自动选择最新的 run_x。"""
    base_dir = _get_run_base_dir(root_dir, args)
    load_run = getattr(args, 'load_run', None)
    if load_run is not None:
        return base_dir + '/run_{}'.format(load_run)
    if not os.path.exists(base_dir):
        return base_dir
    run_dirs = []
    for name in os.listdir(base_dir):
        path = base_dir + '/' + name
        if os.path.isdir(path) and name.startswith('run_'):
            try:
                run_dirs.append((int(name.split('_')[-1]), path))
            except ValueError:
                pass
    if len(run_dirs) == 0:
        return base_dir
    return sorted(run_dirs, key=lambda x: x[0])[-1][1]


def get_checkpoint_path(model_dir, suffix, args):
    """返回 checkpoint 路径；未指定 load_checkpoint 时自动选择最新编号。"""
    load_checkpoint = getattr(args, 'load_checkpoint', None)
    if load_checkpoint is not None:
        return model_dir + '/' + str(load_checkpoint) + '_' + suffix
    numbered_paths = []
    if os.path.exists(model_dir):
        for name in os.listdir(model_dir):
            if name.endswith('_' + suffix):
                try:
                    numbered_paths.append((int(name.split('_')[0]), model_dir + '/' + name))
                except ValueError:
                    pass
    if len(numbered_paths) > 0:
        return sorted(numbered_paths, key=lambda x: x[0])[-1][1]
    return model_dir + '/' + suffix


def format_elapsed_time(seconds):
    """将秒数格式化为 HH:MM:SS。"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        """合并默认值和调用参数，将其写入实例后再执行原方法。"""
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def td_lambda_target(batch, max_episode_len, q_targets, args):
    """由奖励、终止标记和目标 Q 值计算每个智能体的 TD(lambda) 回报。"""
    # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
    # q_targets.shape = (episode_num, max_episode_len， n_agents)
    episode_num = batch['o'].shape[0]
    mask = (1 - batch["padded"].float()).repeat(1, 1, args.n_agents)
    terminated = (1 - batch["terminated"].float()).repeat(1, 1, args.n_agents)
    r = batch['r'].repeat((1, 1, args.n_agents))
    # --------------------------------------------------n_step_return---------------------------------------------------
    '''
    1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
    最后一维,第n个数代表 n+1 step。
    2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
    否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
    如果没有置0，在计算td-error后再置0是来不及的
    3. terminated用来将超出当前episode长度的q_targets和r置为0
    '''
    # 最后一维枚举不同跨度的 n-step return，之后再按 lambda 加权融合。
    n_step_return = torch.zeros((episode_num, max_episode_len, args.n_agents, max_episode_len))
    for transition_idx in range(max_episode_len - 1, -1, -1):
        # 最后计算1 step return
        n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
        # 同时要注意n step return对应的index为n-1
        for n in range(1, max_episode_len - transition_idx):
            # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
            # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
            n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
    # --------------------------------------------------n_step_return---------------------------------------------------

    # --------------------------------------------------lambda return---------------------------------------------------
    '''
    lambda_return.shape = (episode_num, max_episode_len，n_agents)
    '''
    lambda_return = torch.zeros((episode_num, max_episode_len, args.n_agents))
    for transition_idx in range(max_episode_len):
        returns = torch.zeros((episode_num, args.n_agents))
        for n in range(1, max_episode_len - transition_idx):
            returns += pow(args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
        lambda_return[:, transition_idx] = (1 - args.td_lambda) * returns + \
                                           pow(args.td_lambda, max_episode_len - transition_idx - 1) * \
                                           n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
    # --------------------------------------------------lambda return---------------------------------------------------
    return lambda_return
