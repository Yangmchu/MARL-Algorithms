"""Training scheduler for rollout, replay, parameter updates and evaluation."""

import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from agent.agent import Agents, CommAgents
from common.parallel_rollout import get_policy_state, rollout_worker
from common.replay_buffer import ReplayBuffer
from common.rollout import CommRolloutWorker, RolloutWorker
from common.utils import format_elapsed_time, get_run_name, get_run_tag, get_save_dir


class Runner:
    """Manage the full training lifecycle for one algorithm on one SMAC map."""

    def __init__(self, env, args):
        self.env = env

        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.losses = []
        self.curve_time_steps = []
        self.best_win_rate = -1
        self.parallel_workers = []
        self.parallel_task_pipes = []
        self.parallel_result_pipes = []

        self.run_tag = get_run_tag(args)
        self.run_name = get_run_name(args)
        self.save_path = get_save_dir(self.args.result_dir, args)
        self.args.training_start_time = time.time()
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.initial_time_steps = 0
        self.initial_train_steps = 0
        if getattr(self.args, 'append_results', False):
            self._load_saved_results()
        if self.args.use_parallel and not self.args.evaluate:
            self._start_parallel_workers()

    def _get_parallel_process_num(self):
        return max(1, min(int(self.args.num_cpu), mp.cpu_count()))

    def _start_parallel_workers(self):
        """Start persistent rollout workers with independent SMAC envs."""
        process_num = self._get_parallel_process_num()
        for worker_id in range(process_num):
            task_parent, task_child = mp.Pipe()
            result_parent, result_child = mp.Pipe()
            worker = mp.Process(
                target=rollout_worker,
                args=(worker_id, self.args, task_child, result_child)
            )
            worker.start()
            self.parallel_workers.append(worker)
            self.parallel_task_pipes.append(task_parent)
            self.parallel_result_pipes.append(result_parent)

    def _close_parallel_workers(self):
        """Ask rollout workers to exit and wait for them."""
        for task_pipe in self.parallel_task_pipes:
            try:
                task_pipe.send('TERMINATE')
            except (BrokenPipeError, EOFError):
                pass
        for worker in self.parallel_workers:
            worker.join()
        self.parallel_workers = []
        self.parallel_task_pipes = []
        self.parallel_result_pipes = []

    def _merge_episodes(self, episodes):
        """Merge single-episode batches along the episode dimension."""
        episode_batch = episodes[0]
        for episode in episodes[1:]:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
        return episode_batch

    def _collect_serial_episodes(self):
        """Collect episodes one by one with the main-process environment."""
        episodes = []
        total_steps = 0
        for episode_idx in range(self.args.n_episodes):
            episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
            episodes.append(episode)
            total_steps += steps
        return self._merge_episodes(episodes), total_steps

    def _anneal_parallel_epsilon(self, total_steps, episode_count):
        """Update epsilon in the main process after a parallel rollout batch."""
        if self.args.epsilon_anneal_scale == 'episode':
            anneal_count = episode_count
        elif self.args.epsilon_anneal_scale == 'step':
            anneal_count = total_steps
        else:
            anneal_count = 0
        if anneal_count <= 0:
            return
        epsilon = self.rolloutWorker.epsilon - self.rolloutWorker.anneal_epsilon * anneal_count
        self.rolloutWorker.epsilon = max(self.rolloutWorker.min_epsilon, epsilon)

    def _collect_parallel_episodes(self):
        """Collect one batch of num_cpu episodes with independent worker envs."""
        policy_state = get_policy_state(self.agents.policy)
        epsilon = self.rolloutWorker.epsilon
        for episode_idx, task_pipe in enumerate(self.parallel_task_pipes):
            task_pipe.send((policy_state, epsilon, episode_idx))
        results = [
            result_pipe.recv()
            for result_pipe in self.parallel_result_pipes
        ]
        episodes = [result[0] for result in results]
        total_steps = sum(result[3] for result in results)
        self._anneal_parallel_epsilon(total_steps, len(episodes))
        return self._merge_episodes(episodes), total_steps

    def _record_loss(self, loss):
        """Record one learner update loss."""
        if loss is None:
            return
        self.losses.append(float(loss))

    def _load_curve(self, name, num):
        path = self.save_path + '/{}_{}.npy'.format(name, num)
        if not os.path.exists(path):
            return []
        return np.load(path, allow_pickle=True).tolist()

    def _load_saved_results(self):
        """Load existing curves before a resumed run so new points append."""
        run_num = getattr(self.args, 'save_run', None)
        if run_num is None:
            run_num = getattr(self.args, 'run_num', 0)

        self.win_rates = self._load_curve('win_rates', run_num)
        self.episode_rewards = self._load_curve('episode_rewards', run_num)
        self.losses = self._load_curve('losses', run_num)
        self.curve_time_steps = self._load_curve('time_steps', run_num)
        if len(self.win_rates) > 0:
            self.best_win_rate = max(self.win_rates)

        saved_time_steps = self._load_curve('time_steps', run_num)
        if len(saved_time_steps) > 0:
            self.initial_time_steps = int(saved_time_steps[-1])
        elif len(self.win_rates) > 0:
            self.initial_time_steps = max(0, (len(self.win_rates) - 1) * self.args.evaluate_cycle)
            self.curve_time_steps = [
                step * self.args.evaluate_cycle
                for step in range(len(self.win_rates))
            ]

        self.initial_train_steps = len(self.losses)
        print(
            'Append results from {}, loaded {} win rates, {} rewards, {} losses, resume time_steps {}'.format(
                self.save_path,
                len(self.win_rates),
                len(self.episode_rewards),
                len(self.losses),
                self.initial_time_steps
            )
        )

    def run(self, num):
        """Train until n_steps, evaluating and saving curves periodically."""
        try:
            time_steps = self.initial_time_steps
            train_steps = self.initial_train_steps
            evaluate_steps = time_steps // self.args.evaluate_cycle
            while time_steps < self.args.n_steps:
                print('Run {}, time_steps {}, train_steps {}'.format(num, time_steps, train_steps))
                if time_steps // self.args.evaluate_cycle > evaluate_steps:
                    win_rate, episode_reward = self.evaluate()
                    self.win_rates.append(win_rate)
                    self.episode_rewards.append(episode_reward)
                    self.curve_time_steps.append(time_steps)
                    self._save_qplex_bst_model(win_rate, train_steps)
                    self.plt(num)
                    evaluate_steps += 1

                if self.args.use_parallel:
                    episode_batch, collected_steps = self._collect_parallel_episodes()
                else:
                    episode_batch, collected_steps = self._collect_serial_episodes()
                time_steps += collected_steps

                # Rollout workers only collect trajectories; all learning stays here.
                if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                    update_count = self.args.train_steps if self.args.use_parallel else 1
                    for train_step in range(update_count):
                        loss = self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                        self._record_loss(loss)
                        train_steps += 1
                else:
                    self.buffer.store_episode(episode_batch)
                    for train_step in range(self.args.train_steps):
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                        loss = self.agents.train(mini_batch, train_steps)
                        self._record_loss(loss)
                        train_steps += 1

            win_rate, episode_reward = self.evaluate()
            print('win_rate is ', win_rate)
            self.win_rates.append(win_rate)
            self.episode_rewards.append(episode_reward)
            self.curve_time_steps.append(time_steps)
            self._save_qplex_bst_model(win_rate, train_steps)
            self.plt(num)
            elapsed_time = time.time() - self.args.training_start_time
            print('Save final model at train_step {}, elapsed time {}'.format(train_steps, format_elapsed_time(elapsed_time)))
            self.agents.policy.save_model('final')
        finally:
            self._close_parallel_workers()

    def _save_qplex_bst_model(self, win_rate, train_steps):
        if self.args.alg != 'qplex':
            return
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            elapsed_time = time.time() - getattr(self.args, 'training_start_time', time.time())
            print('Save qplex bst model at train_step {}, win_rate {}, elapsed time {}'.format(
                train_steps, win_rate, format_elapsed_time(elapsed_time)
            ))
            self.agents.policy.save_model('bst')

    def evaluate(self):
        """Run evaluation episodes and return mean win rate and reward."""
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        """Save current win-rate and reward curves plus NumPy data."""
        plt.figure()
        plt.suptitle(self.run_tag + '/' + self.run_name)
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(3, 1, 1)
        win_x = self.curve_time_steps if len(self.curve_time_steps) == len(self.win_rates) else range(len(self.win_rates))
        plt.plot(win_x, self.win_rates)
        plt.xlabel('time_steps')
        plt.ylabel('win_rates')

        plt.subplot(3, 1, 2)
        reward_x = self.curve_time_steps if len(self.curve_time_steps) == len(self.episode_rewards) else range(len(self.episode_rewards))
        plt.plot(reward_x, self.episode_rewards)
        plt.xlabel('time_steps')
        plt.ylabel('episode_rewards')

        plt.subplot(3, 1, 3)
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('loss')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        np.save(self.save_path + '/losses_{}'.format(num), self.losses)
        np.save(self.save_path + '/time_steps_{}'.format(num), self.curve_time_steps)
        plt.close()
