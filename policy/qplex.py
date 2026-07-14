import os

import torch

from common.utils import get_checkpoint_path, get_load_dir, get_save_dir
from network.base_net import RNN
from network.qplex_net import QPLEXNet


class QPLEX:
    """Basic QPLEX with recurrent individual Q networks and Double Q targets."""

    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.device = self._resolve_device(args)

        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qplex_net = QPLEXNet(args)
        self.target_qplex_net = QPLEXNet(args)

        self.eval_rnn.to(self.device)
        self.target_rnn.to(self.device)
        self.eval_qplex_net.to(self.device)
        self.target_qplex_net.to(self.device)

        self.model_dir = get_load_dir(args.model_dir, args) if self.args.load_model else get_save_dir(args.model_dir, args)
        if self.args.load_model:
            path_rnn = get_checkpoint_path(self.model_dir, 'rnn_net_params.pkl', args)
            path_qplex = get_checkpoint_path(self.model_dir, 'qplex_net_params.pkl', args)
            if os.path.exists(path_rnn) and os.path.exists(path_qplex):
                map_location = self.device
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qplex_net.load_state_dict(torch.load(path_qplex, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qplex))
            else:
                raise Exception("No model!")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qplex_net.load_state_dict(self.eval_qplex_net.state_dict())

        self.eval_parameters = list(self.eval_qplex_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg QPLEX')

    def _resolve_device(self, args):
        requested_device = getattr(args, 'device', None)
        requested_device = requested_device if requested_device is not None else ('cuda:0' if args.cuda else 'cpu')
        device = torch.device(requested_device)
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise Exception('CUDA device was requested, but CUDA is not available.')
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise Exception(
                    'CUDA device {} was requested, but only {} CUDA device(s) are available.'.format(
                        requested_device, torch.cuda.device_count()
                    )
                )
        return device

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_next = batch['s'], batch['s_next']
        u, u_onehot = batch['u'], batch['u_onehot']
        r, terminated = batch['r'], batch['terminated']
        avail_u, avail_u_next = batch['avail_u'], batch['avail_u_next']
        mask = 1 - batch["padded"].float()

        q_evals, q_eval_next, q_targets = self.get_q_values(batch, max_episode_len)

        s = s.to(self.device)
        s_next = s_next.to(self.device)
        u = u.to(self.device)
        u_onehot = u_onehot.to(self.device)
        r = r.to(self.device)
        terminated = terminated.to(self.device)
        avail_u = avail_u.to(self.device)
        avail_u_next = avail_u_next.to(self.device)
        mask = mask.to(self.device)

        chosen_q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_evals_for_max = q_evals.clone()
        q_evals_for_max[avail_u == 0.0] = -9999999
        max_q_i = q_evals_for_max.max(dim=3)[0]

        q_total_eval = self.eval_qplex_net(chosen_q_evals, s, is_v=True)
        q_total_eval += self.eval_qplex_net(
            chosen_q_evals,
            s,
            actions=u_onehot,
            max_q_i=max_q_i,
            is_v=False
        )

        q_eval_next_for_action = q_eval_next.detach().clone()
        q_eval_next_for_action[avail_u_next == 0.0] = -9999999
        target_actions = q_eval_next_for_action.argmax(dim=3, keepdim=True)
        target_actions_onehot = torch.zeros_like(q_targets).scatter_(3, target_actions, 1)

        target_chosen_qs = torch.gather(q_targets, dim=3, index=target_actions).squeeze(3)
        q_targets_for_max = q_targets.clone()
        q_targets_for_max[avail_u_next == 0.0] = -9999999
        target_max_q_i = q_targets_for_max.max(dim=3)[0]

        q_total_target = self.target_qplex_net(target_chosen_qs, s_next, is_v=True)
        q_total_target += self.target_qplex_net(
            target_chosen_qs,
            s_next,
            actions=target_actions_onehot,
            max_q_i=target_max_q_i,
            is_v=False
        )

        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        td_error = q_total_eval - targets.detach()
        masked_td_error = mask * td_error
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qplex_net.load_state_dict(self.eval_qplex_net.state_dict())

    def _get_sequence_inputs(self, batch, transition_idx, max_episode_len):
        if transition_idx < max_episode_len:
            obs = batch['o'][:, transition_idx]
        else:
            obs = batch['o_next'][:, transition_idx - 1]
        u_onehot = batch['u_onehot'][:]
        episode_num = obs.shape[0]

        inputs = [obs]
        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            agent_ids = torch.eye(self.args.n_agents, device=obs.device).unsqueeze(0).expand(episode_num, -1, -1)
            inputs.append(agent_ids)

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len + 1):
            inputs = self._get_sequence_inputs(batch, transition_idx, max_episode_len)
            inputs = inputs.to(self.device)

            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_evals.append(q_eval.view(episode_num, self.n_agents, -1))

            with torch.no_grad():
                q_target, self.target_hidden = self.target_rnn(inputs, self.target_hidden)
                q_targets.append(q_target.view(episode_num, self.n_agents, -1))

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals[:, :-1], q_evals[:, 1:], q_targets[:, 1:]

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim), device=self.device)
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim), device=self.device)

    def save_model(self, train_step):
        num = 'final' if train_step == 'final' else str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qplex_net.state_dict(), self.model_dir + '/' + num + '_qplex_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
