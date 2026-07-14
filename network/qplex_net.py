"""Basic QPLEX mixer network.

This module implements the non-attention DMAQ/QPLEX mixer. The mixer first
applies a state-conditioned transformation to each individual agent Q value,
then decomposes the joint value into value and advantage branches:

    Q_tot = V_tot + A_tot
"""

import torch
import torch.nn as nn


class QPLEXNet(nn.Module):
    """QPLEX mixer with an internal state-action advantage weight network."""

    def __init__(self, args):
        super(QPLEXNet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = args.state_shape
        self.action_dim = args.n_agents * args.n_actions
        self.num_kernel = args.num_kernel

        hypernet_embed = args.hypernet_embed
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents)
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents)
        )

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        for _ in range(self.num_kernel):
            self.key_extractors.append(self._build_adv_hypernet(self.state_dim, 1))
            self.agents_extractors.append(self._build_adv_hypernet(self.state_dim, self.n_agents))
            self.action_extractors.append(
                self._build_adv_hypernet(self.state_dim + self.action_dim, self.n_agents)
            )

    def _build_adv_hypernet(self, input_dim, output_dim):
        """Build one advantage-weight hypernetwork branch."""
        layers = getattr(self.args, "adv_hypernet_layers", 1)
        hidden_dim = self.args.adv_hypernet_embed
        if layers == 1:
            return nn.Linear(input_dim, output_dim)
        if layers == 2:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        if layers == 3:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        raise Exception("Error setting number of adv hypernet layers.")

    def calc_si_weight(self, states, actions):
        """Generate non-negative per-agent advantage weights from state-action."""
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        state_actions = torch.cat([states, actions], dim=-1)

        head_weights = []
        for key_net, agent_net, action_net in zip(
            self.key_extractors, self.agents_extractors, self.action_extractors
        ):
            key = torch.abs(key_net(states)).repeat(1, self.n_agents) + 1e-10
            agent = torch.sigmoid(agent_net(states))
            action = torch.sigmoid(action_net(state_actions))
            head_weights.append(key * agent * action)

        weights = torch.stack(head_weights, dim=1)
        weights = weights.view(-1, self.num_kernel, self.n_agents)
        return weights.sum(dim=1)

    def calc_v(self, agent_qs):
        """Value branch: sum transformed individual Q values."""
        agent_qs = agent_qs.view(-1, self.n_agents)
        return torch.sum(agent_qs, dim=-1)

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        """Advantage branch weighted by state-action dependent coefficients."""
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = agent_qs - max_q_i
        if self.args.is_stop_gradient:
            adv_q = adv_q.detach()
        adv_w = self.calc_si_weight(states, actions).view(-1, self.n_agents)
        if self.args.is_minus_one:
            return torch.sum(adv_q * (adv_w - 1.0), dim=-1)
        return torch.sum(adv_q * adv_w, dim=-1)

    def forward(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        """Mix individual Q values into a joint value.

        Args:
            agent_qs: Tensor shaped [episode, time, n_agents].
            states: Tensor shaped [episode, time, state_shape].
            actions: Joint action one-hot shaped [episode, time, n_agents, n_actions].
            max_q_i: Per-agent max Q values shaped [episode, time, n_agents].
            is_v: If True, return the value branch; otherwise return advantage.
        """
        episode_num = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        w_final = torch.abs(self.hyper_w_final(states)).view(-1, self.n_agents) + 1e-10
        v = self.V(states).view(-1, self.n_agents)

        if self.args.weighted_head:
            agent_qs = w_final * agent_qs + v

        if is_v:
            y = self.calc_v(agent_qs)
        else:
            if actions is None or max_q_i is None:
                raise Exception("QPLEX advantage branch requires actions and max_q_i.")
            max_q_i = max_q_i.view(-1, self.n_agents)
            if self.args.weighted_head:
                max_q_i = w_final * max_q_i + v
            y = self.calc_adv(agent_qs, states, actions, max_q_i)

        return y.view(episode_num, -1, 1)
