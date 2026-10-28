
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.utils import add_self_loops,softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch


def _unpack_dqn_batch(batch, action_dim):
    if any(len(transition) not in (6, 7, 8) for transition in batch):
        raise ValueError("DQN transition must have 6, 7, or 8 fields")
    fields = list(zip(*(transition[:6] for transition in batch)))
    if all(len(transition) == 6 for transition in batch):
        return (*fields, None, None)
    current_masks = np.ones((len(batch), action_dim), dtype=bool)
    next_masks = np.ones((len(batch), action_dim), dtype=bool)
    for row, transition in enumerate(batch):
        values = (
            (None, transition[6]) if len(transition) == 7 else
            (transition[6], transition[7]) if len(transition) == 8 else
            (None, None)
        )
        for masks, value in zip((current_masks, next_masks), values):
            if value is not None:
                mask = np.asarray(value, dtype=bool).reshape(-1)
                if mask.size != action_dim:
                    raise ValueError(
                        f"valid_action_mask must have {action_dim} entries, got {mask.size}"
                    )
                masks[row] = mask
    return (*fields, current_masks, next_masks)


def _mask_next_q(q_values, valid_action_masks):
    if valid_action_masks is None:
        return q_values
    mask = torch.as_tensor(valid_action_masks, dtype=torch.bool, device=q_values.device)
    return q_values.masked_fill(~mask, float('-inf'))

def _bootstrap_action_masks(valid_action_masks, mark, done, action_dim, device):
    masks = (
        torch.ones((len(done), action_dim), dtype=torch.bool, device=device)
        if valid_action_masks is None else
        torch.as_tensor(valid_action_masks, dtype=torch.bool, device=device).clone()
    )
    masks[mark.bool(), 4] = False
    empty = ~masks.any(dim=1)
    if (empty & ~done.bool()).any():
        raise ValueError("Nonterminal DQN state has no legal next action")
    return masks

def get_activation(act_type: str):
    if act_type == 'LeakyRelu':
        return nn.LeakyReLU()
    elif act_type == 'Relu':
        return nn.ReLU()
    elif act_type == 'PRelu':
        return nn.PReLU()
    else:
        return nn.Identity()


class QNetwork(nn.Module):
    __constants__ = ['dueling', 'scale']

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, activation: str = 'LeakyRelu',
                 hidden_layers: int = 2, dueling = False, scale = 1.):
        super(QNetwork, self).__init__()

        self.in_layer = nn.Linear(state_dim, hidden_dim)
        self.act = get_activation(activation)
        self.dueling = dueling
        if self.dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.out_layer = nn.Linear(hidden_dim, action_dim)

        self.scale = scale

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.mid_acts = nn.ModuleList([get_activation(activation) for _ in range(hidden_layers)])

    def forward(self, observation, valid_action_mask=None):
            x = self.in_layer(observation)
            x = self.act(x)

            for mid_layer, mid_act in zip(self.mid_layers, self.mid_acts):
                x = mid_layer(x)
                x = mid_act(x)

            if self.dueling:
                value = self.value_stream(x)
                advantages = self.advantage_stream(x)
                if valid_action_mask is None:
                    mean_advantages = advantages.mean(dim=1, keepdim=True)
                else:
                    mask = torch.as_tensor(
                        valid_action_mask, dtype=torch.bool, device=advantages.device
                    )
                    if mask.ndim == 1:
                        if mask.numel() != advantages.shape[-1]:
                            raise ValueError(
                                "Dueling action mask must match Q output shape"
                            )
                        mask = mask.unsqueeze(0).expand(advantages.shape[0], -1)
                    elif mask.shape != advantages.shape:
                        raise ValueError("Dueling action mask must match Q output shape")
                    weights = mask.to(advantages.dtype)
                    mean_advantages = (advantages * weights).sum(
                        dim=1, keepdim=True
                    ) / weights.sum(dim=1, keepdim=True).clamp_min(1)
                x = value + (advantages - mean_advantages)
            else:
                x = self.out_layer(x)

            if self.scale > 1:
                x *= self.scale
            return x

class StandardGATConv(MessagePassing):
    def __init__(self, in_channels, edge_attr_dim, out_channels):
        super(StandardGATConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * out_channels + edge_attr_dim, 1),
            nn.LeakyReLU(0.2)
        )
        self.mlp = nn.Linear(out_channels + edge_attr_dim, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp.weight)
        torch.nn.init.xavier_uniform_(self.lin.weight)
        for layer in self.attention_mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr,
            fill_value=torch.zeros(1, edge_attr.size(1), device=edge_attr.device),
            num_nodes=x.size(0)
        )
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, edge_attr):
        eij = torch.cat([x_i, x_j, edge_attr], dim=1)
        alpha = self.attention_mlp(eij)
        alpha = softmax(alpha, edge_index_i)
        value = self.mlp(torch.cat([x_j, edge_attr], dim=1))
        return value * alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

class MultiLayerGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels):
        super(MultiLayerGAT, self).__init__()
        self.conv1 = StandardGATConv(in_channels, edge_dim, hidden_channels)
        self.conv2 = StandardGATConv(hidden_channels, edge_dim, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x

class DDQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 0.001, repeat=1, shuffle_func=None,
                 learning_starts=None, train_freq=None, initial_updates=None,
                 target_update_interval=32, max_grad_norm=None, huber_delta=None):
        self.device = device
        self.online_net = QNetwork(
            state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling,
        ).to(device)
        self.target_net = QNetwork(
            state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling,
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.q_mask = q_mask
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.initial_updates = initial_updates
        self.new_experience_count = 0
        self.next_update_at = learning_starts
        self.initial_update_done = False
        self.target_update_interval = target_update_interval
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self.huber_delta = (
            None if huber_delta is None else float(huber_delta)
        )
        self._updates_since_target = 0
        self.target_update_events = 0
        self.optimizer_steps_total = 0

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        optimizer_steps = 0
        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done, current_masks, next_masks = _unpack_dqn_batch(
                batch, self.action_dim
            )
            # Neighbor permutations are applied when the transition is
            # collected.  Do not reshuffle only the current state here:
            # that would put state/action and next_state in different
            # coordinate systems.
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)
            curr_q = self.online_net(state, current_masks)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            with torch.no_grad():
                action_masks = _bootstrap_action_masks(
                    next_masks, mark, done, self.action_dim, self.device
                )
                online_next_q = _mask_next_q(
                    self.online_net(next_state, action_masks), action_masks
                )
                target_next_q = self.target_net(next_state, action_masks)
                next_action = online_next_q.argmax(dim=1)
                next_q = target_next_q.gather(1, next_action.unsqueeze(1)).squeeze(1)
                next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)

            expected_q = reward + (1 - done) * self.gamma * next_q

            # ``None`` keeps the original MSE/no-clipping behavior; the
            # DDQN config can enable robust updates independently.
            if self.huber_delta is None:
                loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            else:
                loss = torch.nn.functional.smooth_l1_loss(
                    curr_q,
                    expected_q.detach(),
                    beta=self.huber_delta,
                )
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.online_net.parameters(), self.max_grad_norm
                )
            self.optimizer.step()
            optimizer_steps += 1
            self.optimizer_steps_total += 1
            self._updates_since_target += 1
            if (self.target_update_interval
                    and self._updates_since_target >= self.target_update_interval):
                self.target_update()
                self._updates_since_target = 0

        return optimizer_steps

    def observe_experience(self, experience):
        """Train according to new-transition counts when count scheduling is configured."""
        if self.learning_starts is None or self.train_freq is None or self.initial_updates is None:
            self.update([experience])
            return None

        self.replay_buffer.append(experience)
        self.new_experience_count += 1
        update_count = 0
        if not self.initial_update_done and self.new_experience_count >= self.learning_starts:
            update_count = self.initial_updates
            self.initial_update_done = True
            self.next_update_at = self.learning_starts + self.train_freq
        elif self.initial_update_done and self.new_experience_count >= self.next_update_at:
            update_count = ((self.new_experience_count - self.next_update_at) // self.train_freq) + 1
            self.next_update_at += update_count * self.train_freq

        if not update_count:
            return None
        previous_repeat = self.repeat
        try:
            self.repeat = update_count
            optimizer_steps = self.update([])
        finally:
            self.repeat = previous_repeat
        return dict(samples=len(self.replay_buffer), optimizer_steps=optimizer_steps,
                    new_experiences=self.new_experience_count)

    def target_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_update_events += 1

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            state_dict = torch.load(
                file_path, map_location=self.device, weights_only=True
            )
            self.online_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences


def _gather_direction_embeddings(embeddings, graph_batch):
    # Replay graphs place the four action directions first in each graph.
    indices = graph_batch.ptr[:-1, None] + torch.arange(4, device=embeddings.device)
    return embeddings[indices].reshape(graph_batch.num_graphs, -1)


class GNN_DDQN_Agent:
    def __init__(self, in_channels, hidden_channels, hidden_dim, edge_dim, gnn_dim, other_dim, action_dim, buffer_length, batch_size,
                 gamma, device, q_mask, activation='LeakyRelu', hidden_layers=2,
                 dueling=False, learning_rate=0.001, repeat=1, shuffle_func=None, gnn_factory=None,
                 target_update_interval=32, max_grad_norm=None):
        self.device = device

        gnn_factory = gnn_factory or MultiLayerGAT
        self.online_gnn = gnn_factory(in_channels, hidden_channels, edge_dim, gnn_dim).to(device)
        self.target_gnn = gnn_factory(in_channels, hidden_channels, edge_dim, gnn_dim).to(device)
        self.target_gnn.load_state_dict(self.online_gnn.state_dict())

        encoder_dim = getattr(self.online_gnn, 'output_dim', gnn_dim)
        combined_state_dim = encoder_dim * 4 + other_dim
        self.online_net = QNetwork(combined_state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.target_net = QNetwork(combined_state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.q_mask = q_mask
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(
            list(self.online_gnn.parameters()) + list(self.online_net.parameters()),
            lr=self.learning_rate
        )

        self.shuffle_func = shuffle_func
        self.repeat = repeat
        self.target_update_interval = target_update_interval
        self.max_grad_norm = (
            None if max_grad_norm is None else float(max_grad_norm)
        )
        self._updates_since_target = 0
        self.target_update_events = 0
        self.optimizer_steps_total = 0

    def update(self, experiences):
        pack = getattr(self, 'replay_graph_packer', None)
        if pack is not None:
            experiences = pack(experiences)
        graph_batch = getattr(self, 'replay_graph_collate', Batch.from_data_list)
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        optimizer_steps = 0
        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done, current_masks, next_masks = _unpack_dqn_batch(
                batch, self.action_dim
            )
            gnn_data, other_state = zip(*state)

            gnn_data = graph_batch(gnn_data).to(self.device)
            next_gnn_data, next_other_state = zip(*next_state)
            next_gnn_data = graph_batch(next_gnn_data).to(self.device)
            next_other_state = torch.tensor(np.array(next_other_state), dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_gnn_state = self.online_gnn(gnn_data)
            combined_vector = _gather_direction_embeddings(curr_gnn_state, gnn_data)

            combined_vector, other_state, action, current_masks = self.shuffle(
                (combined_vector, other_state, action, current_masks)
            )
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            other_state = torch.tensor(np.array(other_state), dtype=torch.float).to(self.device)
            if not torch.is_tensor(combined_vector):
                combined_vector = torch.stack(combined_vector, dim=0)

            q_state = torch.cat([combined_vector, other_state], dim=-1)
            curr_q = self.online_net(q_state, current_masks)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()

            with torch.no_grad():
                online_next_gnn_state = self.online_gnn(next_gnn_data)
                target_next_gnn_state = self.target_gnn(next_gnn_data)
                online_combined_vector = _gather_direction_embeddings(online_next_gnn_state, next_gnn_data)
                target_combined_vector = _gather_direction_embeddings(target_next_gnn_state, next_gnn_data)
                online_q_state = torch.cat([online_combined_vector, next_other_state], dim=-1)
                target_q_state = torch.cat([target_combined_vector, next_other_state], dim=-1)
                action_masks = _bootstrap_action_masks(
                    next_masks, mark, done, self.action_dim, self.device
                )
                online_next_q = _mask_next_q(
                    self.online_net(online_q_state, action_masks), action_masks
                )
                target_next_q = self.target_net(target_q_state, action_masks)

                next_action = online_next_q.argmax(dim=1)
                next_q = target_next_q.gather(1, next_action.unsqueeze(1)).squeeze(1)
                next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.online_gnn.parameters()) + list(self.online_net.parameters()),
                    self.max_grad_norm,
                )
            self.optimizer.step()

            optimizer_steps += 1
            self.optimizer_steps_total += 1
            self._updates_since_target += 1
            if (self.target_update_interval
                    and self._updates_since_target >= self.target_update_interval):
                self.target_update()
                self._updates_since_target = 0

        return optimizer_steps

    def target_update(self):
        self.target_gnn.load_state_dict(self.online_gnn.state_dict())
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_update_events += 1

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.online_gnn.state_dict(), file_path + '/gnn.pth')
        torch.save(self.online_net.state_dict(), file_path + '/qnet.pth')

    def load_model(self,file_path):
        if file_path:
            self.online_gnn.load_state_dict(torch.load(
                file_path + '/gnn.pth', map_location=self.device, weights_only=True
            ))
            self.online_net.load_state_dict(torch.load(
                file_path + '/qnet.pth', map_location=self.device, weights_only=True
            ))

    def shuffle(self, experiences):
        if self.shuffle_func:
            gnn_states = []
            states = []
            actions = []
            masks = []
            for gnn_state, state, action, mask in zip(*experiences):
                gnn_state, state, action, mask = self.shuffle_func(
                    gnn_state, state, action, mask
                )
                gnn_states.append(gnn_state)
                states.append(state)
                actions.append(action)
                masks.append(mask)
            return gnn_states, states, actions, masks
        else:
            return experiences


class DQN_Agent:
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling=False, learning_rate: float = 0.001, repeat=1, shuffle_func=None):
        self.device = device
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.q_mask = q_mask
        self.action_dim = action_dim
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat
        self.optimizer_steps_total = 0

    def update(self, experiences):
        self.replay_buffer.extend(experiences)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        optimizer_steps = 0
        for _ in range(self.repeat):
            indices = np.arange(len(self.replay_buffer))
            chosen_indices = np.random.choice(indices, self.batch_size, replace=False)
            batch = [self.replay_buffer[i] for i in chosen_indices]

            state, mark, action, reward, next_state, done, current_masks, next_masks = _unpack_dqn_batch(
                batch, self.action_dim
            )
            # Neighbor permutations are applied when the transition is
            # collected; replay must keep the stored state/action mapping.
            state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
            mark = torch.tensor(mark, dtype=torch.long).to(self.device)
            done = torch.tensor(done, dtype=torch.long).to(self.device)

            curr_q = self.online_net(state, current_masks)
            curr_q = curr_q.gather(1, action.unsqueeze(1)).squeeze()
            action_masks = _bootstrap_action_masks(
                next_masks, mark, done, self.action_dim, self.device
            )
            next_q = _mask_next_q(
                self.online_net(next_state, action_masks), action_masks
            ).max(dim=1)[0]
            next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)

            expected_q = reward + (1 - done) * self.gamma * next_q

            loss = torch.nn.functional.mse_loss(curr_q, expected_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            optimizer_steps += 1
            self.optimizer_steps_total += 1

        return optimizer_steps

    def save_model(self, file_path):
        torch.save(self.online_net.state_dict(), file_path)

    def load_model(self, file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(
                file_path, map_location=self.device, weights_only=True
            ))

    def target_update(self):
        pass

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences


def ppo_gae(rewards, values, next_values, dones, trajectory_steps, gamma, gae_lambda):
    """GAE on interleaved packet trajectories; bootstrap at rollout cuts/gaps.

    A rollout boundary is a truncation, not a terminal state. Policy-version
    changes therefore do not terminate a packet trajectory; the selected
    rollout is already collected under one behavior policy.
    """
    deltas = rewards + gamma * (1 - dones) * next_values - values
    advantages = np.zeros_like(deltas)
    tails = {}
    for i in sorted(range(len(rewards)), key=lambda j: trajectory_steps[j], reverse=True):
        trajectory_id, step = trajectory_steps[i]
        advantage = deltas[i]
        next_step, next_advantage = tails.get(trajectory_id, (None, 0.0))
        if not dones[i] and next_step == step + 1:
            advantage += gamma * gae_lambda * next_advantage
        advantages[i] = advantage
        tails[trajectory_id] = (step, advantage)
    return advantages, advantages + values


class PPO_Agent:
    """PPO-Clip with per-packet GAE and current-policy rollout minibatches."""
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, buffer_length: int, batch_size: int,
                 gamma: float, device, q_mask: int, activation: str = 'LeakyRelu', hidden_layers: int = 2,
                 dueling = False, learning_rate: float = 0.001, repeat=1, shuffle_func=None,
                 rollout_size=4096, gae_lambda=0.95, clip_ratio=0.2, entropy_coef=0.01,
                 max_grad_norm=0.5, target_kl=None, use_gradient_clipping=True):
        self.device = device
        self.action_dim = action_dim
        # Keep logits in a normal range while PPO is exploring. A 100x scale
        # makes the initial categorical policy almost deterministic.
        self.online_net = QNetwork(state_dim, hidden_dim, action_dim, activation, hidden_layers, dueling).to(device)
        self.critic_net = QNetwork(state_dim, hidden_dim, 1, activation, hidden_layers, dueling).to(device)

        self.q_mask = q_mask
        self.replay_buffer = deque(maxlen=buffer_length)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.optimizer_actor = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic_net.parameters(), lr=learning_rate)
        self.shuffle_func = shuffle_func
        self.repeat = repeat
        if not 0 < batch_size <= rollout_size <= buffer_length:
            raise ValueError("PPO requires 0 < batch_size <= rollout_size <= buffer_length")
        self.rollout_size = rollout_size
        self.gae_lambda = gae_lambda
        self.eps_clip = clip_ratio
        self.max_grad_norm = max_grad_norm
        self.use_gradient_clipping = bool(use_gradient_clipping)
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.policy_version = 0
        self.online_net.ppo_policy_version = self.policy_version
        self.discarded_stale = 0

    def policy_distribution(self, state, valid_action_masks=None):
        logits = self.online_net(state)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        # Computed packets cannot select local computation again.
        mask[state[:, -1].bool(), 4] = True
        if valid_action_masks is not None:
            valid_action_masks = torch.as_tensor(
                valid_action_masks, dtype=torch.bool, device=logits.device
            )
            if valid_action_masks.shape != logits.shape:
                raise ValueError(
                    f"PPO action mask shape {tuple(valid_action_masks.shape)} "
                    f"does not match logits {tuple(logits.shape)}"
                )
            mask |= ~valid_action_masks
        if (~mask).sum(dim=1).eq(0).any():
            raise ValueError("PPO action mask removed every action")
        return torch.distributions.Categorical(logits=logits.masked_fill(mask, float('-inf')))

    def _prepare_rollout_tensors(self, rollout):
        state, _, metadata, reward, next_state, done = zip(*rollout)
        state = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.critic_net(state).squeeze(-1).cpu().numpy()
            next_values = self.critic_net(next_state).squeeze(-1).cpu().numpy()
            advantages, returns = ppo_gae(
                np.asarray(reward, dtype=np.float32), values, next_values,
                np.asarray(done, dtype=np.float32), [(a[3], a[4]) for a in metadata],
                self.gamma, self.gae_lambda,
            )
        return (
            state,
            next_state,
            torch.as_tensor(advantages, device=self.device),
            torch.as_tensor(returns, device=self.device),
            metadata,
        )

    def update(self, experiences):
        # Only current-policy transitions train Actor and Critic. Any late
        # transition from an older policy is discarded entirely.
        fresh = []
        for experience in experiences:
            metadata = experience[2]
            if len(metadata) not in (5, 6, 7):
                raise ValueError(
                    "PPO needs action, log-probability, policy version, packet ID, step "
                    "and optional forced-action flag/action mask"
                )
            policy_lag = self.policy_version - metadata[2]
            if policy_lag == 0:
                fresh.append(experience)
            else:
                self.discarded_stale += 1
        if len(self.replay_buffer) + len(fresh) > self.replay_buffer.maxlen:
            raise ValueError("PPO rollout exceeded buffer_length; increase its capacity")
        self.replay_buffer.extend(fresh)
        if not self.replay_buffer or len(self.replay_buffer) < self.rollout_size:
            return

        # A simulation step can finish more than rollout_size transitions.
        # Keep only the newest complete rollout so the update boundary remains
        # aligned with the latest environment step.
        rollout = list(self.replay_buffer)[-self.rollout_size:]
        state, next_state, advantages, returns, metadata = self._prepare_rollout_tensors(rollout)
        action = torch.as_tensor([a[0] for a in metadata], dtype=torch.long, device=self.device)
        old_log_prob = torch.as_tensor([a[1] for a in metadata], dtype=torch.float32, device=self.device)
        forced_action = torch.as_tensor(
            [bool(a[5]) if len(a) > 5 else False for a in metadata],
            dtype=torch.bool,
            device=self.device,
        )
        valid_action_masks = torch.as_tensor(
            [a[6] if len(a) > 6 else [True] * self.action_dim for a in metadata],
            dtype=torch.bool,
            device=self.device,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # A sample without a following transition in this rollout is cut by
        # the rollout boundary. Its done flag remains false, so next_values
        # provide the required bootstrap instead of treating it as terminal.
        trajectory_steps = {(item[2][3], item[2][4]) for item in rollout}
        bootstrap_cuts = sum(
            1
            for experience in rollout
            if not experience[5]
            and (experience[2][3], experience[2][4] + 1)
            not in trajectory_steps
        )

        # Freeze old log-probs, GAE and value targets for all epochs. Each epoch
        # visits every sample once, including the final partial minibatch.
        count = len(state)
        optimizer_steps = 0
        epochs_completed = 0
        early_stopped = False
        approx_kl = 0.0
        for _ in range(self.repeat):
            indices = np.random.permutation(count)
            for start in range(0, count, self.batch_size):
                batch = torch.as_tensor(indices[start:start + self.batch_size], device=self.device)
                dist = self.policy_distribution(
                    state[batch], valid_action_masks[batch]
                )
                log_ratio = dist.log_prob(action[batch]) - old_log_prob[batch]
                ratios = torch.exp(log_ratio)
                actor_mask = ~forced_action[batch]
                batch_approx_kl = None
                if actor_mask.any():
                    actor_ratios = ratios[actor_mask]
                    actor_advantages = advantages[batch][actor_mask]
                    actor_entropy = dist.entropy()[actor_mask]
                    surr1 = actor_ratios * actor_advantages
                    surr2 = actor_ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * actor_advantages
                    actor_loss = (
                        -torch.min(surr1, surr2).mean()
                        - self.entropy_coef * actor_entropy.mean()
                    )
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(
                            self.online_net.parameters(), self.max_grad_norm
                        )
                    self.optimizer_actor.step()

                    with torch.no_grad():
                        updated_dist = self.policy_distribution(
                            state[batch], valid_action_masks[batch]
                        )
                        updated_log_ratio = (
                            updated_dist.log_prob(action[batch]) - old_log_prob[batch]
                        )[actor_mask]
                        batch_approx_kl = (
                            (updated_log_ratio.exp() - 1) - updated_log_ratio
                        ).mean().item()
                        approx_kl = batch_approx_kl

                state_value = self.critic_net(state[batch]).squeeze(-1)
                critic_loss = nn.functional.mse_loss(state_value, returns[batch])
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_net.parameters(), self.max_grad_norm
                    )
                self.optimizer_critic.step()
                optimizer_steps += 1
                if (
                    batch_approx_kl is not None
                    and self.target_kl is not None
                    and batch_approx_kl > 1.5 * self.target_kl
                ):
                    early_stopped = True
                    break
            if early_stopped:
                epochs_completed += 1
                break
            epochs_completed += 1

        self.replay_buffer.clear()
        self.policy_version += 1
        self.online_net.ppo_policy_version = self.policy_version
        stats = dict(samples=count, epochs=epochs_completed, optimizer_steps=optimizer_steps,
                     discarded_stale=self.discarded_stale,
                     policy_version=self.policy_version - 1,
                     bootstrap_cuts=bootstrap_cuts,
                     approx_kl=approx_kl,
                     early_stopped=early_stopped,
                     forced_action_samples=int(forced_action.sum().item()),
                     actor_samples=int((~forced_action).sum().item()))
        self.discarded_stale = 0
        return stats

    def observe_experience(self, experience):
        """Add one completed transition and update exactly at the rollout boundary."""
        return self.update([experience])

    def save_model(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.online_net.state_dict(), file_path + '/actor.pth')
        torch.save(self.critic_net.state_dict(), file_path + '/critic.pth')

    def load_model(self,file_path):
        if file_path:
            self.online_net.load_state_dict(torch.load(
                file_path + '/actor.pth', map_location=self.device, weights_only=True
            ))
            self.critic_net.load_state_dict(torch.load(
                file_path + '/critic.pth', map_location=self.device, weights_only=True
            ))

    def shuffle(self, experiences):
        if self.shuffle_func:
            states = []
            actions = []
            for state, action in zip(*experiences):
                state, action = self.shuffle_func(state, action)
                states.append(state)
                actions.append(action)
            return states, actions
        else:
            return experiences

def shuffle_neighbor_state(neighbor_states, other_states):
    parts = np.array_split(neighbor_states, 4)
    indices = np.random.permutation(4)
    new_state = np.concatenate([parts[idx] for idx in indices])
    return np.concatenate([new_state, other_states]), indices

def shuffle_neighbors(neighbor_states, other_states, action):
    new_state, indices = shuffle_neighbor_state(neighbor_states, other_states)
    if action < 4:
        new_action = int(np.where(indices == action)[0])
    else:
        new_action = action
    return new_state, new_action

class ShuffleEx:
    def __init__(self, shuffle_mask):
        self.shuffle_mask = shuffle_mask

    def shuffle(self, state, action):
        return shuffle_neighbors(state[:self.shuffle_mask], state[self.shuffle_mask:], action)

    def shuffle_for_policy(self, state):
        return shuffle_neighbor_state(state[:self.shuffle_mask], state[self.shuffle_mask:])

def shuffle_neighbors_gnn(gnn_states, neighbor_states, other_states, action, action_mask=None):
    gnn_parts = torch.tensor_split(gnn_states, 4, dim=-1)
    parts = np.array_split(neighbor_states, 4)
    indices = np.random.permutation(4)
    new_state = np.concatenate([parts[idx] for idx in indices])
    new_gnn_state = torch.cat([gnn_parts[idx] for idx in indices], dim=-1)
    if action < 4:
        new_action = int(np.where(indices == action)[0])
    else:
        new_action = action
    result = new_gnn_state, np.concatenate([new_state, other_states]), new_action
    if action_mask is None:
        return result
    new_mask = np.asarray(action_mask, dtype=bool).copy()
    new_mask[:4] = new_mask[indices]
    return (*result, new_mask)

class ShuffleEx_gnn:
    def __init__(self, shuffle_mask):
        self.shuffle_mask = shuffle_mask

    def shuffle(self, gnn_state, state, action, action_mask=None):
        return shuffle_neighbors_gnn(
            gnn_state, state[:self.shuffle_mask], state[self.shuffle_mask:],
            action, action_mask,
        )

def cal_agent_dim(neighbors_dim: int, edges_dim: int, distance_dim: int, mission_dim: int, current_dim: int,
                  action_dim: int):
    return neighbors_dim + edges_dim + distance_dim + mission_dim + current_dim, action_dim, -(
            mission_dim + current_dim)
