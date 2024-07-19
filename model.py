import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple
from collections import namedtuple
import math

# Define a named tuple for storing transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

class RainbowDQN(nn.Module):
    def __init__(self, h: int, w: int, outputs: int, atom_size: int, support: torch.Tensor):
        super(RainbowDQN, self).__init__()
        self.support = support
        self.outputs = outputs
        self.atom_size = atom_size

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.advantage_hidden = NoisyLinear(linear_input_size, 512)
        self.advantage_output = NoisyLinear(512, outputs * atom_size)

        self.value_hidden = NoisyLinear(linear_input_size, 512)
        self.value_output = NoisyLinear(512, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        advantage = F.relu(self.advantage_hidden(x))
        advantage = self.advantage_output(advantage).view(-1, self.outputs, self.atom_size)

        value = F.relu(self.value_hidden(x))
        value = self.value_output(value).view(-1, 1, self.atom_size)

        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_dist, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)  # 防止取log时出现nan

        return q_dist

    def reset_noise(self):
        self.advantage_hidden.reset_noise()
        self.advantage_output.reset_noise()
        self.value_hidden.reset_noise()
        self.value_output.reset_noise()

class SumTree:
    """
    A sum tree data structure used for efficient sampling in prioritized experience replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayMemory:
    """
    A prioritized replay buffer that uses a sum tree for efficient sampling.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, *args):
        max_priority = np.max(self.tree.tree[-self.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, Transition(*args))

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        self.beta = min(1.0, self.beta + self.frame * (1.0 - self.beta) / self.beta_frames)
        self.frame += 1

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, idxs, priorities):
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha)

    def __len__(self):
        return self.tree.n_entries

class ActionSelector:
    """
    Epsilon-greedy action selector for DQN.
    """
    def __init__(self, initial_epsilon: float, final_epsilon: float, policy_net: nn.Module,
                 eps_decay: int, n_actions: int, device: torch.device, support: torch.Tensor):
        self._eps = initial_epsilon
        self._final_epsilon = final_epsilon
        self._initial_epsilon = initial_epsilon
        self._policy_net = policy_net
        self._eps_decay = eps_decay
        self._n_actions = n_actions
        self._device = device
        self._support = support

    def select_action(self, state: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, float]:
        sample = random.random()
        if train:
            self._eps -= (self._initial_epsilon - self._final_epsilon) / self._eps_decay
            self._eps = max(self._eps, self._final_epsilon)
        if sample > self._eps:
            with torch.no_grad():
                dist = self._policy_net(state)
                expected_value = dist * self._support
                action = expected_value.sum(2).max(1)[1].view(1, 1)
                return action, self._eps
        else:
            return torch.tensor([[random.randrange(self._n_actions)]], device=self._device, dtype=torch.long), self._eps

def fp(n_frame):
    """
    Frame Processor: Converts input frames to the correct format for the DQN.
    """
    if isinstance(n_frame, np.ndarray):
        return torch.from_numpy(n_frame).float().permute(2, 0, 1).unsqueeze(0)
    elif isinstance(n_frame, torch.Tensor):
        if len(n_frame.shape) == 3:
            return n_frame.float().permute(2, 0, 1).unsqueeze(0)
        elif len(n_frame.shape) == 4:
            return n_frame.float().permute(0, 3, 1, 2)
    elif hasattr(n_frame, '__array__'):
        return fp(np.array(n_frame))
    else:
        raise ValueError(f"Unsupported frame type: {type(n_frame)}. Expected numpy array or torch tensor.")