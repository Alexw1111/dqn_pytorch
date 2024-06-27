import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from typing import List, Tuple
import cv2

# 定义经验回放的数据结构
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, h: int, w: int, outputs: int):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        self.fc = nn.Linear(linear_input_size, 512)
        self.out = nn.Linear(512, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.out(x)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class ActionSelector:
    def __init__(self, initial_epsilon: float, final_epsilon: float, policy_net: nn.Module,
                 eps_decay: int, n_actions: int, device: torch.device):
        self._eps = initial_epsilon
        self._final_epsilon = final_epsilon
        self._initial_epsilon = initial_epsilon
        self._policy_net = policy_net
        self._eps_decay = eps_decay
        self._n_actions = n_actions
        self._device = device

    def select_action(self, state: torch.Tensor, train: bool = False) -> Tuple[torch.Tensor, float]:
        sample = random.random()
        if train:
            self._eps -= (self._initial_epsilon - self._final_epsilon) / self._eps_decay
            self._eps = max(self._eps, self._final_epsilon)
        if sample > self._eps:
            with torch.no_grad():
                return self._policy_net(state).max(1)[1].view(1, 1), self._eps
        else:
            return torch.tensor([[random.randrange(self._n_actions)]], device=self._device, dtype=torch.long), self._eps

def fp(n_frame):
    """Frame Processor"""
    if isinstance(n_frame, np.ndarray):
        return torch.from_numpy(n_frame).float().permute(2, 0, 1).unsqueeze(0)
    elif isinstance(n_frame, torch.Tensor):
        if len(n_frame.shape) == 3:
            return n_frame.float().permute(2, 0, 1).unsqueeze(0)
        elif len(n_frame.shape) == 4:
            return n_frame.float().permute(0, 3, 1, 2)
    elif hasattr(n_frame, '__array__'):  # 处理类似 LazyFrames 的对象
        return fp(np.array(n_frame))
    else:
        raise ValueError(f"Unsupported frame type: {type(n_frame)}. Expected numpy array or torch tensor.")

class FrameProcessor:
    def __init__(self, im_size: int = 84):
        self.im_size = im_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.im_size, self.im_size), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]  # Add a channel dimension