import math
import random
import numpy as np
import os
from collections import deque
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import model as m
from atari_wrappers import wrap_deepmind, make_atari
import yaml

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 从配置文件中读取参数
device = torch.device(config['device'])
env_name = config['env_name']
BATCH_SIZE = int(config['batch_size'])
GAMMA = float(config['gamma'])
EPS_START = float(config['eps_start'])
EPS_END = float(config['eps_end'])
EPS_DECAY = int(config['eps_decay'])
TARGET_UPDATE = int(config['target_update'])
NUM_STEPS = int(config['num_steps'])
M_SIZE = int(config['memory_size'])
POLICY_UPDATE = int(config['policy_update'])
EVALUATE_FREQ = int(config['evaluate_freq'])
LEARNING_RATE = float(config['lr'])
ADAM_EPS = float(config['adam_eps'])
EVALUATE_EPS = float(config['evaluate_eps'])
EVALUATE_NUM_EPISODES = int(config['evaluate_num_episodes'])

# 创建环境
env_raw = make_atari(f'{env_name}NoFrameskip-v4')
env = wrap_deepmind(env_raw, frame_stack=True, episode_life=True, clip_rewards=True)

# 获取观察空间的形状
obs, _ = env.reset()
obs = m.fp(np.array(obs))
c, h, w = obs.shape[1], obs.shape[2], obs.shape[3]
n_actions = env.action_space.n
print(f"Observation shape: {(c, h, w)}")
print(f"Number of actions: {n_actions}")

# 初始化网络
policy_net = m.DQN(h, w, n_actions).to(device)
target_net = m.DQN(h, w, n_actions).to(device)
policy_net.apply(policy_net.init_weights)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 初始化优化器
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

# 初始化经验回放内存
memory = m.ReplayMemory(M_SIZE)

# 初始化动作选择器
sa = m.ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

# 初始化 TensorBoard
writer = SummaryWriter(log_dir=config['log_dir'])

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = m.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()

def evaluate(step, num_episodes=EVALUATE_NUM_EPISODES):
    eval_env = wrap_deepmind(make_atari(f'{env_name}NoFrameskip-v4'), frame_stack=True, episode_life=False, clip_rewards=False)
    eval_sa = m.ActionSelector(EVALUATE_EPS, EVALUATE_EPS, policy_net, EPS_DECAY, n_actions, device)
    
    total_reward = 0
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        obs = m.fp(np.array(obs))
        done = False
        episode_reward = 0
        while not done:
            state = obs.to(device)
            action, _ = eval_sa.select_action(state, train=False)
            next_obs, reward, terminated, truncated, _ = eval_env.step(action.item())
            done = terminated or truncated
            obs = m.fp(np.array(next_obs))
            episode_reward += reward
        total_reward += episode_reward
    
    avg_reward = total_reward / num_episodes
    writer.add_scalar('Reward/eval', avg_reward, global_step=step)
    print(f"Step {step}: Evaluation average reward: {avg_reward}")
    return avg_reward

# 主训练循环
obs, _ = env.reset()
obs = m.fp(np.array(obs))
episode_reward = 0

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:
    state = obs.to(device)
    action, eps = sa.select_action(state, train=True)
    next_obs, reward, terminated, truncated, _ = env.step(action.item())
    done = terminated or truncated
    reward = torch.tensor([reward], device=device)

    if not done:
        next_state = m.fp(np.array(next_obs))
    else:
        next_state = None

    memory.push(state, action, next_state, reward, done)

    obs = next_state if next_state is not None else m.fp(np.array(env.reset()[0]))

    episode_reward += reward.item()

    if step % POLICY_UPDATE == 0:
        loss = optimize_model()
        if loss is not None:
            writer.add_scalar('Loss/train', loss, global_step=step)

    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if done:
        writer.add_scalar('Reward/train', episode_reward, global_step=step)
        writer.add_scalar('Epsilon', eps, global_step=step)
        episode_reward = 0

    if step % EVALUATE_FREQ == 0:
        evaluate(step)

    if (step + 1) % 1000000 == 0:
        torch.save(policy_net.state_dict(), os.path.join(config['save_dir'], f'{env_name}_model_step_{step+1}.pth'))

writer.close()
print("Training completed.")