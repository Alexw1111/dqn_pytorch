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
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 从配置文件中读取参数
device = config['device']
env_name = config['env_name']
BATCH_SIZE = config['batch_size']
GAMMA = config['gamma']
EPS_START = config['eps_start']
EPS_END = config['eps_end']
EPS_DECAY = config['eps_decay']
TARGET_UPDATE = config['target_update']
NUM_STEPS = config['num_steps']
M_SIZE = config['memory_size']
POLICY_UPDATE = config['policy_update']
EVALUATE_FREQ = config['evaluate_freq']
LEARNING_RATE = config['lr']
ADAM_EPS = config['adam_eps']
EVALUATE_EPS = config['evaluate_eps']
EVALUATE_NUM_EPISODES = config['evaluate_num_episodes']

env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
env = wrap_deepmind(env_raw, frame_stack=False, episode_life=True, clip_rewards=True)

c, h, w = m.fp(env.reset()).shape
n_actions = env.action_space.n
print(n_actions)

policy_net = m.DQN(h, w, n_actions, device).to(device)
target_net = m.DQN(h, w, n_actions, device).to(device)
policy_net.apply(policy_net.init_weights)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# 创建adam
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

memory = m.ReplayMemory(M_SIZE, [5, h, w], n_actions, device)
sa = m.ActionSelector(EPS_START, EPS_END, policy_net, EPS_DECAY, n_actions, device)

steps_done = 0
writer = SummaryWriter(log_dir=config['log_dir'])

def optimize_model(train):
    if not train:
        return
    state_batch, action_batch, reward_batch, n_state_batch, done_batch = memory.sample(BATCH_SIZE)

    q = policy_net(state_batch).gather(1, action_batch)
    nq = target_net(n_state_batch).max(1)[0].detach()

    expected_state_action_values = (nq * GAMMA) * (1. - done_batch[:, 0]) + reward_batch[:, 0]

    loss = F.smooth_l1_loss(q, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    writer.add_scalar('Loss/train', loss.item(), global_step=steps_done)

def evaluate(step, policy_net, device, env, n_actions, eps=0.05, num_episode=5):
    env = wrap_deepmind(env)
    sa = m.ActionSelector(eps, eps, policy_net, EPS_DECAY, n_actions, device)
    e_rewards = []
    q = deque(maxlen=5)
    for i in range(num_episode):
        env.reset()
        e_reward = 0
        for _ in range(10):
            n_frame, _, done, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)

        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, eps = sa.select_action(state, train=False)
            n_frame, reward, done, info = env.step(action)
            n_frame = m.fp(n_frame)
            q.append(n_frame)

            e_reward += reward
        e_rewards.append(e_reward)

    avg_reward = float(sum(e_rewards)) / float(num_episode)
    writer.add_scalar('Reward/eval', avg_reward, global_step=step)
    writer.add_scalar('Eps/eval', eps, global_step=step)

q = deque(maxlen=5)
done = True
eps = 0
episode_len = 0

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=50, leave=False, unit='b')
for step in progressive:
    if done:
        env.reset()
        sum_reward = 0
        episode_len = 0
        img, _, _, _ = env.step(1)
        for i in range(10):
            n_frame, _, _, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)

    train = len(memory) > 50000
    state = torch.cat(list(q))[1:].unsqueeze(0)
    action, eps = sa.select_action(state, train)
    n_frame, reward, done, info = env.step(action)
    n_frame = m.fp(n_frame)

    q.append(n_frame)
    memory.push(torch.cat(list(q)).unsqueeze(0), action, reward, done)
    episode_len += 1

    if step % POLICY_UPDATE == 0:
        optimize_model(train)

    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if step % EVALUATE_FREQ == 0:
        evaluate(step, policy_net, device, env_raw, n_actions, eps=EVALUATE_EPS, num_episode=EVALUATE_NUM_EPISODES)

writer.close()