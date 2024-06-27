import torch
import numpy as np
from collections import deque
from atari_wrappers import wrap_deepmind, make_atari
import model as m
import yaml
import gym
from gym import wrappers

# 加载配置文件
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 从配置文件中读取参数
device = config['device']
env_name = config['env_name']
EVALUATE_EPS = config['evaluate_eps']

def load_model(model_path):
    env_raw = make_atari('{}NoFrameskip-v4'.format(env_name))
    env = wrap_deepmind(env_raw, frame_stack=False, episode_life=False, clip_rewards=False)
    
    c, h, w = m.fp(env.reset()).shape
    n_actions = env.action_space.n
    
    policy_net = m.DQN(h, w, n_actions, device).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()
    
    return policy_net, env, n_actions

def run_inference(policy_net, env, n_actions, num_episodes=10, record_video=True):
    sa = m.ActionSelector(EVALUATE_EPS, EVALUATE_EPS, policy_net, 0, n_actions, device)
    
    if record_video:
        env = gym.wrappers.Monitor(env, "./videos", video_callable=lambda episode_id: True, force=True)
    
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        done = False
        q = deque(maxlen=5)
        
        for _ in range(10):
            n_frame, _, done, _ = env.step(0)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
        
        while not done:
            state = torch.cat(list(q))[1:].unsqueeze(0)
            action, _ = sa.select_action(state, train=False)
            n_frame, reward, done, info = env.step(action)
            n_frame = m.fp(n_frame)
            q.append(n_frame)
            
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

if __name__ == "__main__":
    model_path = "path_to_your_model.pth"  # 替换为您的模型路径
    policy_net, env, n_actions = load_model(model_path)
    run_inference(policy_net, env, n_actions, record_video=True)