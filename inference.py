import os
from typing import Tuple, Any

import numpy as np
import torch
import yaml
import gymnasium as gym

import model as m
from atari_wrappers import wrap_deepmind, make_atari

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model(model_path: str, env_name: str, device: str) -> Tuple[m.DQN, Any, int]:
    """Load the trained model and set up the environment."""
    env_raw = make_atari(f'{env_name}NoFrameskip-v4')
    env = wrap_deepmind(env_raw, frame_stack=True, episode_life=False, clip_rewards=False)
    
    obs, _ = env.reset()
    obs = m.fp(np.array(obs))
    c, h, w = obs.shape[1], obs.shape[2], obs.shape[3]
    n_actions = env.action_space.n
    
    policy_net = m.DQN(h, w, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    
    return policy_net, env, n_actions

def run_inference(policy_net: m.DQN, env: Any, n_actions: int, device: str, 
                  evaluate_eps: float, num_episodes: int = 10, record_video: bool = True):
    """Run the trained model on the environment and optionally record videos."""
    sa = m.ActionSelector(evaluate_eps, evaluate_eps, policy_net, 0, n_actions, device)
    
    if record_video:
        try:
            env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda episode_id: True)
        except gym.error.DependencyNotInstalled:
            print("Warning: Unable to record video. moviepy is not installed.")
            print("Continuing without video recording.")
            record_video = False
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = m.fp(np.array(obs))
        total_reward = 0
        done = False
        
        while not done:
            state = obs.to(device)
            # 确保state的维度正确
            if state.dim() == 3:
                state = state.unsqueeze(0)  # 添加batch维度
            elif state.dim() == 5:
                state = state.squeeze(1)  # 移除多余的维度
            
            action, _ = sa.select_action(state, train=False)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            obs = m.fp(np.array(next_obs))
            
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

def main():
    """Main function to run the inference."""
    config = load_config('config.yaml')
    device = torch.device(config['device'])
    env_name = config['env_name']
    evaluate_eps = config['evaluate_eps']
    
    model_path = config['inference_model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    policy_net, env, n_actions = load_model(model_path, env_name, device)
    run_inference(policy_net, env, n_actions, device, evaluate_eps, 
                  num_episodes=10, record_video=True)

if __name__ == "__main__":
    main()