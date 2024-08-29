import os
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model as m
from atari_wrappers import make_atari, wrap_deepmind

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_environment(env_name: str) -> Any:
    env_raw = make_atari(f'{env_name}NoFrameskip-v4')
    return wrap_deepmind(env_raw, frame_stack=True, episode_life=True, clip_rewards=True)

def initialize_networks(observation_shape: tuple, n_actions: int, device: torch.device) -> tuple:
    h, w = observation_shape[2], observation_shape[3]
    policy_net = m.DQN(h, w, n_actions).to(device)
    target_net = m.DQN(h, w, n_actions).to(device)
    policy_net.apply(policy_net.init_weights)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net

def optimize_model(policy_net: m.DQN, target_net: m.DQN, optimizer: optim.Optimizer, 
                   memory: m.ReplayMemory, batch_size: int, gamma: float, device: torch.device) -> float:
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)
    batch = m.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=100)
    optimizer.step()

    # Clear unnecessary tensors from GPU memory
    del state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states
    torch.cuda.empty_cache()

    return loss.item()

def evaluate(policy_net: m.DQN, env: Any, action_selector: m.ActionSelector, device: torch.device, 
             num_episodes: int) -> float:
    total_reward = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = m.fp(np.array(obs))
        done = False
        episode_reward = 0
        while not done:
            state = obs.to(device)
            action, _ = action_selector.select_action(state, train=False)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            obs = m.fp(np.array(next_obs))
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes

def save_checkpoint(step: int, policy_net: m.DQN, optimizer: optim.Optimizer, 
                    best_eval_reward: float, save_path: str):
    torch.save({
        'step': step,
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_eval_reward': best_eval_reward
    }, save_path)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(checkpoint_path: str, policy_net: m.DQN, optimizer: optim.Optimizer, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    best_eval_reward = checkpoint['best_eval_reward']
    print(f"Loaded checkpoint from step {start_step}")
    return start_step, best_eval_reward

def main():
    config = load_config('config.yaml')
    device = torch.device(config['device'])
    env = setup_environment(config['env_name'])

    obs, _ = env.reset()
    obs = m.fp(np.array(obs))
    observation_shape = obs.shape
    n_actions = env.action_space.n

    policy_net, target_net = initialize_networks(observation_shape, n_actions, device)
    optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'], eps=config['adam_eps'])
    memory = m.ReplayMemory(config['memory_size'])
    action_selector = m.ActionSelector(config['eps_start'], config['eps_end'], 
                                       policy_net, config['eps_decay'], n_actions, device)

    writer = SummaryWriter(log_dir=config['log_dir'])
    os.makedirs(config['save_dir'], exist_ok=True)

    start_step = 0
    best_eval_reward = float('-inf')

    if config['resume_training']:
        start_step, best_eval_reward = load_checkpoint(config['resume_model_path'], policy_net, optimizer, device)
        target_net.load_state_dict(policy_net.state_dict())

    obs = m.fp(np.array(env.reset()[0]))
    episode_reward = 0

    for step in tqdm(range(start_step, config['num_steps']), initial=start_step, total=config['num_steps'], ncols=50, leave=False, unit='b'):
        state = obs.to(device)
        action, eps = action_selector.select_action(state, train=True)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        reward = torch.tensor([reward], device=device)

        next_state = m.fp(np.array(next_obs)) if not done else None
        memory.push(state, action, next_state, reward, done)
        obs = next_state if next_state is not None else m.fp(np.array(env.reset()[0]))

        episode_reward += reward.item()

        if step % config['policy_update'] == 0:
            loss = optimize_model(policy_net, target_net, optimizer, memory, 
                                  config['batch_size'], config['gamma'], device)
            if loss is not None:
                writer.add_scalar('Loss/train', loss, global_step=step)

        if step % config['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            writer.add_scalar('Reward/train', episode_reward, global_step=step)
            writer.add_scalar('Epsilon', eps, global_step=step)
            episode_reward = 0

        if step % config['evaluate_freq'] == 0:
            eval_reward = evaluate(policy_net, env, action_selector, device, config['evaluate_num_episodes'])
            writer.add_scalar('Reward/eval', eval_reward, global_step=step)
            print(f"Step {step}: Evaluation average reward: {eval_reward}")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(policy_net.state_dict(), 
                           os.path.join(config['save_dir'], f'{config["env_name"]}_best_model.pth'))
                print(f"New best model saved with reward: {best_eval_reward}")

        if (step + 1) % config['save_frequency'] == 0:
            save_checkpoint(step + 1, policy_net, optimizer, best_eval_reward,
                            os.path.join(config['save_dir'], f'{config["env_name"]}_checkpoint_step_{step+1}.pth'))

        # Periodically clear CUDA cache
        if step % 10000 == 0:
            torch.cuda.empty_cache()

    final_model_path = os.path.join(config['save_dir'], f'{config["env_name"]}_final_model.pth')
    torch.save(policy_net.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")

    writer.close()

if __name__ == "__main__":
    main()