import os
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

import model as m
from atari_wrappers import wrap_deepmind, make_atari

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_environment(env_name: str) -> Any:
    """Set up the Atari environment."""
    env_raw = make_atari(f'{env_name}NoFrameskip-v4')
    return wrap_deepmind(env_raw, frame_stack=True, episode_life=True, clip_rewards=True)

def initialize_networks(observation_shape: Tuple[int, ...], n_actions: int, device: torch.device) -> Tuple[m.DQN, m.DQN]:
    """Initialize the policy and target networks."""
    policy_net = m.DQN(observation_shape[2], observation_shape[3], n_actions).to(device)
    target_net = m.DQN(observation_shape[2], observation_shape[3], n_actions).to(device)
    policy_net.apply(policy_net.init_weights)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net

def optimize_model(policy_net: m.DQN, target_net: m.DQN, optimizer: optim.Optimizer, 
                   memory: m.PrioritizedReplayMemory, batch_size: int, gamma: float, 
                   device: torch.device, scaler: GradScaler) -> float:
    """Perform one step of optimization on the DQN."""
    if len(memory) < batch_size:
        return None

    transitions, idxs, is_weights = memory.sample(batch_size)
    batch = m.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    with autocast():
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].to(torch.float32)
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        loss = (F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none') * torch.tensor(is_weights, device=device)).mean()

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=100)
    scaler.step(optimizer)
    scaler.update()

    # Update priorities
    with torch.no_grad():
        td_errors = torch.abs(state_action_values.float() - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
    new_priorities = td_errors + 1e-6  # small constant to ensure non-zero priority
    memory.update_priorities(idxs, new_priorities)

    return loss.item()

def main():
    """Main training loop."""
    config = load_config('config.yaml')
    device = torch.device(config['device'])
    env = setup_environment(config['env_name'])

    obs, _ = env.reset()
    obs = m.fp(np.array(obs))
    observation_shape = obs.shape
    n_actions = env.action_space.n

    policy_net, target_net = initialize_networks(observation_shape, n_actions, device)
    optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'], eps=config['adam_eps'])
    memory = m.PrioritizedReplayMemory(config['memory_size'])
    action_selector = m.ActionSelector(config['eps_start'], config['eps_end'], 
                                       policy_net, config['eps_decay'], n_actions, device)

    writer = SummaryWriter(log_dir=config['log_dir'])
    os.makedirs(config['save_dir'], exist_ok=True)

    scaler = GradScaler()

    obs = m.fp(np.array(env.reset()[0])).to(device)
    episode_reward = 0

    for step in tqdm(range(config['num_steps']), total=config['num_steps'], ncols=50, leave=False, unit='b'):
        action, eps = action_selector.select_action(obs, train=True)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        reward = torch.tensor([reward], device=device)

        next_state = m.fp(np.array(next_obs)).to(device) if not done else None
        memory.push(obs, action, next_state, reward, done)

        obs = next_state if next_state is not None else m.fp(np.array(env.reset()[0])).to(device)

        episode_reward += reward.item()

        if step % config['policy_update'] == 0:
            loss = optimize_model(policy_net, target_net, optimizer, memory, 
                                  config['batch_size'], config['gamma'], device, scaler)
            if loss is not None:
                writer.add_scalar('Loss/train', loss, global_step=step)

        if step % config['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            writer.add_scalar('Reward/train', episode_reward, global_step=step)
            writer.add_scalar('Epsilon', eps, global_step=step)
            episode_reward = 0

        if (step + 1) % config['save_frequency'] == 0:
            torch.save({
                'step': step + 1,
                'policy_net_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config['save_dir'], f'{config["env_name"]}_checkpoint_step_{step+1}.pth'))

    writer.close()
    print("Training completed.")

if __name__ == "__main__":
    main()