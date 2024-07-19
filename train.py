import os
import random
from typing import Dict, Any, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

import gc
import psutil

import model as m
from atari_wrappers import wrap_deepmind, make_atari

def setup_logging(log_dir: str) -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_environment(env_name: str) -> Any:
    """Set up the Atari environment."""
    env_raw = make_atari(env_name)  # Remove the f'{env_name}NoFrameskip-v4' part
    return wrap_deepmind(env_raw, frame_stack=True, episode_life=True, clip_rewards=False)

def initialize_networks(observation_shape: Tuple[int, ...], n_actions: int, config: Dict[str, Any], support: torch.Tensor) -> Tuple[nn.Module, nn.Module]:
    """Initialize the policy and target networks."""
    policy_net = m.RainbowDQN(observation_shape[2], observation_shape[3], n_actions, config['atom_size'], support)
    target_net = m.RainbowDQN(observation_shape[2], observation_shape[3], n_actions, config['atom_size'], support)
    target_net.load_state_dict(policy_net.state_dict())
    
    if config['device'] == 'cuda' and torch.cuda.is_available():
        if len(config['gpus']) > 1:
            policy_net = nn.DataParallel(policy_net, device_ids=config['gpus'])
            target_net = nn.DataParallel(target_net, device_ids=config['gpus'])
            print(f"Using GPUs: {config['gpus']}")
        policy_net = policy_net.to(config['device'])
        target_net = target_net.to(config['device'])
    else:
        policy_net = policy_net.to(config['device'])
        target_net = target_net.to(config['device'])
    
    target_net.eval()
    return policy_net, target_net

def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    memory: m.PrioritizedReplayMemory,
    batch_size: int,
    gamma: float,
    device: torch.device,
    scaler: GradScaler,
    support: torch.Tensor,
    n_atoms: int,
    v_min: float,
    v_max: float,
    curriculum_stage: int
) -> float:
    if len(memory) < batch_size:
        return None

    transitions, indices, weights = memory.sample(batch_size)
    batch = m.Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    with autocast():
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_probs = policy_net(state_batch)
        state_action_probs = state_action_probs[range(batch_size), action_batch.squeeze()]

        # Compute V(s_{t+1}) for all next states.
        next_state_probs = torch.zeros((batch_size, n_atoms), device=device)
        if non_final_mask.any():
            with torch.no_grad():
                next_state_probs[non_final_mask] = target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        reward_batch = reward_batch.unsqueeze(1)
        delta_z = (v_max - v_min) / (n_atoms - 1)
        tz = reward_batch + (gamma * support.unsqueeze(0))
        tz = tz.clamp(v_min, v_max)
        b = (tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Distribute probability of Tz
        projected_probs = torch.zeros_like(next_state_probs)
        projected_probs.scatter_add_(1, l, next_state_probs * (u.float() - b))
        projected_probs.scatter_add_(1, u, next_state_probs * (b - l.float()))

        # Compute loss
        loss = -(projected_probs * state_action_probs.log()).sum(1)
        td_errors = loss.detach()
        loss = (loss * torch.tensor(weights, device=device)).mean()

    # Optimize the model
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Update priorities
    new_priorities = td_errors.cpu().numpy() + 1e-6
    memory.update_priorities(indices, new_priorities)

    return loss.item()

def evaluate(policy_net: nn.Module, env: Any, device: torch.device, n_episodes: int, support: torch.Tensor) -> Dict[str, float]:
    """Evaluate the policy network."""
    policy_net.eval()
    rewards = []
    episode_lengths = []
    success_count = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            state = m.fp(np.array(obs)).to(device)
            with torch.no_grad():
                dist = policy_net(state)
                action = (dist * support).sum(2).max(1)[1].view(1, 1)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_reward > 0:  # Define your own success criterion
            success_count += 1

    policy_net.train()
    return {
        'mean_reward': np.mean(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'success_rate': success_count / n_episodes
    }

def adjust_reward(reward: float, curriculum_stage: int) -> float:
    """Adjust reward based on curriculum stage."""
    if curriculum_stage == 0:
        return reward * 2 if reward > 0 else reward  # Encourage any positive action
    elif curriculum_stage == 1:
        return reward * 1.5 if reward > 1 else reward  # Encourage breaking multiple bricks
    else:
        return reward  # No adjustment in later stages

def main() -> None:
    """Main training loop."""
    config = load_config('config.yaml')
    setup_logging(config['log_dir'])
    
    if config['device'] == 'cuda' and torch.cuda.is_available():
        if not config['gpus']:
            config['gpus'] = list(range(torch.cuda.device_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['gpus']))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    config['device'] = device
    
    v_min, v_max = config['v_min'], config['v_max']
    n_atoms = config['atom_size']
    support = torch.linspace(v_min, v_max, n_atoms).to(device)
    
    env = setup_environment(config['env_name'])

    obs, _ = env.reset()
    obs = m.fp(np.array(obs))
    observation_shape = obs.shape
    n_actions = env.action_space.n

    policy_net, target_net = initialize_networks(observation_shape, n_actions, config, support)
    optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'], eps=config['adam_eps'])
    memory = m.PrioritizedReplayMemory(config['memory_size'], 
                                       alpha=config['prioritized_replay_alpha'], 
                                       beta_start=config['prioritized_replay_beta'], 
                                       beta_frames=config['prioritized_replay_beta_frames'])
    action_selector = m.ActionSelector(config['eps_start'], config['eps_end'], 
                                       policy_net.module if isinstance(policy_net, nn.DataParallel) else policy_net, 
                                       config['eps_decay'], n_actions, device, support)

    writer = SummaryWriter(log_dir=config['log_dir'])
    os.makedirs(config['save_dir'], exist_ok=True)

    scaler = GradScaler(enabled=config['use_mixed_precision'])

    obs = m.fp(np.array(env.reset()[0])).to(device)
    episode_reward = 0
    episode_rewards = deque(maxlen=100)
    best_eval_reward = float('-inf')
    curriculum_stage = 0
    curriculum_thresholds = [50, 100, 150]  # Example thresholds for curriculum stages

    try:
        progress_bar = tqdm(range(config['num_steps']), desc="Training", ncols=100)
        for step in progress_bar:
            action, eps = action_selector.select_action(obs, train=True)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Apply curriculum learning
            adjusted_reward = adjust_reward(reward, curriculum_stage)
            reward = torch.tensor([adjusted_reward], device=device)

            next_state = m.fp(np.array(next_obs)).to(device) if not done else None
            memory.push(obs.cpu(), action.cpu(), next_state.cpu() if next_state is not None else None, reward.cpu(), done)

            obs = next_state if next_state is not None else m.fp(np.array(env.reset()[0])).to(device)

            episode_reward += reward.item()

            if step % config['policy_update'] == 0:
                loss = optimize_model(policy_net, target_net, optimizer, memory, 
                                      config['batch_size'], config['gamma'], device, scaler,
                                      support, n_atoms, v_min, v_max, curriculum_stage)
                if loss is not None:
                    writer.add_scalar('Loss/train', loss, global_step=step)

            if step % config['target_update'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                episode_rewards.append(episode_reward)
                avg_reward = np.mean(episode_rewards)
                writer.add_scalar('Reward/train', episode_reward, global_step=step)
                writer.add_scalar('Reward/average', avg_reward, global_step=step)
                writer.add_scalar('Epsilon', eps, global_step=step)
                
                # Update curriculum stage
                if avg_reward > curriculum_thresholds[curriculum_stage] and curriculum_stage < len(curriculum_thresholds) - 1:
                    curriculum_stage += 1
                    logging.info(f"Advancing to curriculum stage {curriculum_stage}")
                
                progress_bar.set_postfix({
                    'Reward': f'{episode_reward:.2f}',
                    'Avg Reward': f'{avg_reward:.2f}',
                    'Epsilon': f'{eps:.2f}',
                    'Curriculum': f'Stage {curriculum_stage}'
                })
                
                episode_reward = 0

            if (step + 1) % config['evaluate_freq'] == 0:
                eval_metrics = evaluate(policy_net.module if isinstance(policy_net, nn.DataParallel) else policy_net, 
                                        env, device, config['evaluate_num_episodes'], support)
                for metric_name, metric_value in eval_metrics.items():
                    writer.add_scalar(f'Eval/{metric_name}', metric_value, global_step=step)
                
                current_eval_reward = eval_metrics['mean_reward']
                is_best = current_eval_reward > best_eval_reward
                best_eval_reward = max(best_eval_reward, current_eval_reward)

                checkpoint = {
                    'step': step + 1,
                    'policy_net_state_dict': policy_net.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_eval_reward': best_eval_reward,
                    'curriculum_stage': curriculum_stage,
                    'config': config,
                }
                save_checkpoint(
                    checkpoint,
                    is_best,
                    os.path.join(config['save_dir'], f'{config["env_name"]}_checkpoint_step_{step+1}.pth'),
                    os.path.join(config['save_dir'], f'{config["env_name"]}_best_model.pth')
                )

            if (step + 1) % 100000 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= config['learning_rate_decay']
                    writer.add_scalar('Learning_rate', param_group['lr'], global_step=step)

# 定期进行垃圾回收和内存监控
            if step % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                writer.add_scalar('Memory/RSS', memory_info.rss / 1e6, global_step=step)  # RSS in MB
                writer.add_scalar('Memory/VMS', memory_info.vms / 1e6, global_step=step)  # VMS in MB

            # Reset noise for the noisy layers
            policy_net.reset_noise()
            target_net.reset_noise()

    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving final model...")
    except Exception as e:
        logging.error(f"An error occurred during training: {str(e)}")
        raise
    finally:
        final_checkpoint = {
            'step': config['num_steps'],
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_eval_reward': best_eval_reward,
            'curriculum_stage': curriculum_stage,
            'config': config,
        }
        final_model_path = os.path.join(config['save_dir'], f'{config["env_name"]}_final_model.pth')
        torch.save(final_checkpoint, final_model_path)
        logging.info(f"Training completed. Final model saved to {final_model_path}")

        writer.close()
        print("Training completed.")

def save_checkpoint(state: Dict[str, Any], is_best: bool, filename: str, best_filename: str) -> None:
    """Save model checkpoint."""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)
        logging.info(f"Saved new best model to {best_filename}")

if __name__ == "__main__":
    main()