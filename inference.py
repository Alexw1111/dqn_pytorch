"""
Inference script for trained Rainbow DQN model on Atari games.

This script loads a trained Rainbow DQN model and runs it on the specified Atari environment,
optionally recording videos of the gameplay.
"""

import os
import logging
from typing import Tuple, Any, Dict

import numpy as np
import torch
import yaml
import gymnasium as gym

import model as m
from atari_wrappers import wrap_deepmind, make_atari

def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('inference.log')
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise

def load_model(model_path: str, env_name: str, device: str, config: Dict[str, Any]) -> Tuple[m.RainbowDQN, Any, int]:
    """
    Load the trained model and set up the environment.

    Args:
        model_path (str): Path to the saved model checkpoint.
        env_name (str): Name of the Atari environment.
        device (str): Device to load the model on ('cpu' or 'cuda').
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Tuple[m.RainbowDQN, Any, int]: Loaded policy network, environment, and number of actions.

    Raises:
        FileNotFoundError: If the model file is not found.
        ValueError: If the checkpoint doesn't contain the expected data.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env_raw = make_atari(env_name)
    env = wrap_deepmind(env_raw, frame_stack=True, episode_life=False, clip_rewards=False)
    
    obs, _ = env.reset()
    obs = m.fp(np.array(obs))
    c, h, w = obs.shape[1], obs.shape[2], obs.shape[3]
    n_actions = env.action_space.n
    
    v_min, v_max = config['v_min'], config['v_max']
    atom_size = config['atom_size']
    support = torch.linspace(v_min, v_max, atom_size).to(device)
    
    policy_net = m.RainbowDQN(h, w, n_actions, atom_size, support).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'policy_net_state_dict' in checkpoint:
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        else:
            raise ValueError("The checkpoint does not contain 'policy_net_state_dict'")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

    policy_net.eval()
    return policy_net, env, n_actions

def run_inference(
    policy_net: m.RainbowDQN,
    env: Any,
    n_actions: int,
    device: str,
    evaluate_eps: float,
    support: torch.Tensor,
    num_episodes: int = 10,
    record_video: bool = True
) -> None:
    """
    Run the trained model on the environment and optionally record videos.

    Args:
        policy_net (m.RainbowDQN): Trained policy network.
        env (Any): Atari environment.
        n_actions (int): Number of possible actions.
        device (str): Device to run inference on.
        evaluate_eps (float): Epsilon value for evaluation.
        support (torch.Tensor): Support for the categorical DQN.
        num_episodes (int): Number of episodes to run.
        record_video (bool): Whether to record videos of gameplay.
    """
    sa = m.ActionSelector(evaluate_eps, evaluate_eps, policy_net, 0, n_actions, device, support)
    
    if record_video:
        try:
            env = gym.wrappers.RecordVideo(env, "videos", episode_trigger=lambda episode_id: True)
        except gym.error.DependencyNotInstalled:
            logging.warning("Unable to record video. moviepy is not installed. Continuing without video recording.")
            record_video = False
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = m.fp(np.array(obs))
        total_reward = 0
        done = False
        
        while not done:
            state = obs.to(device)
            if state.dim() == 3:
                state = state.unsqueeze(0)
            elif state.dim() == 5:
                state = state.squeeze(1)
            
            action, _ = sa.select_action(state, train=False)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            obs = m.fp(np.array(next_obs))
            
            total_reward += reward
        
        logging.info(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()

def main() -> None:
    """Main function to run the inference."""
    setup_logging()
    
    try:
        config = load_config('config.yaml')
        device = torch.device(config['device'])
        env_name = config['env_name']
        evaluate_eps = config['evaluate_eps']
        
        model_path = config['inference_model_path']
        
        v_min, v_max = config['v_min'], config['v_max']
        atom_size = config['atom_size']
        support = torch.linspace(v_min, v_max, atom_size).to(device)
        
        logging.info(f"Loading model from {model_path}")
        policy_net, env, n_actions = load_model(model_path, env_name, device, config)
        logging.info("Model loaded successfully")
        
        logging.info("Starting inference")
        run_inference(
            policy_net,
            env,
            n_actions,
            device,
            evaluate_eps,
            support,
            num_episodes=10,
            record_video=True
        )
        logging.info("Inference completed")
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
