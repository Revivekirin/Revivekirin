"""
PARS Training Script with Visualization and Evaluation
Supports both D4RL (AntMaze) and Robomimic environments
"""

import gym
import numpy as np
from tqdm import tqdm
import wandb
from collections import defaultdict
import argparse

# D4RL import (optional)
try:
    import d4rl
    D4RL_AVAILABLE = True
except ImportError:
    D4RL_AVAILABLE = False
    print("Warning: d4rl not available. D4RL environments will not work.")

# Robomimic import (optional)
try:
    import robomimic
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    ROBOMIMIC_AVAILABLE = True
except ImportError:
    ROBOMIMIC_AVAILABLE = False
    print("Warning: robomimic not available. Robomimic environments will not work.")

from algorithms.pars import PARSAgent, get_config
from utils.datasets import ReplayBuffer
from utils.visualization import PARSVisualizer


def is_robomimic_env(env_name: str) -> bool:
    """Check if environment is from Robomimic"""
    robomimic_keywords = ['lift', 'can', 'square', 'transport', 'tool_hang', 
                          'pickplace', 'threading', 'image', 'low_dim']
    return any(keyword in env_name.lower() for keyword in robomimic_keywords)


def make_robomimic_env(env_name: str, dataset_path: str = None):
    """Create Robomimic environment and load dataset
    
    Args:
        env_name: Environment name (e.g., 'lift-ph-image-v0')
        dataset_path: Path to hdf5 dataset file
        
    Returns:
        env, dataset dict
    """
    if not ROBOMIMIC_AVAILABLE:
        raise ImportError("robomimic is not installed. Install with: pip install robomimic")
    
    # Parse environment name
    # Format: {task}-{dataset_type}-{obs_modality}-v{version}
    # Example: lift-ph-image-v0 (lift task, proficient-human data, image obs)
    parts = env_name.split('-')
    task_name = parts[0]  # lift, can, square, etc.
    dataset_type = parts[1]  # ph (proficient-human), mh (multi-human), mg (machine-generated)
    obs_modality = parts[2]  # image, low_dim
    
    # Default dataset path
    if dataset_path is None:
        # Robomimic datasets are usually in ~/robomimic/datasets/
        import os
        dataset_path = os.path.expanduser(
            f"~/robomimic/datasets/{task_name}/{dataset_type}/{obs_modality}.hdf5"
        )
    
    print(f"Loading Robomimic environment: {env_name}")
    print(f"Dataset path: {dataset_path}")
    
    # Load dataset metadata
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    
    # Create environment
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
    )
    
    # Create eval environment (with rendering for video)
    eval_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,  # For video recording
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset_dict = load_robomimic_dataset(dataset_path, obs_modality)
    
    return env, eval_env, dataset_dict


def load_robomimic_dataset(dataset_path: str, obs_modality: str):
    """Load Robomimic dataset from hdf5 file
    
    Args:
        dataset_path: Path to hdf5 file
        obs_modality: 'image' or 'low_dim'
        
    Returns:
        Dictionary compatible with ReplayBuffer
    """
    import h5py
    
    dataset = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminals': [],
    }
    
    with h5py.File(dataset_path, 'r') as f:
        demos = list(f['data'].keys())
        
        print(f"Found {len(demos)} demonstrations")
        
        for demo_name in tqdm(demos, desc="Loading demos"):
            demo = f[f'data/{demo_name}']
            
            # Observations
            if obs_modality == 'image':
                # For image observations, use 'agentview_image' or 'robot0_eye_in_hand_image'
                if 'obs/agentview_image' in demo:
                    obs = demo['obs/agentview_image'][:]
                elif 'obs/robot0_eye_in_hand_image' in demo:
                    obs = demo['obs/robot0_eye_in_hand_image'][:]
                else:
                    # Fallback: find first image observation
                    obs_keys = [k for k in demo['obs'].keys() if 'image' in k]
                    if obs_keys:
                        obs = demo[f'obs/{obs_keys[0]}'][:]
                    else:
                        raise ValueError(f"No image observations found in {demo_name}")
                
                # Normalize images to [0, 1]
                obs = obs.astype(np.float32) / 255.0
                
            else:  # low_dim
                # Concatenate all low-dim observations
                obs_keys = [k for k in demo['obs'].keys() 
                           if 'image' not in k and 'depth' not in k]
                obs_list = [demo[f'obs/{k}'][:] for k in sorted(obs_keys)]
                obs = np.concatenate(obs_list, axis=-1)
            
            # Actions
            actions = demo['actions'][:]
            
            # Rewards (Robomimic uses sparse rewards)
            rewards = demo['rewards'][:]
            
            # Terminals (last step of trajectory)
            terminals = np.zeros(len(rewards), dtype=np.float32)
            terminals[-1] = 1.0
            
            # Add to dataset
            dataset['observations'].append(obs)
            dataset['actions'].append(actions)
            dataset['rewards'].append(rewards)
            dataset['terminals'].append(terminals)
    
    # Concatenate all demonstrations
    dataset = {
        'observations': np.concatenate(dataset['observations'], axis=0),
        'actions': np.concatenate(dataset['actions'], axis=0),
        'rewards': np.concatenate(dataset['rewards'], axis=0),
        'terminals': np.concatenate(dataset['terminals'], axis=0),
    }
    
    # Add next_observations
    dataset['next_observations'] = np.concatenate([
        dataset['observations'][1:],
        dataset['observations'][-1:],
    ], axis=0)
    
    print(f"Dataset loaded: {len(dataset['observations'])} transitions")
    print(f"Observation shape: {dataset['observations'].shape}")
    print(f"Action shape: {dataset['actions'].shape}")
    
    return dataset


def make_d4rl_env(env_name: str):
    """Create D4RL environment and load dataset"""
    if not D4RL_AVAILABLE:
        raise ImportError("d4rl is not installed. Install with: pip install git+https://github.com/Farama-Foundation/d4rl")
    
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    
    return env, env, dataset  # D4RL uses same env for train and eval


def evaluate_policy(agent, env, num_episodes=10, is_robomimic=False):
    """Evaluate policy performance
    
    Args:
        agent: PARS agent
        env: Environment
        num_episodes: Number of evaluation episodes
        is_robomimic: Whether environment is Robomimic
        
    Returns:
        mean_return, std_return, success_rate
    """
    returns = []
    successes = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, dict):  # Robomimic returns dict
            obs = obs['agentview_image'] if 'agentview_image' in obs else obs['robot0_eye_in_hand']
        
        done = False
        episode_return = 0
        steps = 0
        max_steps = 500 if is_robomimic else 1000
        
        while not done and steps < max_steps:
            # Sample action
            action = agent.sample_actions(
                observations=obs[None],
                seed=None,
                eval_mode=True,
            )
            action = np.array(action[0])
            
            # Step environment
            step_output = env.step(action)
            
            if is_robomimic:
                # Robomimic returns (obs_dict, reward, done, info)
                obs_dict, reward, done, info = step_output
                obs = obs_dict['agentview_image'] if 'agentview_image' in obs_dict else obs_dict['robot0_eye_in_hand']
            else:
                # D4RL returns (obs, reward, done, info)
                obs, reward, done, info = step_output
            
            episode_return += reward
            steps += 1
            
            # Check success
            if is_robomimic:
                # Robomimic has success flag in info
                if info.get('success', False):
                    successes.append(1)
                    break
            else:
                # D4RL AntMaze: reward > 0 means success
                if reward > 0:
                    successes.append(1)
                    break
        else:
            successes.append(0)
        
        returns.append(episode_return)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(successes) if successes else 0.0
    
    return mean_return, std_return, success_rate


def main():
    # ========== Argument Parsing ==========
    parser = argparse.ArgumentParser(description='PARS Training Script')
    
    # Environment
    parser.add_argument('--env_name', type=str, default='antmaze-umaze-v0',
                       help='Environment name (D4RL or Robomimic)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to Robomimic dataset (hdf5 file)')
    
    # Training
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--total_steps', type=int, default=1_000_000,
                       help='Total training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    
    # PARS hyperparameters
    parser.add_argument('--reward_scale', type=float, default=1000.0,
                       help='Reward scaling factor')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='PA loss weight')
    parser.add_argument('--beta', type=float, default=0.01,
                       help='BC loss weight')
    
    # Logging
    parser.add_argument('--eval_freq', type=int, default=10_000,
                       help='Evaluation frequency')
    parser.add_argument('--vis_freq', type=int, default=50_000,
                       help='Visualization frequency')
    parser.add_argument('--log_freq', type=int, default=1000,
                       help='Logging frequency')
    
    # WandB
    parser.add_argument('--project', type=str, default='pars-unified',
                       help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                       help='WandB entity')
    parser.add_argument('--group', type=str, default=None,
                       help='WandB group name')
    
    args = parser.parse_args()
    
    # ========== Environment Detection ==========
    is_robomimic = is_robomimic_env(args.env_name)
    
    print(f"\n{'='*70}")
    print(f"PARS Training with Q-Distribution Analysis")
    print(f"{'='*70}")
    print(f"Environment: {args.env_name}")
    print(f"Type: {'Robomimic' if is_robomimic else 'D4RL'}")
    print(f"Total Steps: {args.total_steps:,}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # ========== Initialize WandB ==========
    wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        config=vars(args),
        name=f"pars_{args.env_name}_seed{args.seed}",
    )
    
    # ========== Environment Setup ==========
    if is_robomimic:
        env, eval_env, dataset = make_robomimic_env(args.env_name, args.dataset_path)
    else:
        env, eval_env, dataset = make_d4rl_env(args.env_name)
    
    # ========== Replay Buffer ==========
    buffer = ReplayBuffer.create_from_d4rl(dataset=dataset)
    
    # ========== PARS Agent ==========
    config = get_config()
    config['reward_scale'] = args.reward_scale
    config['alpha'] = args.alpha
    config['beta'] = args.beta
    config['batch_size'] = args.batch_size
    
    # Robomimic-specific adjustments
    if is_robomimic:
        config['discount'] = 0.99  # Shorter horizon for manipulation
        config['encoder'] = 'resnet'  # Use CNN encoder for images
        print("Using ResNet encoder for image observations")
    else:
        config['discount'] = 0.995  # Longer horizon for navigation
        config['encoder'] = None
    
    agent = PARSAgent.create(
        seed=args.seed,
        ex_observations=dataset['observations'][:1],
        ex_actions=dataset['actions'][:1],
        config=config,
    )
    
    # ========== Visualizer ==========
    visualizer = PARSVisualizer(save_dir=f'./figures/{args.env_name}')
    
    # ========== Training History ==========
    history = defaultdict(list)
    
    # ========== Training Loop ==========
    print("\n🚀 Starting training...\n")
    
    pbar = tqdm(range(args.total_steps), desc="Training", ncols=100)
    
    for step in pbar:
        # Sample batch and update
        batch = buffer.sample(config['batch_size'])
        update_actor = (step % config['actor_update_freq'] == 0)
        agent, info = agent.update(batch, update_actor=update_actor)
        
        # ========== Logging ==========
        if step % args.log_freq == 0:
            # Store metrics
            history['steps'].append(step)
            for key, value in info.items():
                if 'critic/' in key:
                    metric_name = key.replace('critic/', '')
                    history[metric_name].append(float(value))
                elif 'actor/' in key:
                    metric_name = key.replace('actor/', '')
                    history[metric_name].append(float(value))
            
            # Update progress bar
            q_gap_true = info.get('critic/q_gap_true', info.get('critic/q_gap_raw', 0))
            pbar.set_postfix({
                'Gap': f"{q_gap_true:.1f}",
                'Q_ID': f"{info['critic/q_mean']:.1f}",
                'Q_inf': f"{info['critic/infeasible_q_mean']:.1f}",
                'Viol%': f"{info['critic/q_min_violations']*100:.1f}"
            })
            
            # WandB logging
            wandb.log(info, step=step)
        
        # ========== Q-Distribution Visualization ==========
        if step % args.vis_freq == 0 and step > 0:
            print(f"\n📊 Generating Q-distribution visualization at step {step}...")
            
            vis_stats = visualizer.plot_q_distribution(
                agent=agent,
                buffer=buffer,
                step=step,
                num_samples=1000
            )
            
            # Log visualization stats
            wandb.log({
                'vis/q_id_mean': vis_stats['q_id_mean'],
                'vis/q_infeasible_mean': vis_stats['q_infeasible_mean'],
                'vis/q_gap_raw': vis_stats['q_gap_raw'],
                'vis/q_gap_true': vis_stats['q_gap_true'],
                'vis/violations_inf': vis_stats['violations_inf'],
            }, step=step)
            
            # Plot learning curves
            visualizer.plot_learning_curves(history, step)
            
            print(f"✓ Visualization saved")
        
        # ========== Evaluation ==========
        if step % args.eval_freq == 0 and step > 0:
            print(f"\n🎯 Evaluating at step {step}...")
            
            eval_return, eval_std, success_rate = evaluate_policy(
                agent, eval_env, num_episodes=50, is_robomimic=is_robomimic
            )
            
            # Normalized score
            if is_robomimic:
                # Robomimic doesn't have get_normalized_score
                normalized_score = success_rate * 100  # Use success rate as score
            else:
                normalized_score = env.get_normalized_score(eval_return) * 100
            
            # Store evaluation results
            history['eval_steps'].append(step)
            history['eval_returns'].append(eval_return)
            history['eval_stds'].append(eval_std)
            history['eval_success_rates'].append(success_rate)
            history['eval_normalized_scores'].append(normalized_score)
            
            print(f"  Return: {eval_return:.3f} ± {eval_std:.3f}")
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Score: {normalized_score:.2f}")
            
            # WandB logging
            wandb.log({
                'eval/return': eval_return,
                'eval/return_std': eval_std,
                'eval/normalized_score': normalized_score,
                'eval/success_rate': success_rate,
            }, step=step)
    
    # ========== Final Evaluation ==========
    print(f"\n{'='*70}")
    print("🏁 Training Complete! Running final evaluation...")
    print(f"{'='*70}\n")
    
    final_return, final_std, final_success = evaluate_policy(
        agent, eval_env, num_episodes=100, is_robomimic=is_robomimic
    )
    
    if is_robomimic:
        final_score = final_success * 100
    else:
        final_score = env.get_normalized_score(final_return) * 100
    
    print(f"\n{'='*70}")
    print("Final Evaluation Results")
    print(f"{'='*70}")
    print(f"Return:        {final_return:.3f} ± {final_std:.3f}")
    print(f"Success Rate:  {final_success:.1%}")
    print(f"Score:         {final_score:.2f}")
    print(f"{'='*70}\n")
    
    # Final visualization
    print("📊 Generating final visualizations...")
    visualizer.plot_q_distribution(agent, buffer, args.total_steps, num_samples=1000)
    visualizer.plot_learning_curves(history, args.total_steps)
    print("✓ Final visualizations saved\n")
    
    # Save final results to WandB
    wandb.summary['final_return'] = final_return
    wandb.summary['final_score'] = final_score
    wandb.summary['final_success_rate'] = final_success
    
    wandb.finish()
    
    print("✅ Training and evaluation complete!")


if __name__ == "__main__":
    main()