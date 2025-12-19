"""
PARS Training Script with Visualization and Evaluation
"""

import gym
import d4rl
import numpy as np
from tqdm import tqdm
import wandb
from collections import defaultdict

from algorithms.pars import PARSAgent, get_config
from algorithms.td3_bc import TD3BCAgent, get_config

from utils.datasets import ReplayBuffer
from utils.visualization import PARSVisualizer


def evaluate_policy(agent, env, num_episodes=10):
    """Evaluate policy performance"""
    returns = []
    successes = []
    
    for ep in range(num_episodes):
        obs, done = env.reset(), False
        episode_return = 0
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # eval_mode=True로 deterministic action 샘플링
            action = agent.sample_actions(
                observations=obs[None],
                seed=None,  #
                eval_mode=True,  # static argument
            )
            action = np.array(action[0])
            
            obs, reward, done, info = env.step(action)
            episode_return += reward
            steps += 1
            
            # Check success (AntMaze)
            if reward > 0:  # Goal reached
                successes.append(1)
                break
        else:
            successes.append(0)
        
        returns.append(episode_return)
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    success_rate = np.mean(successes)
    
    return mean_return, std_return, success_rate


def main():
    # ========== Configuration ==========
    ENV_NAME = 'antmaze-umaze-v0'
    SEED = 0
    TOTAL_STEPS = 1_000_000
    EVAL_FREQ = 10_000  # Evaluate every 10k steps
    VIS_FREQ = 50_000   # Visualize Q-distribution every 50k steps
    LOG_FREQ = 1000     # Log metrics every 1k steps
    
    # Set random seeds
    np.random.seed(SEED)
    
    # ========== Initialize WandB ==========
    wandb.init(
        project="pars-qgap-analysis",
        entity="sophia435256-robros",
        config={
            'env': ENV_NAME,
            'seed': SEED,
            'total_steps': TOTAL_STEPS,
        },
        name=f"pars_{ENV_NAME}_seed{SEED}",
    )
    
    print(f"\n{'='*70}")
    print(f"PARS Training with Q-Distribution Analysis")
    print(f"{'='*70}")
    print(f"Environment: {ENV_NAME}")
    print(f"Total Steps: {TOTAL_STEPS:,}")
    print(f"Seed: {SEED}")
    print(f"{'='*70}\n")
    
    # ========== Environment Setup ==========
    env = gym.make(ENV_NAME)
    dataset = d4rl.qlearning_dataset(env)
    
    # ========== Replay Buffer ==========
    buffer = ReplayBuffer.create_from_d4rl(dataset=dataset)
    
    # ========== PARS Agent ==========
    config = get_config()
    config['reward_scale'] = 1000
    config['alpha'] = 0.001
    config['beta'] = 0.01
    
    agent = PARSAgent.create(
        seed=SEED,
        ex_observations=dataset['observations'][:1],
        ex_actions=dataset['actions'][:1],
        config=config,
    )
    # agent = TD3BCAgent.create(
    #     seed=SEED,
    #     ex_observations=dataset['observations'][:1],
    #     ex_actions=dataset['actions'][:1],
    #     config=config,
    # )
    
    # ========== Visualizer ==========
    visualizer = PARSVisualizer(save_dir=f'./figures/{ENV_NAME}')
    
    # ========== Training History ==========
    history = defaultdict(list)
    
    # ========== Training Loop ==========
    print("\n🚀 Starting training...\n")
    
    pbar = tqdm(range(TOTAL_STEPS), desc="Training", ncols=100)
    
    for step in pbar:
        # Sample batch and update
        batch = buffer.sample(config['batch_size'])
        update_actor = (step % config['actor_update_freq'] == 0)
        agent, info = agent.update(batch, update_actor=update_actor)
        
        # ========== Logging ==========
        if step % LOG_FREQ == 0:
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
                'Q_gap': f"{q_gap_true:.1f}",
                'Q_ID': f"{info['critic/q_mean']:.1f}",
                'Q_inf': f"{info['critic/infeasible_q_mean']:.1f}",
                'Viol': f"{info['critic/q_min_violations']*100:.1f}%"
            })
            
            # WandB loggingq_gap_raw
            wandb.log(info, step=step)
        
        # ========== Q-Distribution Visualization ==========
        if step % VIS_FREQ == 0 and step > 0:
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
        if step % EVAL_FREQ == 0 and step > 0:
            print(f"\n🎯 Evaluating at step {step}...")
            
            eval_return, eval_std, success_rate = evaluate_policy(
                agent, env, num_episodes=100
            )
            normalized_score = env.get_normalized_score(eval_return) * 100
            
            # Store evaluation results
            history['eval_steps'].append(step)
            history['eval_returns'].append(eval_return)
            history['eval_stds'].append(eval_std)
            history['eval_success_rates'].append(success_rate)
            history['eval_normalized_scores'].append(normalized_score)
            
            print(f"  Return: {eval_return:.3f} ± {eval_std:.3f}")
            print(f"  Normalized Score: {normalized_score:.2f}")
            print(f"  Success Rate: {success_rate:.1%}")
            
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
        agent, env, num_episodes=100
    )
    final_score = env.get_normalized_score(final_return) * 100
    
    print(f"\n{'='*70}")
    print("Final Evaluation Results")
    print(f"{'='*70}")
    print(f"Return:           {final_return:.3f} ± {final_std:.3f}")
    print(f"Normalized Score: {final_score:.2f}")
    print(f"Success Rate:     {final_success:.1%}")
    print(f"{'='*70}\n")
    
    # Final visualization
    print("📊 Generating final visualizations...")
    visualizer.plot_q_distribution(agent, buffer, TOTAL_STEPS, num_samples=1000)
    visualizer.plot_learning_curves(history, TOTAL_STEPS)
    print("✓ Final visualizations saved\n")
    
    # Save final results to WandB
    wandb.summary['final_return'] = final_return
    wandb.summary['final_normalized_score'] = final_score
    wandb.summary['final_success_rate'] = final_success
    
    wandb.finish()
    
    print("✅ Training and evaluation complete!")


if __name__ == "__main__":
    main()