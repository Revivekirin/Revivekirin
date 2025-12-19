"""
Visualization utilities for PARS training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import jax.numpy as jnp


class PARSVisualizer:
    """Real-time visualization for PARS training"""
    
    def __init__(self, save_dir: str = './figures'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (15, 10)
    
    def plot_q_distribution(
        self,
        agent,
        buffer,
        step: int,
        num_samples: int = 1000
    ):
        """Plot Q-value distributions for ID and infeasible actions"""
        
        # Sample batch
        batch = buffer.sample(num_samples)
        
        # Get ID Q-values
        q_id = agent.network.select('critic')(
            batch['observations'],
            actions=batch['actions']
        )
        q_id = np.array(q_id).flatten()
        
        # Get infeasible Q-values
        infeasible_actions = agent._sample_infeasible_actions(
            agent.rng,
            num_samples,
            batch['actions'].shape[1]
        )
        q_infeasible = agent.network.select('critic')(
            batch['observations'],
            actions=infeasible_actions
        )
        q_infeasible = np.array(q_infeasible).flatten()
        
        # Q_min line
        q_min = agent.config['q_min']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Histogram comparison
        ax = axes[0, 0]
        ax.hist(q_id, bins=50, alpha=0.6, label='Q_ID', color='blue', density=True)
        ax.hist(q_infeasible, bins=50, alpha=0.6, label='Q_infeasible', color='red', density=True)
        ax.axvline(q_min, color='black', linestyle='--', linewidth=2, label=f'Q_min={q_min:.1f}')
        ax.set_xlabel('Q-value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Q-value Distribution (Step {step:,})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax = axes[0, 1]
        box_data = [q_id, q_infeasible]
        bp = ax.boxplot(box_data, labels=['Q_ID', 'Q_infeasible'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        for box in bp['boxes']:
            box.set_alpha(0.6)
        ax.axhline(q_min, color='black', linestyle='--', linewidth=2, label=f'Q_min={q_min:.1f}')
        ax.set_ylabel('Q-value', fontsize=12)
        ax.set_title('Q-value Box Plot', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Violation analysis
        ax = axes[1, 0]
        violations_id = (q_id < q_min).mean() * 100
        violations_inf = (q_infeasible < q_min).mean() * 100
        bars = ax.bar(['Q_ID', 'Q_infeasible'], [violations_id, violations_inf], 
                      color=['blue', 'red'], alpha=0.6)
        ax.set_ylabel('Violation Rate (%)', fontsize=12)
        ax.set_title(f'Q < Q_min Violations', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        for bar, val in zip(bars, [violations_id, violations_inf]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_data = [
            ['Metric', 'Q_ID', 'Q_infeasible'],
            ['Mean', f'{q_id.mean():.2f}', f'{q_infeasible.mean():.2f}'],
            ['Std', f'{q_id.std():.2f}', f'{q_infeasible.std():.2f}'],
            ['Min', f'{q_id.min():.2f}', f'{q_infeasible.min():.2f}'],
            ['Max', f'{q_id.max():.2f}', f'{q_infeasible.max():.2f}'],
            ['Q_gap (raw)', f'{q_id.mean() - q_infeasible.mean():.2f}', ''],
            ['Q_gap (true)', f'{q_id.mean() - max(q_infeasible.mean(), q_min):.2f}', ''],
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(stats_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/q_distribution_step_{step}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return statistics
        return {
            'q_id_mean': q_id.mean(),
            'q_id_std': q_id.std(),
            'q_infeasible_mean': q_infeasible.mean(),
            'q_infeasible_std': q_infeasible.std(),
            'violations_id': violations_id,
            'violations_inf': violations_inf,
            'q_gap_raw': q_id.mean() - q_infeasible.mean(),
            'q_gap_true': q_id.mean() - max(q_infeasible.mean(), q_min),
        }
    
    def plot_learning_curves(
        self,
        history: Dict[str, List],
        step: int
    ):
        """Plot learning curves"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Q Gap (both raw and true)
        ax = axes[0, 0]
        if 'q_gap_raw' in history:
            ax.plot(history['steps'], history['q_gap_raw'], 
                   label='Q Gap (raw)', linewidth=2, alpha=0.7)
        if 'q_gap_true' in history:
            ax.plot(history['steps'], history['q_gap_true'], 
                   label='Q Gap (true)', linewidth=2, alpha=0.7)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Q Gap', fontsize=12)
        ax.set_title('Q Gap Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. Losses
        ax = axes[0, 1]
        if 'td_loss' in history:
            ax.plot(history['steps'], history['td_loss'], 
                   label='TD Loss', linewidth=2, alpha=0.7)
        if 'pa_loss' in history:
            ax.plot(history['steps'], history['pa_loss'], 
                   label='PA Loss', linewidth=2, alpha=0.7)
        if 'bc_loss' in history:
            ax.plot(history['steps'], history['bc_loss'], 
                   label='BC Loss', linewidth=2, alpha=0.7)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. Q-values
        ax = axes[0, 2]
        if 'q_mean' in history:
            ax.plot(history['steps'], history['q_mean'], 
                   label='Q_ID', linewidth=2, alpha=0.7, color='blue')
        if 'infeasible_q_mean' in history:
            ax.plot(history['steps'], history['infeasible_q_mean'], 
                   label='Q_infeasible', linewidth=2, alpha=0.7, color='red')
        if 'q_min' in history and len(history['q_min']) > 0:
            ax.axhline(history['q_min'][0], color='black', 
                      linestyle='--', linewidth=2, label='Q_min')
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Q-value', fontsize=12)
        ax.set_title('Q-value Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 4. Violations
        ax = axes[1, 0]
        if 'q_min_violations' in history:
            violations_pct = np.array(history['q_min_violations']) * 100
            ax.plot(history['steps'], violations_pct, 
                   linewidth=2, alpha=0.7, color='orange')
            ax.fill_between(history['steps'], 0, violations_pct, alpha=0.3, color='orange')
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Violation Rate (%)', fontsize=12)
        ax.set_title('Q_infeasible < Q_min Violations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        # 5. PA Active Ratio
        ax = axes[1, 1]
        if 'pa_active_ratio' in history:
            active_pct = np.array(history['pa_active_ratio']) * 100
            ax.plot(history['steps'], active_pct, 
                   linewidth=2, alpha=0.7, color='green')
            ax.fill_between(history['steps'], 0, active_pct, alpha=0.3, color='green')
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Active Ratio (%)', fontsize=12)
        ax.set_title('PA Penalty Active Ratio', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
        # 6. Evaluation Performance
        ax = axes[1, 2]
        if 'eval_returns' in history and len(history['eval_returns']) > 0:
            ax.plot(history['eval_steps'], history['eval_returns'], 
                   'o-', linewidth=2, markersize=6, alpha=0.7)
        ax.set_xlabel('Steps', fontsize=12)
        ax.set_ylabel('Return', fontsize=12)
        ax.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/learning_curves_step_{step}.png', dpi=300, bbox_inches='tight')
        plt.close()