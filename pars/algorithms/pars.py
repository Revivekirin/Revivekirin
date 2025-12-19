"""
PARS: Penalizing infeasible Actions and Reward Scaling
Extends TD3+BC with reward scaling and infeasible action penalization
"""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import ml_collections
from algorithms.td3_bc import TD3BCAgent


class PARSAgent(TD3BCAgent):
    """PARS agent extending TD3+BC.
    
    Key additions:
    1. Reward scaling (RS) with layer normalization (LN)
    2. Penalizing infeasible actions (PA)
    """

    def critic_loss(self, batch: Dict, grad_params: Any, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        """Compute PARS critic loss.
        
        L_Total = L_TD + α * L_PA
        
        where:
        - L_TD: Standard TD3 loss with reward scaling
        - L_PA: Penalty for infeasible actions
        """
        rng, sample_rng, infeasible_rng = jax.random.split(rng, 3)
        
        # ===== Standard TD Loss with Reward Scaling =====
        
        # Scale rewards (RS component)
        scaled_rewards = self.config['reward_scale'] * batch['rewards']
        
        # Target actions with noise
        next_dist = self.network.select('target_actor')(batch['next_observations'])
        next_actions = next_dist.mode()
        noise = jnp.clip(
            jax.random.normal(sample_rng, next_actions.shape) * self.config['target_noise'],
            -self.config['noise_clip'],
            self.config['noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        # Target Q-values
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        next_q = next_qs.min(axis=0)
        
        target_q = scaled_rewards + self.config['discount'] * batch['masks'] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q-values
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        
        # TD loss
        td_loss = jnp.square(q - target_q).mean()

        # ===== Penalizing Infeasible Actions (PA) =====
        
        # Sample infeasible actions
        batch_size = batch['observations'].shape[0]
        action_dim = batch['actions'].shape[1]
        infeasible_actions = self._sample_infeasible_actions(
            infeasible_rng, 
            batch_size, 
            action_dim
        )
        
        # Q-values for infeasible actions
        infeasible_q = self.network.select('critic')(
            batch['observations'], 
            actions=infeasible_actions, 
            params=grad_params
        )
        
        # Penalize to Q_min
        pa_loss = jnp.square(infeasible_q - self.config['q_min']).mean()
        
        # Total critic loss
        total_loss = td_loss + self.config['alpha'] * pa_loss

        info = {
            'critic_loss': total_loss,
            'td_loss': td_loss,
            'pa_loss': pa_loss,
            'q_mean': q.mean(),
            'q_std': q.std(),
            'infeasible_q_mean': infeasible_q.mean(),
            'infeasible_q_std': infeasible_q.std(),
            'q_gap': q.mean() - infeasible_q.mean(),  # Gap between ID and OOD Q-values
            'target_q_mean': target_q.mean(),
        }

        return total_loss, info

    def _sample_infeasible_actions(
        self, 
        rng: jax.random.PRNGKey, 
        batch_size: int, 
        action_dim: int
    ) -> jnp.ndarray:
        """Sample infeasible actions from the infeasible region.
        
        For action space [-1, 1]^n, infeasible region is defined as:
        A_I = [2*L_I, L_I] ∪ [U_I, 2*U_I]
        
        where L_I < -1 and U_I > 1 (with guard intervals).
        
        Args:
            rng: Random key
            batch_size: Number of actions to sample
            action_dim: Action dimension
            
        Returns:
            Infeasible actions of shape (batch_size, action_dim)
        """
        L = self.config['L_infeasible']  # Distance from feasible boundary
        
        # Sample uniformly in [-1, 1]
        actions = jax.random.uniform(
            rng,
            shape=(batch_size, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
        
        # Map to infeasible region: [2*L, L] or [U, 2*U]
        # If action < 0: map to [-2L, -L]
        # If action >= 0: map to [L, 2L]
        infeasible_actions = jnp.where(
            actions < 0,
            (actions - 1) * L,  # Negative side
            (actions + 1) * L,  # Positive side
        )
        
        return infeasible_actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: ml_collections.ConfigDict,
    ) -> 'PARSAgent':
        """Create a new PARS agent.
        
        Adds PARS-specific configuration on top of TD3+BC.
        """
        # Compute Q_min for penalization
        # For sparse reward tasks like AntMaze: Q_min = reward_scale * min_reward / (1 - γ)
        min_reward = config.get('min_reward', 0.0)  # AntMaze has min_reward = 0
        q_min = config['reward_scale'] * min_reward / (1 - config['discount'])
        config['q_min'] = q_min
        
        # Force layer normalization for critic (essential for PARS)
        config['layer_norm'] = True
        
        print(f"\n{'='*60}")
        print(f"Creating PARS Agent")
        print(f"{'='*60}")
        print(f"Reward scale: {config['reward_scale']}")
        print(f"Alpha (PA weight): {config['alpha']}")
        print(f"Beta (BC weight): {config['beta']}")
        print(f"Q_min: {q_min:.3f}")
        print(f"L_infeasible: {config['L_infeasible']}")
        print(f"Layer Norm: {config['layer_norm']}")
        print(f"{'='*60}\n")
        
        # Create using parent TD3+BC
        return super().create(seed, ex_observations, ex_actions, config)


def get_config() -> ml_collections.ConfigDict:
    """Get default PARS configuration."""
    # Start with TD3+BC config
    from algorithms.td3_bc import get_config as get_td3bc_config
    config = get_td3bc_config()
    
    # Override with PARS-specific settings
    config.update({
        'agent_name': 'pars',
        
        # PARS hyperparameters
        'reward_scale': 100.0,  # Reward scaling factor (creward)
        'alpha': 0.001,  # PA loss weight
        'beta': 0.01,  # BC loss weight
        'L_infeasible': 1000.0,  # Distance to infeasible region
        'min_reward': 0.0,  # Minimum reward (for Q_min calculation)
        
        # Network (force layer norm)
        'layer_norm': True,  # Critical for PARS
        'critic_hidden_dims': (256, 256, 256),
        'actor_hidden_dims': (256, 256, 256),
        
        # TD3 settings
        'discount': 0.995,  # Higher for AntMaze
        'tau': 0.005,
        'actor_update_freq': 2,
        'target_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
    })
    
    return config