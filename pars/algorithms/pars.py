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
    2. Penalizing infeasible actions (PA) with proper constraints
    """

    def critic_loss(self, batch: Dict, grad_params: Any, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        """Compute PARS critic loss with proper Q_infeasible handling.
        
        L_Total = L_TD + α * L_PA
        
        where:
        - L_TD: Standard TD3 loss with reward scaling
        - L_PA: One-sided penalty for infeasible actions (only if Q > Q_min)
        """
        rng, sample_rng, infeasible_rng = jax.random.split(rng, 3)
        
        # ===== Standard TD Loss with Reward Scaling =====
        
        # Scale rewards (RS component)
        scaled_rewards = self.config['reward_scale'] * batch['rewards']
        scaled_rewards = scaled_rewards.squeeze(-1)  # Ensure (batch_size,)
        masks = batch['masks'].squeeze(-1)  # Ensure (batch_size,)
        
        # Target actions with noise
        next_dist = self.network.select('target_actor')(batch['next_observations'])
        next_actions = next_dist.mode()
        noise = jnp.clip(
            jax.random.normal(sample_rng, next_actions.shape) * self.config['target_noise'],
            -self.config['noise_clip'],
            self.config['noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        # Target Q-values (ensemble minimum)
        next_qs = self.network.select('target_critic')(
            batch['next_observations'], 
            actions=next_actions
        )
        next_q = next_qs.min(axis=0)  # Shape: (batch_size,)
        
        target_q = scaled_rewards + self.config['discount'] * masks * next_q
        target_q = jax.lax.stop_gradient(target_q)  # Shape: (batch_size,)

        # Current Q-values
        q = self.network.select('critic')(
            batch['observations'], 
            actions=batch['actions'], 
            params=grad_params
        )
        # q shape: (num_critics, batch_size)
        
        # Compute TD loss for each critic and average
        target_q_broadcasted = jnp.broadcast_to(target_q[None, :], q.shape)
        td_loss = jnp.square(q - target_q_broadcasted).mean()

        # ===== Penalizing Infeasible Actions (PA) with One-sided Penalty =====
        
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
        # infeasible_q shape: (num_critics, batch_size)
        
        # ===== One-sided Penalty: Only penalize if Q > Q_min =====
        q_min = self.config['q_min']
        q_excess = jnp.maximum(infeasible_q - q_min, 0.0)
        # pa_loss = jnp.square(q_excess).mean()
        pa_loss = jnp.square(infeasible_q - q_min).mean()
        
        # Total critic loss
        total_loss = td_loss + self.config['alpha'] * pa_loss

        # ===== Compute Detailed Statistics =====
        
        # ID Q-values
        q_mean = q.mean()
        q_std = q.std()
        q_min_val = q.min()
        q_max_val = q.max()
        
        # Infeasible Q-values
        infeasible_q_mean = infeasible_q.mean()
        infeasible_q_std = infeasible_q.std()
        infeasible_q_min = infeasible_q.min()
        infeasible_q_max = infeasible_q.max()
        
        # Violations and gaps
        violations = (infeasible_q < q_min).mean()  # Ratio of Q_inf < Q_min
        infeasible_q_clamped_mean = jnp.maximum(infeasible_q_mean, q_min)
        q_gap_raw = q_mean - infeasible_q_mean  # Raw gap (can be inflated)
        q_gap_true = q_mean - infeasible_q_clamped_mean  # Corrected gap
        
        # PA effectiveness
        pa_active_ratio = (q_excess > 0).mean()  # Ratio where penalty is active

        info = {
            'critic_loss': total_loss,
            'td_loss': td_loss,
            'pa_loss': pa_loss,
            
            # ID Q statistics
            'q_mean': q_mean,
            'q_std': q_std,
            'q_min': q_min_val,
            'q_max': q_max_val,
            
            # Infeasible Q statistics
            'infeasible_q_mean': infeasible_q_mean,
            'infeasible_q_std': infeasible_q_std,
            'infeasible_q_min': infeasible_q_min,
            'infeasible_q_max': infeasible_q_max,
            
            # Gaps and violations
            'q_gap_raw': q_gap_raw,
            'q_gap_true': q_gap_true,
            'q_min_violations': violations,
            'pa_active_ratio': pa_active_ratio,
            
            # Target Q
            'target_q_mean': target_q.mean(),
            'target_q_std': target_q.std(),
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
        """
        # L = self.config['L_infeasible']
        L=1000
        
        # Sample uniformly in [-1, 1]
        actions = jax.random.uniform(
            rng,
            shape=(batch_size, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
        
        # Map to infeasible region
        infeasible_actions = jnp.where(
            actions < 0,
            (actions - 1) * L,  # Negative side: [-2L, -L]
            (actions + 1) * L,  # Positive side: [L, 2L]
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
        """Create a new PARS agent."""
        # Compute Q_min
        min_reward = config.get('min_reward', 0.0)
        q_min = config['reward_scale'] * min_reward / (1 - config['discount'])
        config['q_min'] = q_min
        
        # Force layer normalization
        config['layer_norm'] = True
        
        print(f"\n{'='*60}")
        print(f"Creating PARS Agent")
        print(f"{'='*60}")
        print(f"Reward scale: {config['reward_scale']}")
        print(f"Alpha (PA weight): {config['alpha']}")
        print(f"Beta (BC weight): {config['beta']}")
        print(f"Q_min: {q_min:.3f}")
        # print(f"L_infeasible: {config['L_infeasible']}")
        print(f"Layer Norm: {config['layer_norm']}")
        print(f"One-sided PA: Enabled (only penalize Q > Q_min)")
        print(f"{'='*60}\n")
        
        return super().create(seed, ex_observations, ex_actions, config)


def get_config() -> ml_collections.ConfigDict:
    """Get default PARS configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='pars',
            
            # Learning
            lr=3e-4,
            batch_size=256,
            discount=0.995,  # Higher for AntMaze
            tau=0.005,
            
            # PARS hyperparameters
            reward_scale=100.0,  # Reward scaling factor
            alpha=0.001,  # PA loss weight
            beta=0.01,  # BC loss weight
            L_infeasible=1000.0,  # Distance to infeasible region
            min_reward=0.0,  # Minimum reward
            
            # Networks
            actor_hidden_dims=(256, 256, 256),
            critic_hidden_dims=(256, 256, 256),
            layer_norm=True,
            actor_layer_norm=False,
            
            # Actor
            tanh_squash=True,
            actor_fc_scale=1.0,
            
            # TD3 specifics
            actor_update_freq=2,
            target_noise=0.2,
            noise_clip=0.5,
            exploration_noise=0.1,
            
            # Encoder
            encoder=None,
        )
    )
    return config