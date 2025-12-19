"""
TD3+BC Implementation (Base for PARS)
Based on ReBRAC code structure
source from https://github.com/seohongpark/fql/blob/master/agents/rebrac.py
"""

import copy
from functools import partial
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class TD3BCAgent(flax.struct.PyTreeNode):
    """TD3+BC agent implementation.
    
    This is a simplified version of ReBRAC without separate actor/critic penalization.
    Serves as the baseline for PARS algorithm.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch: Dict, grad_params: Any, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        """Compute the TD3 critic loss (without ReBRAC's critic penalization).
        
        Args:
            batch: Batch of transitions
            grad_params: Parameters to compute gradients for
            rng: Random key
            
        Returns:
            Tuple of (loss, info_dict)
        """
        rng, sample_rng = jax.random.split(rng)
        
        # Sample next actions from target policy with noise (TD3 style)
        next_dist = self.network.select('target_actor')(batch['next_observations'])
        next_actions = next_dist.mode()
        
        # Add clipped noise for smoothing
        noise = jnp.clip(
            jax.random.normal(sample_rng, next_actions.shape) * self.config['target_noise'],
            -self.config['noise_clip'],
            self.config['noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        # Compute target Q-values (min of ensemble)
        next_qs = self.network.select('target_critic')(
            batch['next_observations'], 
            actions=next_actions
        )
        next_q = next_qs.min(axis=0)

        # TD target (standard Bellman backup)
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        target_q = jax.lax.stop_gradient(target_q)

        # Current Q-values
        q = self.network.select('critic')(
            batch['observations'], 
            actions=batch['actions'], 
            params=grad_params
        )
        
        # MSE loss
        critic_loss = jnp.square(q - target_q).mean()

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
        }

        return critic_loss, info

    def actor_loss(self, batch: Dict, grad_params: Any, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict]:
        """Compute the TD3+BC actor loss.
        
        L_actor = -Q(s, π(s)) + β * ||π(s) - a||²
        
        Args:
            batch: Batch of transitions
            grad_params: Parameters to compute gradients for
            rng: Random key
            
        Returns:
            Tuple of (loss, info_dict)
        """
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        actions = dist.mode()

        # Q-value loss (policy gradient)
        qs = self.network.select('critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0)
        
        # Normalize Q values for scale invariance (from ReBRAC)
        lam = jax.lax.stop_gradient(1 / (jnp.abs(q).mean() + 1e-6))
        actor_loss = -(lam * q).mean()

        # Behavior cloning loss
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)
        bc_loss = self.config['beta'] * mse.mean()

        total_loss = actor_loss + bc_loss

        # Action statistics
        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev()

        info = {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'total_actor_loss': total_loss,
            'action_std': action_std.mean(),
            'action_mse': mse.mean(),
            'q_for_policy': q.mean(),
        }

        return total_loss, info

    @partial(jax.jit, static_argnames=('update_actor',))
    def total_loss(
        self, 
        batch: Dict, 
        grad_params: Any, 
        update_actor: bool = True, 
        rng: jax.random.PRNGKey = None
    ) -> Tuple[jnp.ndarray, Dict]:
        """Compute the total loss (critic + actor).
        
        Args:
            batch: Batch of transitions
            grad_params: Parameters to compute gradients for
            update_actor: Whether to update actor (delayed update)
            rng: Random key
            
        Returns:
            Tuple of (total_loss, info_dict)
        """
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # Critic loss
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        # Actor loss (with delayed update)
        if update_actor:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            actor_loss = 0.0

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network: TrainState, module_name: str):
        """Soft update of target network (Polyak averaging).
        
        Args:
            network: Network train state
            module_name: Name of the module to update ('critic' or 'actor')
        """
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @partial(jax.jit, static_argnames=('update_actor',))
    def update(self, batch: Dict, update_actor: bool = True) -> Tuple['TD3BCAgent', Dict]:
        """Update the agent.
        
        Args:
            batch: Batch of transitions
            update_actor: Whether to update actor (TD3 delayed update)
            
        Returns:
            Tuple of (new_agent, info_dict)
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, update_actor, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        
        if update_actor:
            # Update target networks (only when actor is updated)
            self.target_update(new_network, 'critic')
            self.target_update(new_network, 'actor')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations: jnp.ndarray,
        seed: jax.random.PRNGKey = None,
        temperature: float = 1.0,
        eval_mode: bool = False,
    ) -> jnp.ndarray:
        """Sample actions from the policy.
        
        Args:
            observations: Observations
            seed: Random seed for noise
            temperature: Temperature for exploration noise
            eval_mode: If True, return deterministic actions
            
        Returns:
            Actions
        """
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.mode()
        
        if not eval_mode and seed is not None:
            # Add exploration noise
            noise = jnp.clip(
                jax.random.normal(seed, actions.shape) * self.config['exploration_noise'] * temperature,
                -self.config['noise_clip'],
                self.config['noise_clip'],
            )
            actions = jnp.clip(actions + noise, -1, 1)
        
        return actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: ml_collections.ConfigDict,
    ) -> 'TD3BCAgent':
        """Create a new TD3+BC agent.

        Args:
            seed: Random seed
            ex_observations: Example observations for initialization
            ex_actions: Example actions for initialization
            config: Configuration dictionary

        Returns:
            New TD3BCAgent instance
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        # Define encoders (for visual observations)
        encoders = dict()
        if config.get('encoder') is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define critic network (ensemble of Q-functions)
        critic_def = Value(
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,  # TD3 uses 2 Q-functions
            encoder=encoders.get('critic'),
        )
        
        # Define actor network
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
            encoder=encoders.get('actor'),
        )

        # Create network dictionary
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
            target_actor=(copy.deepcopy(actor_def), (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize target networks
        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config() -> ml_collections.ConfigDict:
    """Get default TD3+BC configuration."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='td3_bc',
            # Learning
            lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,  # Target network update rate
            
            # Networks
            actor_hidden_dims=(256, 256, 256),
            critic_hidden_dims=(256, 256, 256),
            layer_norm=False,  # Critic layer norm (PARS will set to True)
            actor_layer_norm=False,
            
            # Actor
            tanh_squash=True,
            actor_fc_scale=1.0,
            beta=0.01,  # BC coefficient (α in TD3+BC paper)
            
            # TD3 specifics
            actor_update_freq=2,  # Delayed policy update
            target_noise=0.2,  # Target policy smoothing noise
            noise_clip=0.5,  # Target policy smoothing noise clip
            exploration_noise=0.1,  # Exploration noise for online
            
            # Encoder (for visual observations)
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config