"""
Unified Replay Buffer for Offline and Online RL
Based on ReBRAC's Dataset class with minimal modifications for PARS
source from https://github.com/seohongpark/fql/blob/master/utils/datasets.py
"""

from functools import partial
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image."""
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """
    Base Dataset class (from ReBRAC).
    Immutable for offline data.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields."""
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

        # Compute terminal and initial locations
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        if len(self.terminal_locs) > 0:
            self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        else:
            self.initial_locs = np.array([0])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        
        if self.frame_stack is not None:
            # Stack frames
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []
            next_obs = []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        
        if self.p_aug is not None:
            # Apply random-crop image augmentation
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """
    Unified Replay Buffer for Offline-to-Online RL.
    
    Extends Dataset to support:
    1. Loading D4RL datasets (offline)
    2. Adding new transitions (online)
    3. Mixed sampling (offline + online)
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from example transition."""
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_d4rl(cls, dataset: Dict, max_size: Optional[int] = None):
        """
        Create replay buffer from D4RL dataset.
        
        Args:
            dataset: D4RL dataset dictionary
            max_size: Maximum buffer size (if None, use dataset size)
                     If larger than dataset, allows adding online transitions
        
        Returns:
            ReplayBuffer instance
        """
        n_offline = len(dataset['observations'])
        
        # Determine buffer size
        if max_size is None:
            max_size = n_offline
        elif max_size < n_offline:
            raise ValueError(f"max_size ({max_size}) < dataset size ({n_offline})")
        
        state_dim = dataset['observations'].shape[1:]
        action_dim = dataset['actions'].shape[1:]
        
        # Create buffer structure
        buffer_dict = {
            'observations': np.zeros((max_size, *state_dim), dtype=np.float32),
            'actions': np.zeros((max_size, *action_dim), dtype=np.float32),
            'next_observations': np.zeros((max_size, *state_dim), dtype=np.float32),
            'rewards': np.zeros((max_size, 1), dtype=np.float32),
            'terminals': np.zeros((max_size, 1), dtype=np.float32),
        }
        
        # Fill with offline data
        buffer_dict['observations'][:n_offline] = dataset['observations']
        buffer_dict['actions'][:n_offline] = dataset['actions']
        
        # Handle next_observations
        if 'next_observations' in dataset:
            buffer_dict['next_observations'][:n_offline] = dataset['next_observations']
        else:
            buffer_dict['next_observations'][:n_offline-1] = dataset['observations'][1:]
            buffer_dict['next_observations'][n_offline-1] = dataset['observations'][-1]
        
        # Handle rewards and terminals
        rewards = dataset['rewards']
        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        buffer_dict['rewards'][:n_offline] = rewards
        
        terminals = dataset['terminals']
        if terminals.ndim == 1:
            terminals = terminals.reshape(-1, 1)
        buffer_dict['terminals'][:n_offline] = terminals
        
        # Add masks for PARS/TD3+BC
        buffer_dict['masks'] = 1.0 - buffer_dict['terminals']
        
        # Create buffer
        buffer = cls(buffer_dict)
        buffer.max_size = max_size
        buffer.size = n_offline
        buffer.pointer = n_offline
        buffer.n_offline = n_offline  # Track offline data boundary
        
        # Statistics
        print("\n" + "="*60)
        print("Replay Buffer Created from D4RL")
        print("="*60)
        print(f"Offline transitions: {n_offline:,}")
        print(f"Buffer capacity: {max_size:,}")
        print(f"Space for online: {max_size - n_offline:,}")
        print(f"Obs shape: {state_dim}")
        print(f"Action shape: {action_dim}")
        print(f"Reward range: [{buffer_dict['rewards'][:n_offline].min():.3f}, "
              f"{buffer_dict['rewards'][:n_offline].max():.3f}]")
        print(f"Terminal ratio: {buffer_dict['terminals'][:n_offline].mean():.3f}")
        print(f"Trajectories: {len(buffer.terminal_locs)}")
        print("="*60 + "\n")
        
        return buffer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0
        self.n_offline = 0  # Number of offline transitions

    def add_transition(self, transition: Dict):
        """
        Add a single transition to the replay buffer.
        
        Args:
            transition: Dict with keys: observations, actions, next_observations,
                       rewards, terminals, (masks)
        """
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        # Update trajectory boundaries
        if transition.get('terminals', 0) > 0:
            self.terminal_locs = np.nonzero(self._dict['terminals'][:self.size].flatten() > 0)[0]
            if len(self.terminal_locs) > 0:
                self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def add_transition_tuple(
        self,
        observation,
        action,
        next_observation,
        reward,
        terminal,
    ):
        """
        Add transition from tuple (convenience method).
        
        Args:
            observation: Current state
            action: Action taken
            next_observation: Next state
            reward: Reward received
            terminal: Whether episode terminated
        """
        transition = {
            'observations': np.array(observation),
            'actions': np.array(action),
            'next_observations': np.array(next_observation),
            'rewards': np.array([[reward]], dtype=np.float32),
            'terminals': np.array([[float(terminal)]], dtype=np.float32),
            'masks': np.array([[1.0 - float(terminal)]], dtype=np.float32),
        }
        self.add_transition(transition)

    def sample_mixed(
        self,
        batch_size: int,
        offline_ratio: float = 0.5,
    ) -> Dict:
        """
        Sample batch with mixing ratio from offline and online data.
        
        Args:
            batch_size: Total batch size
            offline_ratio: Ratio of offline samples (0.0 to 1.0)
        
        Returns:
            Mixed batch dictionary
        """
        n_offline_samples = int(batch_size * offline_ratio)
        n_online_samples = batch_size - n_offline_samples
        
        # Sample from offline
        offline_idxs = np.random.randint(0, self.n_offline, size=n_offline_samples)
        
        # Sample from online (if any)
        if self.size > self.n_offline and n_online_samples > 0:
            online_idxs = np.random.randint(self.n_offline, self.size, size=n_online_samples)
            idxs = np.concatenate([offline_idxs, online_idxs])
        else:
            # No online data yet, sample only from offline
            idxs = offline_idxs
        
        # Shuffle
        np.random.shuffle(idxs)
        
        return self.sample(batch_size=len(idxs), idxs=idxs)

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        stats = {
            'total_size': self.size,
            'offline_size': self.n_offline,
            'online_size': max(0, self.size - self.n_offline),
            'capacity': self.max_size,
            'usage': self.size / self.max_size,
        }
        
        if self.size > 0:
            stats.update({
                'reward_mean': float(self._dict['rewards'][:self.size].mean()),
                'reward_std': float(self._dict['rewards'][:self.size].std()),
                'reward_min': float(self._dict['rewards'][:self.size].min()),
                'reward_max': float(self._dict['rewards'][:self.size].max()),
                'terminal_ratio': float(self._dict['terminals'][:self.size].mean()),
                'n_trajectories': len(self.terminal_locs),
            })
        
        return stats

    def clear_online(self):
        """Clear only online data, keep offline data."""
        if self.size > self.n_offline:
            self.size = self.n_offline
            self.pointer = self.n_offline
            
            # Recompute trajectory boundaries
            self.terminal_locs = np.nonzero(
                self._dict['terminals'][:self.size].flatten() > 0
            )[0]
            if len(self.terminal_locs) > 0:
                self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def clear(self):
        """Clear the entire replay buffer."""
        self.size = self.pointer = self.n_offline = 0
        self.terminal_locs = np.array([], dtype=np.int64)
        self.initial_locs = np.array([0], dtype=np.int64)