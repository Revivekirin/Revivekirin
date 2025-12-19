import gym
import d4rl
ids = list(gym.envs.registry.env_specs.keys())
print([k for k in ids if "antmaze" in k])
from algorithms.pars import PARSAgent, get_config
from utils.datasets import ReplayBuffer

# Environment setup
env = gym.make("antmaze-umaze-v0")      
# env = gym.make("antmaze-umaze-diverse-v0") 
print("made:", env)
dataset = d4rl.qlearning_dataset(env)
print("[DEBUG] dataset :", dataset[0])
# Replay buffer
buffer = ReplayBuffer(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    max_size=len(dataset['observations']),
)
buffer.load_d4rl_dataset(dataset)

# PARS agent
config = get_config()
config['reward_scale'] = 1000  # Experiment variable
config['alpha'] = 0.001  # Experiment variable

agent = PARSAgent.create(
    seed=0,
    ex_observations=dataset['observations'][:1],
    ex_actions=dataset['actions'][:1],
    config=config,
)

# Training loop
for step in range(1_000_000):
    batch = buffer.sample(256)
    update_actor = (step % config['actor_update_freq'] == 0)
    agent, info = agent.update(batch, update_actor=update_actor)
    
    if step % 1000 == 0:
        print(f"Step {step}: Q gap = {info['critic/q_gap']:.3f}")