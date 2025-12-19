import gymnasium as gym
import d4rl
from utils.datasets import ReplayBuffer

# D4RL 데이터셋 로드
env = gym.make('antmaze-umaze-diverse-v2')
dataset = d4rl.qlearning_dataset(env)

# 오프라인 전용 버퍼 (데이터셋 크기만큼)
buffer = ReplayBuffer.create_from_d4rl(dataset, max_size=None)

# 학습 루프
for step in range(1_000_000):
    batch = buffer.sample(batch_size=256)
    agent, info = agent.update(batch)