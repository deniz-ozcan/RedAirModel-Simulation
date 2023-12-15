import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class SimpleEnv(gym.Env):
    def __init__(self):
        super(SimpleEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # 0: İleri, 1: Geri, 2: Sağa, 3: Sola
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=float)
        self.position = [0, 0]

    def reset(self):
        self.position = [0, 0]
        return self.position

    def step(self, action):
        if action == 0:  # İleri
            self.position[1] += 1
        elif action == 1:  # Geri
            self.position[1] -= 1
        elif action == 2:  # Sağa
            self.position[0] += 1
        elif action == 3:  # Sola
            self.position[0] -= 1
        reward = -abs(self.position[0] + self.position[1])  # Basit bir ödül fonksiyonu (örnek olarak mesafenin negatif mutlak değeri)
        done = abs(self.position[0]) <= 10 and abs(self.position[1]) <= 10 # Hedefe ulaşıldı mı kontrolü (örneğin, belirli bir mesafe içinde)
        return self.position, reward, done, {}

# Wrap your environment with DummyVecEnv for compatibility with VecNormalize
env = SimpleEnv()
env = DummyVecEnv([lambda: env])

# Now you can normalize the environment
env = VecNormalize(env, norm_obs=True, norm_reward=False)
model = DQN("MlpPolicy", env, learning_rate=1e-3, verbose=1)
model.learn(total_timesteps=int(1e5))
model.save("models/DQNModel")