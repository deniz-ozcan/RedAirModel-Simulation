import jsbsim_gym.jsbsim_gym
import gym #0.21.0
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, DQN #1.6.2
import random
import os
import numpy as np
import torch
class CustomJSBSimEnv(gym.GoalEnv):
    def __init__(self):
        super(CustomJSBSimEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # Örnek bir eylem uzayı
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Örnek bir gözlem uzayı
        self.position = 0
        self.target_position = 10

    def reset(self):
        self.position = 0
        return np.array([self.position], dtype=np.float32)

    def step(self, action):
        if action == 0:  # Sol
            self.position -= 1
        elif action == 1:  # Sağ
            self.position += 1
        # Hedefe ulaşıldığında ödül ver
        reward = 0
        if self.position == self.target_position:
            reward = 1
        # Durum, ödül, bitiş, bilgi
        return np.array([self.position], dtype=np.float32), reward, self.position == self.target_position, {}

    #render with pygame and JSBSim
    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            np.array([self.position], dtype=np.float32)

# env = gym.make("JSBSim-v0")
# Çevreyi oluştur
env = CustomJSBSimEnv()
env.reset()
log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')
policy_kwargs = dict(features_extractor_class = JSBSimFeatureExtractor)
try:
    curr_state = env.reset()
    model = DQN("MlpPolicy", env, learning_rate=0.001,policy_kwargs=policy_kwargs, verbose=1, device='cuda')
    model.learn(total_timesteps=10000)
finally:
    model.save("models/jsbsim_dqn_model")
    model.save_replay_buffer("models/jsbsim_dqn_model2")

model = DQN.load("models/jsbsim_dqn_model")
# JSBSim çevresiyle etkileşime giren bir işlev
def simulate_with_model(model, env):
    obs = env.reset()
    done = False
    step = 0
    while not done:
        render_data = env.render(mode='rgb_array')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(f"Step: {step} | Reward: {reward} | Done: {done}")
        step += 1
    # Simülasyon sona erdiğinde çevreyi sıfırla
    env.reset()

# Modeli JSBSim simülasyonu ile kullan
simulate_with_model(model, env)