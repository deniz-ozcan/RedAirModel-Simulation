import jsbsim_gym.jsbsim_gym 
import gym
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO, DQN
import numpy as np
from os import path

policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
env = gym.make("JSBSim-v0")
env.reset()

# log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')
try:
    curr_state = env.reset()
    model = SAC('MultiInputPolicy', env, verbose=1, policy_kwargs=policy_kwargs,  gradient_steps = -1, device='cuda')# tensorboard_log = log_path,
    model.learn(1500000)
except Exception as e:
    gym.logger.error(f"{e}")
finally:
    model.save("models/jsbsim_sac3")
    model.save_replay_buffer("models/jsbsim_sac_buffer3")