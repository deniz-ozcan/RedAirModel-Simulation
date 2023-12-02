import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
import gym #0.21.0
from os import path
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC #1.6.2
import numpy as np
import random
import os
import torch

policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
env = gym.make("JSBSim-v0")
env.reset()


log_path = path.join(path.abspath(path.dirname(__file__)), 'logs')
try:
    curr_state = env.reset()
    model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_path, gradient_steps=-1, device='cuda')
    model.learn(3000000)
finally:
    model.save("models/jsbsim_sac2")
    model.save_replay_buffer("models/jsbsim_sac_buffer2")