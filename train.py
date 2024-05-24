from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO, DQN
from gym import logger, make
import jsbsim_gym.jsbsim_gym 
policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
env = make("JSBSim-v0")
try:
    curr_state = env.reset()
    # model = SAC('MultiInputPolicy', env, verbose=1, policy_kwargs = policy_kwargs,  gradient_steps = -1, device='cuda')
    model = PPO('MultiInputPolicy', env, verbose=1, policy_kwargs = policy_kwargs, device='cuda')
    model.learn(1500000)
except Exception as e:
    logger.error(f"{e}")
finally:
    # model.save("models/model_sac1")
    model.save("models/model_ppoy")
    # model.save_replay_buffer("models/model_sac1_buffer")