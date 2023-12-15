import gym
from stable_baselines3 import DQN
from jsbsim_gym.features import JSBSimFeatureExtractor
import jsbsim_gym.jsbsim_gym

gym.register(
    id='JSBSim-v0',
    entry_point='jsbsim_gym.jsbsim_gym:JSBSimEnv',
)
policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
env = gym.make("JSBSim-v0")
model = DQN.load("models/DQNModel", env)
obs = env.reset()
done = False
step = 0

while not done:
    render_data = env.render(mode='rgb_array')
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Step: {step} | Reward: {reward} | Done: {done}")
    step += 1

env.close()