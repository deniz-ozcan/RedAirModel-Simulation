import jsbsim_gym.jsbsim_gym
import gym
# import imageio as iio
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC


# JSBSimFeatureExtractor'ı oluştururken gözlem alanını kullan
policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
env = gym.make("JSBSim-v0")
model = SAC.load("models/jsbsim_sac", env)

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