import gym
import jsbsim_gym.jsbsim_gym # This line makes sure the environment is registered
from os import path
# import imageio as iio
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO
from time import time
class Zaman(object):

    def __init__(self):
        self.sonZaman = time()

    def zamanDurumu(self):
        if time() > self.sonZaman + (1 / 10):
            self.sonZaman = time()
            return True
        return False

class Main:

    def __init__(self):
        self.policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
        self.env = gym.make("JSBSim-v0")
        self.obs = self.env.reset()
        self.done = False
        self.step = 0
        self.calculateTime = Zaman()
        # self.model = SAC.load("models/model_sac", self.env)
        self.model = PPO.load("models/model_ppo1", self.env)

    def run(self, option):
        while not self.done:
            # if self.calculateTime.zamanDurumu():
                render_data = self.env.render(mode='rgb_array')
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, reward, self.done, _ = self.env.step(action)
                print(f"Step: {self.step} | Reward: {reward} | Done: {self.done}", end="\n")
                self.step += 1
        self.env.close()
        return True

if __name__ == "__main__":
    c = 0
    starter = Main()
    # starter.run(True)
    quitx = True
    while quitx:
        quitx = starter.run(quitx)
        if quitx:
            c += 1
            starter = Main()
            starter.run(quitx)
            if c > 9:
                quitx = False
                break