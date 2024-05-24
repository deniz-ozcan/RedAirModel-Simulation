from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC, PPO
from time import time
from gym import make
from os import path
import jsbsim_gym.jsbsim_gym
class Zaman(object):

    def __init__(self):
        self.sonZaman = time()

    def zamanDurumu(self):
        if time() > self.sonZaman + (1 / 68):
            self.sonZaman = time()
            return True
        return False

class Main:

    def __init__(self):
        self.policy_kwargs = dict(features_extractor_class=JSBSimFeatureExtractor)
        self.env = make("JSBSim-v0")
        self.obs = self.env.reset()
        self.done = False
        self.step = 0
        self.calculateTime = Zaman()
        # self.model = SAC.load("models/model_sac", self.env)
        self.model = PPO.load("models/model_ppoy", self.env)

    def run(self):
        while not self.done:
            # if self.calculateTime.zamanDurumu():
                render_data = self.env.render(mode='rgb_array')
                action, _ = self.model.predict(self.obs, deterministic=True)
                self.obs, reward, self.done, _ = self.env.step(action)
                if self.done:
                    print(f"Step: {self.step} | Reward: {reward}", end="\n")
                self.step += 1
        self.env.close()
        return True

if __name__ == "__main__":
    c = 0
    starter = Main()
    quitx = True
    while quitx:
        quitx = starter.run()
        if quitx:
            c += 1
            starter = Main()
            starter.run()
            if c > 9:
                quitx = False
                break