import random
import jsbsim
import gym
import numpy as np
from .visualization.rendering import Viewer, load_mesh, RenderObject, Grid
from .visualization.quaternion import Quaternion
from gym import spaces
import torch as th
from math import degrees as deg
import datetime
"""
hedefin de enlem, boylam ve irtifa bilgileri olacak
"""

STATE_FORMAT = [
    "position/lat-gc-rad", "position/long-gc-rad", "position/h-sl-meters", 
    "velocities/mach", "aero/alpha-rad", "aero/beta-rad", 
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad"]

RADIUS = 6.3781e6 # Radius of the earth
class JSBSimEnv(gym.Env):
    def __init__(self, root='.'):
        super().__init__()
        with open("./Results/F-14A (Maverick&Goose) [Blue] .csv", 'w', encoding = 'utf-8') as f:
            f.write(f"Time, Longitude, Latitude, Altitude, Roll (deg), Pitch (deg), Yaw (deg)\n")
        self.action_space = spaces.Box(np.array([-1, -1, -1, 0]), 1, (4,))
        self.observation_space = spaces.Dict({
            "position_lat_gc_rad": spaces.Box(low = float('-inf'), high = float('inf'), shape = (1, ), dtype = np.float32),
            "position_long_gc_rad": spaces.Box(low = float('-inf'), high = float('inf'), shape = (1, ), dtype = np.float32),
            "position_h_sl_meters": spaces.Box(low = 0, high = 15000, shape = (1,), dtype = np.float32),
            "aero_alpha_rad": spaces.Box(low = -0.2618, high = 0.6109, shape = (1,), dtype = np.float32),
            "aero_beta_rad": spaces.Box(low = -0.1745, high = 0.2618, shape = (1,), dtype = np.float32),
            "velocities_mach": spaces.Box(low = 0, high = 2.05, shape = (1,), dtype = np. float32),
            "velocities_p_rad_sec": spaces.Box(low = -0.52, high = 0.52, shape = (1,), dtype = np.float32),
            "velocities_q_rad_sec": spaces.Box(low = -0.44, high = 0.44, shape = (1,), dtype = np.float32),
            "velocities_r_rad_sec": spaces.Box(low = -0.32, high = 0.32, shape = (1,), dtype = np.float32),
            "attitude_phi_rad": spaces.Box(low = -0.2618, high = 0.2618, shape = (1,), dtype = np.float32),
            "attitude_theta_rad": spaces.Box(low = -0.2618, high = 0.2618, shape = (1,), dtype = np.float32),
            "attitude_psi_rad": spaces.Box(low = -3.1416, high = 3.1416, shape = (1,), dtype = np.float32),
            "goal_x": spaces.Box(low = float('-inf'), high = float('inf'), shape = (1, ), dtype = np.float32),
            "goal_y": spaces.Box(low = float('-inf'), high = float('inf'), shape = (1, ), dtype = np.float32),
            "goal_z": spaces.Box(low = 0, high = 15000, shape = (1,), dtype = np.float32),
        })

        # Initialize JSBSim / JSBSim'i başlat
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        # Load F-16 model and set initial conditions / F-16 modelini yükleyin ve başlangıç koşullarını ayarlayın
        self.simulation.load_model('f16') 
        self._set_initial_conditions()
        self.simulation.run_ic()

        self.down_sample = 4
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

    # Set engines running, forward velocity, and altitude / Motorları çalıştır, ileri hız ve irtifa ayarla
    def _set_initial_conditions(self):
        rand = random.random()
        range10 = random.randint(1, 10)
        randdeg = random.uniform(0, 360)
        self.simulation.set_property_value('propulsion/set-running', -1) # motorları daha yavaş çalıştır
        self.simulation.set_property_value('ic/u-fps', 900.)
        # self.simulation.set_property_value('ic/h-sl-ft', 5000)
        
        # self.simulation.set_property_value('ic/u-fps', (range10)*(rand * 100))# farklı hızda başlat
        self.simulation.set_property_value('ic/h-sl-ft', (range10 + 5)*round(rand * 1000)) # farklı bir irtifada başlat
        
        self.simulation.set_property_value('ic/psi-true-deg', round(rand * 100, 4))# farklı bir yönle başlat
        self.simulation.set_property_value('ic/long-gc-deg', -round(randdeg, 4)) # farklı bir boylamda başlat
        self.simulation.set_property_value('ic/lat-gc-deg', round(randdeg, 4)) # farklı bir enlemde başlat
        self.simulation.set_property_value('gear/gear-cmd-norm', -1) # Iniş takımı kontrol komutunu ayarla
        self.simulation.set_property_value('gear/gear-pos-norm', -1) # Iniş takımı pozisyonunu ayarla

        # self.simulation.set_property_value('fcs/aileron-cmd-norm', round(rand, 1)) # Aileron kontrol komutunu ayarla
        # self.simulation.set_property_value('fcs/elevator-cmd-norm', round(rand, 1)) # Elevatör kontrol komutunu ayarla
        # self.simulation.set_property_value('fcs/rudder-cmd-norm', round(rand, 1)) # Rudder kontrol komutunu ayarla
        # self.simulation.set_property_value('fcs/throttle-cmd-norm', round(rand, 1)) # Gaz kontrol komutunu ayarla

    def step(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        # Pass control inputs to JSBSim / Kontrol girişlerini JSBSim'e aktarın
        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)

        # We take multiple steps of the simulation per step of the environment / Ortamın bir adımı için simülasyonun birden fazla adımını alıyoruz
        for _ in range(self.down_sample):
            # Freeze fuel consumption / Yakıt tüketimini dondurun
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            # Set gear up / Dişlileri yukarı kaldır
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)
            self.simulation.run()

        reward, done = 0, False
        obs = self._get_state()
        pos_x_deg = self.simulation.get_property_value('position/long-gc-deg')
        pos_y_deg = self.simulation.get_property_value('position/lat-geod-deg')
        pos_z = self.simulation.get_property_value('position/h-sl-meters')
        theta_rad = self.simulation.get_property_value("attitude/roll-rad")
        phi_rad = self.simulation.get_property_value("attitude/pitch-rad")
        psi_rad = self.simulation.get_property_value("attitude/psi-deg")

        with open("./Results/F-14A (Maverick&Goose) [Blue] .csv", 'a+', encoding = 'utf-8') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}, {pos_x_deg}, {pos_y_deg}, {pos_z}, {deg(theta_rad)}, {deg(phi_rad)}, {psi_rad}\n")

        current_x = obs["position_lat_gc_rad"]
        current_y = obs["position_long_gc_rad"]
        current_z = obs["position_h_sl_meters"]
        goal_x = obs["goal_x"]
        goal_y = obs["goal_y"]
        goal_z = obs["goal_z"]

        if np.sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2) < self.dg and abs(current_z - goal_z) < self.dg:
            print("reached")
            reward = 10000
            done = True
        
        if obs["position_h_sl_meters"] < 10:
            reward = -10
            done = True
        
        return obs, reward, done, {} 

    def _get_state(self):
        obs = {
            "position_lat_gc_rad": np.array([self.simulation.get_property_value("position/lat-gc-rad") * RADIUS]),
            "position_long_gc_rad": np.array([self.simulation.get_property_value("position/long-gc-rad") * RADIUS]),
            "position_h_sl_meters": np.array([self.simulation.get_property_value("position/h-sl-meters")]),
            "velocities_mach": np.array([self.simulation.get_property_value("velocities/mach")]),
            "aero_alpha_rad": np.array([self.simulation.get_property_value("aero/alpha-rad")]),
            "aero_beta_rad": np.array([self.simulation.get_property_value("aero/beta-rad")]),
            "velocities_p_rad_sec": np.array([self.simulation.get_property_value("velocities/p-rad_sec")]),
            "velocities_q_rad_sec": np.array([self.simulation.get_property_value("velocities/q-rad_sec")]),
            "velocities_r_rad_sec": np.array([self.simulation.get_property_value("velocities/r-rad_sec")]),
            "attitude_phi_rad": np.array([self.simulation.get_property_value("attitude/phi-rad")]),
            "attitude_theta_rad": np.array([self.simulation.get_property_value("attitude/theta-rad")]),
            "attitude_psi_rad": np.array([self.simulation.get_property_value("attitude/psi-rad")]),
            "goal_x": np.array([self.goal[0]]),
            "goal_y": np.array([self.goal[1]]),
            "goal_z": np.array([self.goal[2]])
        }
        # obs = self._get_state()
        # self.calculate_roll_angle_error()
        # done = False
        # reward_distance = 0
        # displacement = (obs["position_lat_gc_rad"] - obs["goal_x"],
        #                 obs["position_long_gc_rad"] - obs["goal_y"],
        #                 obs["position_h_sl_meters"] - obs["goal_z"])
        
        return obs

    def reset(self, seed = None):
        # Rerun initial conditions in JSBSim / JSBSim'de başlangıç koşullarını yeniden çalıştırın
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1)

        # Generate a new goal / Yeni bir hedef oluşturun
        rng = np.random.default_rng(seed)
        distance = rng.random() * 9000 + 1000
        bearing = rng.random() * 2 * np.pi
        altitude = rng.random() * 3000

        self.goal[:2] = np.cos(bearing), np.sin(bearing)
        self.goal[:2] *= distance
        self.goal[2] = altitude

        return self._get_state()

    def render(self, mode = 'human'):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)
            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1 / 30
            self.f16.color = 0, 1, 0
            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.targetF16 = RenderObject(goal_mesh)
            self.targetF16.transform.scale = 1 / 30
            self.targetF16.color = 1, 0, 0
            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(self.targetF16)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))

        # Rough conversion from lat/long to meters / yaklaşık dönüşüm
        obs = self._get_state()
        x = obs["position_lat_gc_rad"] * scale
        y = obs["position_long_gc_rad"] * scale
        z = obs["position_h_sl_meters"] * scale

        self.f16.transform.z = x
        self.f16.transform.x = -y
        self.f16.transform.y = z

        rot = Quaternion.from_euler(obs["attitude_phi_rad"], obs["attitude_theta_rad"], obs["attitude_psi_rad"])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot

        x, y, z = self.goal * scale
        self.targetF16.transform.z = x
        self.targetF16.transform.x = -y
        self.targetF16.transform.y = z

        r = self.f16.transform.position - self.targetF16.transform.position
        rhat = r/np.linalg.norm(r)
        x, y, z = r
        yaw = np.arctan2(-x,-z)
        pitch = np.arctan2(-y, np.sqrt(x*2 + z*2))

        self.viewer.set_view(*(r + self.targetF16.transform.position + rhat + np.array([0, .33, 0])), Quaternion.from_euler(-pitch, yaw, 0, mode=1))
        self.viewer.render()

        if mode == 'rgb_array':
            return self.viewer.get_frame()
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class PositionReward(gym.Wrapper):

    def __init__(self, env, gain):
        super().__init__(env)
        self.gain = gain

    def step(self, action):
        obs, reward, done, info = super().step(action)
        distance = self.getDisplacement(obs)
        reward += self.gain * (self.last_distance - distance)
        self.last_distance = distance
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.last_distance = self.getDisplacement(obs)
        return obs

    def getDisplacement(self, obs):
        displacement = np.concatenate((obs["goal_x"] - obs["position_lat_gc_rad"], obs["goal_y"] - obs["position_long_gc_rad"], obs["goal_z"] - obs["position_h_sl_meters"]))
        return np.linalg.norm(displacement)

# Create entry point to wrapped environment
def wrap_jsbsim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)

gym.register(id="JSBSim-v0", entry_point=wrap_jsbsim, max_episode_steps=1200) # Register the wrapped environment, 1500 1600 iyidir

# Short example script to create and run the environment with constant action for 1 simulation second.
# Ortamı oluşturmak ve sabit eylem için 1 simülasyon saniyesi çalıştırmak için kısa bir örnek komut dosyası.
if __name__ == "__main__":
    from time import sleep
    env = JSBSimEnv()
    env.reset()
    env.render()
    for _ in range(300):
        env.step(np.array([0.05, -0.2, 0, .5]))
        env.render()
        sleep(1 / 30)
    env.close()