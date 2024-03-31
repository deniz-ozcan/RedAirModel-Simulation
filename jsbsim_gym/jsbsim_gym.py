from jsbsim import FGFDMExec
from gym import Env, Wrapper, register
from gym.spaces import Box, Dict
from .visualization.rendering import Viewer, load_mesh, RenderObject, Grid
from .visualization.quaternion import Quaternion
from datetime import datetime
from math import degrees as deg
import numpy as np

RADIUS = 6.3781e6

class JSBSimEnv(Env):
    def __init__(self, root='.'):
        super().__init__()
        self.dateObj = datetime.now()
        self.fileName = f"./Results/result_{self.dateObj.strftime('%Y%m%d%H%M')}.acmi"
        with open(self.fileName, 'w', encoding = 'utf-8') as f:
            f.write(f"""FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime={self.dateObj.strftime('%Y-%m-%dT%H:%M:%SZ')}\n0,ReferenceLongitude=0.0\n0,ReferenceLatitude=0.0""")

        self.action_space = Box(np.array([-1, -1, -1, 0]), 1, (4,))
        self.observation_space = Dict({
            "position_lat_gc_rad": Box(low = -1.57, high = 1.57, shape = (1, ), dtype = np.float32),
            "position_long_gc_rad": Box(low = -3.14, high = 3.14, shape = (1, ), dtype = np.float32),
            "position_h_sl_meters": Box(low = 5000, high = 10000, shape = (1,), dtype = np.float32),
            "aero_alpha_rad": Box(low = -0.2618, high = 0.6109, shape = (1,), dtype = np.float32),
            "aero_beta_rad": Box(low = -0.1745, high = 0.2618, shape = (1,), dtype = np.float32),
            "velocities_mach": Box(low = 0, high = 2.05, shape = (1,), dtype = np. float32),
            "velocities_p_rad_sec": Box(low = -0.52, high = 0.52, shape = (1,), dtype = np.float32),
            "velocities_q_rad_sec": Box(low = -0.44, high = 0.44, shape = (1,), dtype = np.float32),
            "velocities_r_rad_sec": Box(low = -0.32, high = 0.32, shape = (1,), dtype = np.float32),
            "attitude_phi_rad": Box(low = -0.2618, high = 0.2618, shape = (1,), dtype = np.float32),
            "attitude_theta_rad": Box(low = -0.2618, high = 0.2618, shape = (1,), dtype = np.float32),
            "attitude_psi_rad": Box(low = -3.1416, high = 3.1416, shape = (1,), dtype = np.float32),
            "goal_x": Box(low = -0.5, high = 0.5, shape = (1, ), dtype = np.float32),
            "goal_y": Box(low = -0.5, high = 0.5, shape = (1, ), dtype = np.float32),
            "goal_z": Box(low = 5000, high = 10000, shape = (1,), dtype = np.float32),
        })

        # Initialize JSBSim
        self.simulation = FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        self.simulation.load_model('f16') 
        self.setInitialConditions()
        self.simulation.run_ic()

        self.down_sample = 4
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

    def setInitialConditions(self):
        print("kaç kere çalıştı")
        rand = np.random.random()
        randhsl = np.random.randint(5000, 10000)
        randlat = np.random.uniform(35.9025, 42.0268)
        randlong = np.random.uniform(25.9090, 44.5742)
        randdeg = np.random.uniform(0, 360)
        
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.)
        self.simulation.set_property_value('ic/h-sl-ft', randhsl) # farklı bir irtifada başlat
        self.simulation.set_property_value('ic/psi-true-deg', round(rand * 100, 4)) # farklı bir yönle başlat
        self.simulation.set_property_value('ic/long-gc-deg', randlong) # farklı bir boylamda başlat
        self.simulation.set_property_value('ic/lat-geod-deg', randlat) # farklı bir enlemde başlat

    def step(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

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
        obs = self.getStates()
        longitude = self.simulation.get_property_value("position/long-gc-deg")
        latitude = self.simulation.get_property_value("position/lat-geod-deg")
        altitude = self.simulation.get_property_value('position/h-sl-meters')
        theta_rad = self.simulation.get_property_value("attitude/roll-rad")
        phi_rad = self.simulation.get_property_value("attitude/pitch-rad")
        psi_deg = self.simulation.get_property_value("attitude/psi-deg")
        date = self.dateObj.strftime('%Y-%m-%dT%H:%M:%SZ').replace("-", "")[0:8]
        rn = f"F{date}" 
        rn2 = f"E{date}" 
        goal_x = obs["goal_x"]
        goal_y = obs["goal_y"]
        goal_z = obs["goal_z"]
        with open(self.fileName, 'a+', encoding = 'utf-8') as f:
            f.write(f"""\n#{round((datetime.now() - self.dateObj).total_seconds(), 2)}\n{rn},T={longitude}|{latitude}|{altitude}|{deg(theta_rad)}|{deg(phi_rad)}|{psi_deg},Name=F-16C-52,Type=Air+FixedWing,Color=Yellow\n{rn2},T={deg(goal_x[0])}|{deg(goal_y[0])}|{goal_z[0]},Name=Target,Type=Air+FixedWing,Color=Red""")

        if np.sqrt((latitude - deg(goal_x)) ** 2 + (longitude - deg(goal_y)) ** 2) < self.dg and abs(altitude - goal_z) < self.dg:
            reward = 10000
            done = True

        if altitude < 10:
            reward = -10
            done = True

        return obs, reward, done, {} 

    def getStates(self):
        return {
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
        # obs = self.getStates()
        # self.calculate_roll_angle_error()
        # done = False
        # reward_distance = 0
        # displacement = (obs["position_lat_gc_rad"] - obs["goal_x"],
        #                 obs["position_long_gc_rad"] - obs["goal_y"],
        #                 obs["position_h_sl_meters"] - obs["goal_z"])
        

    def reset(self, seed = None):
        obs = self.getStates()
        self.setInitialConditions()
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1)
        rng = np.random.default_rng(seed)
        self.goal[:2] = deg(obs["position_lat_gc_rad"]), deg(obs["position_long_gc_rad"])
        self.goal[:2] *= rng.random() * 10 + 10 # distance
        self.goal[2] = rng.random() * 5000 + 5000 # altitude
        return self.getStates()

    def render(self, mode = 'human'):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1600, 900)
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
        obs = self.getStates()
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

class PositionReward(Wrapper):

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

def wrapJsbSim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)

register(id="JSBSim-v0", entry_point = wrapJsbSim, max_episode_steps=1600)

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