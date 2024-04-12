from jsbsim import FGFDMExec
from gym import Env, Wrapper, register
from gym.spaces import Box, Dict
from .visualization.rendering import Viewer, load_mesh, RenderObject, Grid
from .visualization.quaternion import Quaternion
from datetime import datetime
from math import degrees as deg, radians as rad
import numpy as np
from pyproj import Geod

geodesic = Geod(ellps='WGS84')
class PositionReward(Wrapper):

    def __init__(self, env, gain):
        super().__init__(env)
        self.gain = gain

    def step(self, action):
        obs, reward, done, info = super().step(action)
        dist, alt = self.getDistance(obs)
        reward += self.gain * (self.last_dist - dist)
        reward += self.gain * (self.last_alt - alt)
        self.last_dist, self.last_alt =  dist, alt
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.last_dist, self.last_alt = self.getDistance(obs)
        return obs

    def getDistance(self, obs):
        az, _, dist = geodesic.inv(obs["goal_long_gc_deg"], obs["goal_lat_geod_deg"], obs["pos_long_gc_deg"], obs["pos_lat_geod_deg"])
        return dist[0], abs(obs["goal_h_sl_meters"][0]- obs["pos_h_sl_meters"][0])

    # def getDisplacement(self, obs):
    #     displacement = np.concatenate(( np.deg2rad(obs["goal_lat_geod_deg"]) - np.deg2rad(obs["pos_lat_geod_deg"]), 
    #                                     np.deg2rad(obs["goal_long_gc_deg"]) - np.deg2rad(obs["pos_long_gc_deg"]), 
    #                                     obs["goal_h_sl_meters"] - obs["pos_h_sl_meters"]))
    #     return np.linalg.norm(displacement)

class JSBSimEnv(Env):
    def __init__(self, root='.'):
        super().__init__()
        self.dateObj = datetime.now()
        self.action_space = Box(np.array([-1,-1,-1,0]), 1, (4,))
        self.observation_space = Dict({
                "pos_lat_geod_deg": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "pos_long_gc_deg": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "pos_h_sl_meters": Box(low = 0, high = np.inf, shape = (1,)),
                "aero_alpha_rad": Box(low = -np.pi, high = np.pi, shape = (1,)),
                "aero_beta_rad": Box(low = -np.pi, high = np.pi, shape = (1,)),
                "velocities_mach": Box(low = 0, high = np.inf, shape = (1,)),
                "velocities_p_rad_sec": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "velocities_q_rad_sec": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "velocities_r_rad_sec": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "attitude_phi_rad": Box(low = -np.pi, high = np.pi, shape = (1,)),
                "attitude_theta_rad": Box(low = -np.pi, high = np.pi, shape = (1,)),
                "attitude_psi_rad": Box(low = -np.pi, high = np.pi, shape = (1,)),
                "goal_lat_geod_deg": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "goal_long_gc_deg": Box(low = -np.inf, high = np.inf, shape = (1,)),
                "goal_h_sl_meters": Box(low = 0, high = np.inf, shape = (1,)),
            })


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
        rand = np.random.random()
        randhsl = np.random.randint(1000, 10000)
        randlat, randlong = np.random.uniform(39.25, 39.35), np.random.uniform(32.25, 32.35)
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.)
        self.simulation.set_property_value('ic/h-sl-ft', randhsl) # farklı bir irtifada başlat
        self.simulation.set_property_value('ic/psi-true-deg', round(rand * 100, 4)) # farklı bir yönle başlat
        self.simulation.set_property_value('ic/long-gc-deg', randlong) # farklı bir boylamda başlat
        self.simulation.set_property_value('ic/lat-geod-deg', randlat) # farklı bir enlemde başlat

    def reset(self, seed = None):
        self.setInitialConditions()
        self.simulation.run_ic()
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.goal[:2] = np.random.uniform(39.25, 39.35), np.random.uniform(32.25, 32.35)
        self.goal[2] = 3048
        return self.getStates()

    def step(self, action):
        roll_cmd, pitch_cmd, yaw_cmd, throttle = action

        self.simulation.set_property_value("fcs/aileron-cmd-norm", roll_cmd)
        self.simulation.set_property_value("fcs/elevator-cmd-norm", pitch_cmd)
        self.simulation.set_property_value("fcs/rudder-cmd-norm", yaw_cmd)
        self.simulation.set_property_value("fcs/throttle-cmd-norm", throttle)

        # We take multiple steps of the simulation per step of the environment / Ortamın bir adımı için simülasyonun birden fazla adımını alıyoruz
        for _ in range(self.down_sample):
            # Freeze fuel consumption
            self.simulation.set_property_value("propulsion/tank/contents-lbs", 1000)
            self.simulation.set_property_value("propulsion/tank[1]/contents-lbs", 1000)
            # Set gear up
            self.simulation.set_property_value("gear/gear-cmd-norm", 0.0)
            self.simulation.set_property_value("gear/gear-pos-norm", 0.0)
            self.simulation.run()

        date = self.dateObj.strftime('%Y%m%d%H%M')
        obs = self.getStates()
        roll = self.simulation.get_property_value("attitude/roll-rad")
        pitch = self.simulation.get_property_value("attitude/pitch-rad")
        yaw = self.simulation.get_property_value("attitude/psi-deg")
        long = self.simulation.get_property_value("position/long-gc-deg")
        lat = self.simulation.get_property_value("position/lat-geod-deg")
        alt = self.simulation.get_property_value('position/h-sl-meters')
        goal_x = obs["goal_lat_geod_deg"]
        goal_y = obs["goal_long_gc_deg"]
        goal_z = obs["goal_h_sl_meters"]
        with open(f"./Results/result_{self.dateObj.strftime('%Y%m%d%H%M%S')}.acmi", 'a+', encoding = 'utf-8') as f:
            if f.tell() == 0:f.write(f"""FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime={self.dateObj.strftime('%Y-%m-%dT%H:%M:%SZ')}\n0,ReferenceLongitude=0.0\n0,ReferenceLatitude=0.0""")
            f.write(f"""\n#{round((datetime.now() - self.dateObj).total_seconds(), 2)}\nF{date},T={long}|{lat}|{alt}|{deg(roll)}|{deg(pitch)}|{yaw},Name=F-16C-52,Type=Air+FixedWing,Color=Blue\nE{date},T={goal_y[0]}|{goal_x[0]}|{goal_z[0]},Name=F-16C-52,Type=Air+FixedWing,Color=Red""")
        reward, done = 0, False
        if alt < 10: 
            reward, done = -10, True

        if geodesic.inv(goal_y[0], goal_x[0], long, lat)[2] < self.dg and abs(goal_z[0] - alt) < self.dg: 
            reward, done = 10000, True

        return obs, reward, done, {} 


    def getStates(self):
        return {
            "pos_lat_geod_deg": np.array([self.simulation.get_property_value("position/lat-geod-deg")]),
            "pos_long_gc_deg": np.array([self.simulation.get_property_value("position/long-gc-deg")]),
            "pos_h_sl_meters": np.array([self.simulation.get_property_value("position/h-sl-meters")]),
            "velocities_mach": np.array([self.simulation.get_property_value("velocities/mach")]),
            "aero_alpha_rad": np.array([self.simulation.get_property_value("aero/alpha-rad")]),
            "aero_beta_rad": np.array([self.simulation.get_property_value("aero/beta-rad")]),
            "velocities_p_rad_sec": np.array([self.simulation.get_property_value("velocities/p-rad_sec")]),
            "velocities_q_rad_sec": np.array([self.simulation.get_property_value("velocities/q-rad_sec")]),
            "velocities_r_rad_sec": np.array([self.simulation.get_property_value("velocities/r-rad_sec")]),
            "attitude_phi_rad": np.array([self.simulation.get_property_value("attitude/phi-rad")]),
            "attitude_theta_rad": np.array([self.simulation.get_property_value("attitude/theta-rad")]),
            "attitude_psi_rad": np.array([self.simulation.get_property_value("attitude/psi-rad")]),
            "goal_lat_geod_deg": np.array([self.goal[0]]),
            "goal_long_gc_deg": np.array([self.goal[1]]),
            "goal_h_sl_meters": np.array([self.goal[2]])
        }

    def render(self, mode = 'human'):
        scale = 1e-3
        RADIUS = 6.3781e6
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
        obs = self.getStates()
        x = rad(obs["pos_lat_geod_deg"]) * scale * RADIUS
        y = rad(obs["pos_long_gc_deg"]) * scale * RADIUS
        z = obs["pos_h_sl_meters"] * scale

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