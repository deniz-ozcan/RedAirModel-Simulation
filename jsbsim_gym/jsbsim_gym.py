import random
import jsbsim
import gym
import numpy as np
from .visualization.rendering import Viewer, load_mesh, RenderObject, Grid
from .visualization.quaternion import Quaternion
from gym import spaces

"""
### Description Gym environment using JSBSim to simulate an F-16 aerodynamics model with a simple point-to-point
navigation task. The environment terminates when the agent enters a targetF16 around the goal or crashes by flying
lower than sea level. The goal is initialized at a random location in a targetF16 around the agent's starting
position.

### Observation Th,
angular rates, attitude, and position of the goal (concatenated in that order). Units are meters and radians.

### Action Space
Actions are given as normalized body rate commands and throttle command. 
These are passed into a low-level PID controller built into the JSBSim model itself. 
The rate commands should be normalized between [-1, 1] and the 
throttle command should be [0, 1].

### Rewards
A positive reward is given for reaching the goal and a negative reward is given for crashing. 
It is recommended to use the PositionReward wrapper below to eliminate the problem of sparse rewards.

### Basit bir noktadan noktaya navigasyon görevi ile bir F-16 aerodinamik modelini simüle etmek için 
JSBSim'i kullanan ortam. Ajan kalenin etrafında bir silindire girdiğinde veya deniz seviyesinden 
daha alçakta uçarak çarptığında ortam sona erer. Hedef, aracının başlangıç konumu etrafındaki bir silindirde 
rastgele bir konumda başlatılır.

### Gözlem Gözlem, ajanın konumu, hızı (mach, alfa, beta), açısal hızlar, tutum ve hedefin konumu (bu sırayla 
birleştirilmiş) olarak verilir. Birimler metre ve radyandır.

### Aksiyon Alanı
Eylemler normalleştirilmiş vücut hızı komutları ve gaz kelebeği komutu olarak verilir.
Bunlar, JSBSim modelinin kendisinde yerleşik olan düşük seviyeli bir PID denetleyicisine aktarılır.
Hız komutları [-1, 1] ile
gaz kelebeği komutu [0, 1] olmalıdır.

### Ödüller
Hedefe ulaşmak için olumlu bir ödül verilirken, hedefe ulaşmak için olumsuz bir ödül verilir.
Seyrek ödül sorununu ortadan kaldırmak için aşağıdaki PositionReward sarmalayıcısının kullanılması tavsiye edilir.

Bu çevre, JSBSim adlı bir uçuş dinamik modelleme kütüphanesini kullanarak bir uçağın kontrolünü sağlar. 

__init__(self, root='.'): Çevre sınıfının başlatıcı metodu. Gözlem ve eylem uzaylarını tanımlar, 
JSBSim simülasyonunu başlatır, F-16 modelini yükler ve başlangıç koşullarını ayarlar.

_set_initial_conditions(self): JSBSim başlangıç koşullarını ayarlayan özel bir metod.

step(self, action): Bir adım simülasyonu ilerletir. JSBSim'e kontrol girişlerini aktarır, simülasyonu birkaç adım 
ilerletir, ardından durumu alır ve ödülü hesaplar.

_get_state(self): JSBSim'den durumu alır ve çevre sınıfının içindeki self.state özelliğine kaydeder.

reset(self, seed=None): Çevreyi sıfırlar, başlangıç koşullarını yeniden başlatır ve yeni bir hedef belirler.

render(self, mode='human'): Simülasyonu görselleştirir. Görselleştirmeyi sağlayan Viewer sınıfını kullanır.

close(self): Görselleştirmeyi sonlandırır.

Bu çevre, bir uçağın belirli bir hedefe ulaşma görevini simüle eder. Kontrol girişleri roll, pitch, yaw ve gaz (throttle) olarak dört boyutludur. 
Hedefe ulaşma veya çarpışma durumlarına göre ödüller verilir. Görselleştirmek için bir Viewer sınıfını kullanarak uçağın ve hedefin 3D konumunu ve durumunu gösterir.

gym.space.dict bu formatta obs spacelere bakılması gerekiyor.
"""

STATE_FORMAT = [
    "position/lat-gc-rad", "position/long-gc-rad", "position/h-sl-meters", 
    "velocities/mach", "aero/alpha-rad", "aero/beta-rad", 
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec",
    "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad"]

STATE_LOW = np.array([-np.inf, -np.inf, 0, 0, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, 0])
STATE_HIGH = np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf])

# STATE_LOW = np.array([-np.inf, -np.inf, 0, 0, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, 0], dtype = np.float32)
# STATE_HIGH = np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf], dtype = np.float32)

RADIUS = 6.3781e6 # Radius of the earth

class JSBSimEnv(gym.Env):
    def __init__(self, root='.'):
        super().__init__()

        # Sen
        self.action_space = spaces.Box(np.array([-1, -1, -1, 0]), 1, (4,))
        # self.observation_space = spaces.Box(STATE_LOW, STATE_HIGH, (15,))

        self.observation_space = spaces.Dict({
            "position_lat_gc_rad": spaces.Box(low = float('-inf'), high = float('inf'), shape = (1, ), dtype = np.float32),
            "position_long_go_rad": spaces.Box(low = float('-inf'), high = float('inf'), shape = (1, ), dtype = np.float32),
            "position_h_sl_meters": spaces.Box(low = 0, high = 15000, shape = (1,), dtype = np.float32),
            "aero_alpha_rad": spaces.Box(low = -0.2618, high = 0.6109, shape = (1,), dtype = np.float32),
            "aero_beta_rad": spaces.Box(low = -0.1745, high = 0.2618, shape = (1,), dtype = np.float32),
            "velocities_mach": spaces.Box(low = 0, high = 2.05, shape = (1,), dtype = np. float32),
            "velocities_p_rad_sec": spaces.Box(low = -0.52, high = 0.52, shape = (1,), dtype = np.float32),
            "velocities_q_rad_sec": spaces.Box(low = -0.44, high = 0.44, shape = (1,), dtype = np.float32),
            "velocities_r_rad_sec": spaces.Box(low = -0.32, high = 0.32, shape = (1,), dtype = np.float32),
            "attitude/phi-rad": spaces.Box(low = -0.2618, high = 0.2618, shape = (1,), dtype = np.float32),
            "attitude/theta-rad": spaces.Box(low = -0.2618, high = 0.2618, shape = (1,), dtype = np.float32),
            "attitude/psi-rad": spaces.Box(low = -3.1416, high = 3.1416, shape = (1,), dtype = np.float32),
            "goal/x": spaces.Box(low = STATE_LOW[12:13], high = STATE_HIGH[12:13], dtype = np.float32),
            "goal/y": spaces.Box(low = STATE_LOW[13:14], high = STATE_HIGH[13:14], dtype = np.float32),
            "goal/z": spaces.Box(low = STATE_LOW[14:15], high = STATE_HIGH[14:15], dtype = np.float32),
        })
        """
        'attitude/pitch-rad'
        'attitude/roll-rad'
        'attitude/psi-deg'
        'aero/beta-deg'

        # velocities
        'velocities/u-fps'
        'velocities/v-fps'
        'velocities/w-fps'
        'velocities/v-north-fps'
        'velocities/v-east-fps'
        'velocities/v-down-fps'
        'velocities/h-dot-fps'

        # controls state
        'fcs/left-aileron-pos-norm'
        'fcs/right-aileron-pos-norm'
        'fcs/elevator-pos-norm'
        'fcs/rudder-pos-norm'
        'fcs/throttle-pos-norm'
        'gear/gear-pos-norm'

        # engines
        'propulsion/engine/set-running'
        'propulsion/set-running'
        'propulsion/engine/thrust-lbs'

        # controls command
        'fcs/aileron-cmd-norm'
        'fcs/elevator-cmd-norm'
        'fcs/rudder-cmd-norm'
        'fcs/throttle-cmd-norm'
        'fcs/mixture-cmd-norm'
        'fcs/throttle-cmd-norm[1]'
        'fcs/mixture-cmd-norm[1]'
        'gear/gear-cmd-norm'

        # simulation
        'simulation/dt'
        'simulation/sim-time-sec'
        """

        # Initialize JSBSim / JSBSim'i başlat
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        # Load F-16 model and set initial conditions / F-16 modelini yükleyin ve başlangıç koşullarını ayarlayın
        self.simulation.load_model('f16') 
        self._set_initial_conditions()
        self.simulation.run_ic()

        self.down_sample = 4
        self.state = np.zeros(12)
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

    # Set engines running, forward velocity, and altitude / Motorları çalıştır, ileri hız ve irtifa ayarla
    def _set_initial_conditions(self):
        rand = random.random()
        range10 = random.randint(1, 10)
        randdeg = random.uniform(0, 360)
        """
        # initial conditions
        'ic/h-sl-ft'
        'ic/terrain-elevation-ft'
        'ic/long-gc-deg'
        'ic/lat-geod-deg'
        'ic/u-fps'
        'ic/v-fps'
        'ic/w-fps'
        'ic/p-rad_sec'
        'ic/q-rad_sec'
        'ic/r-rad_sec'
        'ic/roc-fpm'
        'ic/psi-true-deg'

        """
        self.simulation.set_property_value('propulsion/set-running', -1) # motorları daha yavaş çalıştır
        self.simulation.set_property_value('ic/u-fps', (range10)*(rand * 100))# farklı hızda başlat
        self.simulation.set_property_value('ic/h-sl-ft', (range10 + 5)*round(rand * 1000)) # farklı bir irtifada başlat
        
        self.simulation.set_property_value('ic/psi-true-deg', round(rand * 100, 4))# farklı bir yönle başlat
        self.simulation.set_property_value('ic/long-gc-deg', -round(randdeg, 4)) # farklı bir boylamda başlat
        self.simulation.set_property_value('ic/lat-gc-deg', round(randdeg, 4)) # farklı bir enlemde başlat
        self.simulation.set_property_value('gear/gear-cmd-norm', round(rand, 1)) # Iniş takımı kontrol komutunu ayarla
        self.simulation.set_property_value('gear/gear-pos-norm', round(rand, 1)) # Iniş takımı pozisyonunu ayarla
        self.simulation.set_property_value('fcs/aileron-cmd-norm', round(rand, 1)) # Aileron kontrol komutunu ayarla
        self.simulation.set_property_value('fcs/elevator-cmd-norm', round(rand, 1)) # Elevatör kontrol komutunu ayarla
        self.simulation.set_property_value('fcs/rudder-cmd-norm', round(rand, 1)) # Rudder kontrol komutunu ayarla
        self.simulation.set_property_value('fcs/throttle-cmd-norm', round(rand, 1)) # Gaz kontrol komutunu ayarla

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
        # Get the JSBSim state and save to self.state / JSBSim durumunu alın ve self.state'e kaydedin
        self._get_state()
        reward = 0
        done = False
        # Check for collision with ground / Yerle çarpışma kontrolü
        if self.state[2] < 10:
            reward = -10
            done = True

        # Check if reached goal /  Hedefe ulaşıldı mı kontrol edin 
        if np.sqrt(np.sum((self.state[:2] - self.goal[:2]) ** 2)) < self.dg and abs(self.state[2] - self.goal[2]) < self.dg:
            reward = 10
            done = True

        return np.hstack([self.state, self.goal]), reward, done, {}

    def _get_state(self):
        # Gather all state properties from JSBSim
        for i, property in enumerate(STATE_FORMAT):
            self.state[i] = self.simulation.get_property_value(property)
        # Rough conversion to meters. This should be fine near zero lat/long
        self.state[:2] *= RADIUS

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
        # Get state from JSBSim and save to self.state / JSBSim'den durumu alın ve self.state'e kaydedin
        self._get_state()

        return np.hstack([self.state, self.goal])

    def render(self, mode = 'human'):
        scale = 1e-3

        if self.viewer is None:
            self.viewer = Viewer(1280, 720)

            f16_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.f16 = RenderObject(f16_mesh)
            self.f16.transform.scale = 1 / 30
            self.f16.color = 0, 0, .4

            goal_mesh = load_mesh(self.viewer.ctx, self.viewer.prog, "f16.obj")
            self.targetF16 = RenderObject(goal_mesh)
            self.targetF16.transform.scale = 1 / 30
            self.targetF16.color = 0, .4, 0

            self.viewer.objects.append(self.f16)
            self.viewer.objects.append(self.targetF16)
            self.viewer.objects.append(Grid(self.viewer.ctx, self.viewer.unlit, 21, 1.))

        # Rough conversion from lat/long to meters / yaklaşık dönüşüm
        x, y, z = self.state[:3] * scale
        self.f16.transform.z = x
        self.f16.transform.x = -y
        self.f16.transform.y = z
        rot = Quaternion.from_euler(*self.state[9:])
        rot = Quaternion(rot.w, -rot.y, -rot.z, rot.x)
        self.f16.transform.rotation = rot
        x, y, z = self.goal * scale
        self.targetF16.transform.z = x
        self.targetF16.transform.x = -y
        self.targetF16.transform.y = z

        r = self.f16.transform.position - self.targetF16.transform.position
        rhat = r / np.linalg.norm(r)
        x, y, z = r
        yaw = np.arctan2(-x, -z)
        pitch = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))

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
        displacement = obs[-3:] - obs[:3]
        distance = np.linalg.norm(displacement)
        reward += self.gain * (self.last_distance - distance)
        self.last_distance = distance
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        displacement = obs[-3:] - obs[:3]
        self.last_distance = np.linalg.norm(displacement)
        return obs

# Create entry point to wrapped environment
def wrap_jsbsim(**kwargs):
    return PositionReward(JSBSimEnv(**kwargs), 1e-2)

gym.register(id="JSBSim-v0", entry_point=wrap_jsbsim, max_episode_steps=1200) # Register the wrapped environment

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