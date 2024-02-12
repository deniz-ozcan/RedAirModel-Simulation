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

### Observation The observation is given as the position of the agent, velocity (mach, alpha, beta),
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


# STATE_LOW = np.array([-np.inf, -np.inf, 0, 0, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, 0], dtype=np.float32)
# STATE_HIGH = np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf], dtype=np.float32)

RADIUS = 6.3781e6 # Radius of the earth

class JSBSimEnv(gym.Env):
    def __init__(self, root='.'):
        super().__init__()

        # Set observation and action space format / Gözlem ve eylem alanı biçimini ayarlayın
        self.action_space = spaces.Box(np.array([-1, -1, -1, 0]), 1, (4,))
        # self.observation_space = spaces.Box(STATE_LOW, STATE_HIGH, (15,))

        self.observation_space = spaces.Dict({
            "position": spaces.Box(low=STATE_LOW[:3], high=STATE_HIGH[:3], dtype=np.float32),
            "mach": spaces.Box(low=STATE_LOW[3:4], high=STATE_HIGH[3:4], dtype=np.float32),
            "alpha_beta": spaces.Box(low=STATE_LOW[4:6], high=STATE_HIGH[4:6], dtype=np.float32),
            "angular_rates": spaces.Box(low=STATE_LOW[6:9], high=STATE_HIGH[6:9], dtype=np.float32),
            "phi_theta": spaces.Box(low=STATE_LOW[9:11], high=STATE_HIGH[9:11], dtype=np.float32),
            "psi": spaces.Box(low=STATE_LOW[11:12], high=STATE_HIGH[11:12], dtype=np.float32),
            "goal": spaces.Box(low=STATE_LOW[12:], high=STATE_HIGH[12:], dtype=np.float32),
            # "position": spaces.Box(low=-100, high=100, shape=(3,), dtype=np.float32),
            # "mach": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # "alpha_beta": spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32),
            # "angular_rates": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            # "phi_theta": spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32),
            # "psi": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            # "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

        # Initialize JSBSim / JSBSim'i başlat
        self.simulation = jsbsim.FGFDMExec(root, None)
        self.simulation.set_debug_level(0)
        self.simulation.load_model('f16') # Load F-16 model and set initial conditions / F-16 modelini yükleyin ve başlangıç koşullarını ayarlayın
        self._set_initial_conditions()
        self.simulation.run_ic()

        self.down_sample = 4
        self.state = np.zeros(12)
        self.goal = np.zeros(3)
        self.dg = 100
        self.viewer = None

    def _set_initial_conditions(self):# Set engines running, forward velocity, and altitude / Motorları çalıştır, ileri hız ve irtifa ayarla
        self.simulation.set_property_value('propulsion/set-running', -1)
        self.simulation.set_property_value('ic/u-fps', 900.)
        self.simulation.set_property_value('ic/h-sl-ft', 5000)

        # add different initial cond.

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
        # Gather all state properties from JSBSim / JSBSim'den tüm durum özelliklerini toplayın
        for i, property in enumerate(STATE_FORMAT):
            self.state[i] = self.simulation.get_property_value(property)

        # Rough conversion to meters. This should be fine near zero lat/long / Metreye yaklaşık dönüşüm. Bu, sıfıra yakın enlem / boylamda iyi olmalıdır.
        self.state[:2] *= RADIUS

    def reset(self, seed=None):
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

    def render(self, mode='human'):
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

        # Rough conversion from lat/long to meters / Enlem / boylamdan metre cinsinden yaklaşık dönüşüm
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
    """
    This wrapper adds an additional reward to the JSBSimEnv.
    The agent is rewarded based when movin closer to the goal and penalized when moving away.
    Staying at the same distance will result in no additional reward. 
    The gain may be set to weight the importance of this reward.

    Bu sarmalayıcı JSBSimEnv'e ek bir ödül ekler.
    Temsilci hedefe yaklaştığında ödüllendirilir, uzaklaştığında ise cezalandırılır.
    Aynı mesafede kalmak ek bir ödülle sonuçlanmayacaktır.
    Kazanç, bu ödülün önemine göre ayarlanabilir.

    Bu Python sınıfı, OpenAI Gym çevresini sarmak (wrapper) ve çevrenin her adımında ödülü değiştirmek için tasarlanmış bir PositionReward sarmalayıcısıdır. 
    Bu sarmalayıcı, çevrenin gözlem bilgisini kullanarak belirli bir konumdan uzaklığı ölçer ve bu uzaklık değişikliklerine dayalı olarak ödülü günceller.
    İşlevselliği şu şekildedir:
    __init__(self, env, gain): Sarmalayıcıyı başlatır. 
    env parametresi, sarmalayıcıya dahil edilecek olan OpenAI Gym çevresidir.
    gain parametresi, her adımda ödül değişikliğini kontrol eden bir faktördür.
    step(self, action): Çevrenin bir adımını gerçekleştirir ve önceki adımdan bu adıma kadar olan konum değişikliğine dayalı olarak ödülü günceller.
    reset(self): Çevreyi sıfırlar ve başlangıç konumundaki gözlemi alır. 
    Başlangıçta önceki konum last_distance olarak kaydedilir.
    Bu sarmalayıcı, her adımda önceki konumdan ne kadar uzaklaşıldığını ölçer ve bu uzaklığa dayalı olarak ödülü günceller. 
    gain parametresi, uzaklık değişikliğinin ödüle olan etkisini kontrol eder. 
    Eğer gain pozitifse, uzaklık arttıkça ödül de artar; eğer negatifse, uzaklık arttıkça ödül azalır.
    Bu tür sarmalayıcılar, çevrelerin ödül fonksiyonunu özelleştirmek ve öğrenme algoritmalarını belirli bir hedefe odaklamak için kullanılır.
    """

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