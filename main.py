import gym
import jsbsim_gym.jsbsim_gym
# import imageio as iio
from jsbsim_gym.features import JSBSimFeatureExtractor
from stable_baselines3 import SAC

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

"""

Bu kod, JSBSim tabanlı bir OpenAI Gym çevresini ve eğitilmiş bir Soft Actor-Critic (SAC) politikasını yükleyerek, çevrede bir döngü içinde politikayı kullanarak simülasyonu çalıştırmaktadır. İşte bu kodun genel açıklamaları:

gym ve jsbsim_gym modüllerini içe aktarıyor. Ayrıca, çevrenin özellik çıkarma sınıfı olan JSBSimFeatureExtractor'ı içe aktarıyor.

stable_baselines3 kütüphanesinden SAC algoritması ve çevreyi oluşturuyor.

obs değişkenine çevreyi sıfırlayarak başlangıç gözlemini alıyor.

done değişkenini kontrol ederek simülasyonun bitip bitmediğini belirleyen bir döngü başlatıyor.

Her döngü adımında, çevreyi render ediyor (env.render(mode='rgb_array')), politikadan bir aksiyon seçiyor, bu aksiyonu çevreye uyguluyor (env.step(action)), elde edilen gözlem, ödül ve bitiş durumu bilgilerini yazdırıyor.

Döngü, simülasyon bitene kadar devam ediyor.

Çevreyi kapatıyor (env.close()).

Bu kod, eğitilmiş bir SAC modelini kullanarak JSBSim çevresinde simülasyon çalıştırmak için kullanılıyor. 
Modelin eğitildiği durumları gözlemleyebilir ve çeşitli simülasyon durumlarında politikanın nasıl performans gösterdiğini gözlemleyebilirsiniz.

"""