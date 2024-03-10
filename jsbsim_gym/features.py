import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import datetime
from math import degrees as deg

class JSBSimFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, obs_space):
        super().__init__(obs_space, 17)

    def forward(self, obs):
        position = th.concat([obs["position_lat_gc_rad"], obs["position_long_gc_rad"], obs["position_h_sl_meters"]], dim=1)
        mach = obs["velocities_mach"]
        alpha_beta = th.concat([obs["aero_alpha_rad"], obs["aero_beta_rad"]], dim=1)
        angular_rates = th.concat([obs["velocities_p_rad_sec"], obs["velocities_q_rad_sec"], obs["velocities_r_rad_sec"]], dim=1)
        phi_theta = th.concat([obs["attitude_phi_rad"], obs["attitude_theta_rad"]], dim=1)
        psi = obs["attitude_psi_rad"]
        goal = th.concat([obs["goal_x"], obs["goal_y"], obs["goal_z"]], dim=1)
        # Transform position
        displacement = goal - position
        distance = th.sqrt(th.sum(displacement[:, :2] ** 2, 1, True))
        dz = displacement[:,2:3]
        altitude = position[:, 2:3]
        abs_bearing = th.atan2(displacement[:,1:2], displacement[:,0:1])
        rel_bearing = abs_bearing - psi
        # We normalize distance this way to bound it between 0 and 1
        dist_norm = 1 / (1 + distance * 1e-3)
        # Normalize these by approximate flight ceiling
        dz_norm = dz / 15000
        alt_norm = altitude / 15000
        # Angles to Sine/Cosine pairs
        cab, sab = th.cos(alpha_beta), th.sin(alpha_beta)
        cpt, spt = th.cos(phi_theta), th.sin(phi_theta)
        cr, sr = th.cos(rel_bearing), th.sin(rel_bearing)
        return th.concat([dist_norm, dz_norm, alt_norm, mach, angular_rates, cab, sab, cpt, spt, cr, sr], dim=1)
"""
Feature extractor to help learn the JSBSim environment.

### Position
This extractor converts the position to relative cylindrical coordinates.
Raw altitude is also preserved since it's necessary to avoid crashing.

The distance to the goal is normalized as 1/(1+distance*scale).
'Scale' is a constant that we have set to 1e-3 (meters to kilometers).
The rest of the equation bounds the value between 0 and 1.
Additionally it approaches 0 as distance goes to infinity.
This means the impact of distance on the network diminishes as it increases.
The intuition behind this is that the policy  should depend more on relative bearing at greater distance (e.g. just turn to face the goal and fly straight.)

Relative height to the goal and raw altitude are normalized by the estimated flight ceiling of the F-16 (15000 meters).

### Velocities and angular rates
Velocities are left unchanged since mach, alpha, and beta are already pretty well scaled.
Angular rates are also left unchanged since they are unlikely to grow too large in practice due to the low-level regulator on the JSBSim model.

### Angles
All angles (attitude, relative bearing, alpha, beta) are converted to sinecosine pairs.
This makes sure that pi and -pi are the same in the feature space and will produce the same output.

JSBSim ortamını öğrenmeye yardımcı olacak özellik çıkarıcı.

### Konum
Bu çıkarıcı, konumu göreceli silindirik koordinatlara dönüştürür.
Çarpmayı önlemek gerektiğinden ham yükseklik de korunur.

Hedefe olan mesafe 1/(1+mesafe*ölçek) olarak normalize edilir.
'Ölçek' 1e-3 (metre ila kilometre) olarak ayarladığımız bir sabittir.
Denklemin geri kalanı 0 ile 1 arasındaki değeri sınırlar.
Ayrıca mesafe sonsuza giderken 0'a yaklaşır.
Bu, mesafe arttıkça ağ üzerindeki etkisinin azaldığı anlamına gelir.
Bunun arkasındaki sezgi, politikanın daha uzak mesafelerdeki göreceli yönlere daha fazla dayanması gerektiğidir (örneğin, sadece hedefe dönüp düz uçmak).

Hedefe göreli yükseklik ve ham irtifa, F-16'nın tahmini uçuş tavanına (15000 metre) göre normalleştirilir.

### Hızlar ve açısal oranlar
Mach, alfa ve beta zaten oldukça iyi ölçeklendiğinden hızlar değişmeden bırakılır.
JSBSim modelindeki düşük seviyeli regülatör nedeniyle pratikte çok fazla büyümeleri muhtemel olmadığından açısal oranlar da değişmeden bırakılmıştır.

### Açılar
Tüm açılar (tutum, bağıl yön, alfa, beta) sinüs çiftlerine dönüştürülür.
Bu, pi ve -pi'nin özellik alanında aynı olmasını ve aynı çıktıyı üretmesini sağlar.

Bu PyTorch JSBSimFeatureExtractor sınıfı, bir gözlem uzayını (obs space) giriş olarak alır ve bu gözlemleri önceden belirlenmiş bir şekilde işleyip özellik vektörlerine dönüştürür.
Bu dönüşüm, JSBSim simulasyonundan elde edilen özel bir gözlem formatı üzerinde gerçekleştirilmiştir.
Bu özellik vektörleri ardından genellikle bir politika ağında veya değer fonksiyonunda kullanılmak üzere tasarlanmıştır.

İşlevselliği şu şekildedir:
__init__(self, obs_space): Sınıfın başlatıcı metodudur. Gözlem uzayını (obs space) ve özellik vektörünün uzunluğunu belirler.
forward(self, obs): Bu metod, giriş olarak aldığı gözlemleri önceden belirlenmiş bir şekilde işleyip özellik vektörlerine dönüştürür.
Gözlemler, belirli bir sırayla kesilir ve işlenir. Daha sonra bu işlenmiş özellikler birleştirilerek tek bir tensor haline getirilir ve döndürülür.
Bu özellik çıkarma (feature extraction) sınıfının çıkarma işlemleri şu adımları içerir:
Pozisyon Dönüşümü (Transform Position): Hedef konumu ve uçağın mevcut pozisyonu arasındaki mesafe, yükseklik ve yön bilgilerini hesaplar ve normalize eder.
Açıları Sinüs/Kosinüs Çiftlerine Dönüştürme: Hız, açısal hız ve açılara ait bilgileri hesaplar ve bunları sinüs ve kosinüs çiftleri olarak döndürür.
Normalizasyon: Mesafe, yükseklik ve diğer bazı değerleri normalize eder.
Özellik Vektörünü Oluşturma: Elde edilen tüm bu normalize edilmiş değerleri birleştirip tek bir özellik vektörü oluşturur ve döndürür.
Bu sınıf, gözlemlerin belirli bir formatında olduğunu ve bu formatı önceden belirlenmiş bir şekilde işleyip özellik vektörlerine dönüştürdüğünü varsayar.
"""