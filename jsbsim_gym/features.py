import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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