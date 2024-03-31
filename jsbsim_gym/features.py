import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class JSBSimFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, obs_space):
        super().__init__(obs_space, 17)

    def forward(self, obs):
        position = th.concat([obs["pos_lat_geod_deg"], obs["pos_long_gc_deg"], obs["pos_h_sl_meters"]], dim=1)
        goal = th.concat([obs["goal_lat_geod_deg"], obs["goal_long_gc_deg"], obs["goal_h_sl_meters"]], dim=1)

        # Transform position
        displacement = goal - position

        # We normalize distance this way to bound it between 0 and 1
        dist_norm = 1 / (1 + th.sqrt(th.sum(displacement[:, :2] ** 2, 1, True)) * 1e-3)

        # Normalize these by approximate flight ceiling
        dz_norm = (obs["goal_h_sl_meters"] - obs["pos_h_sl_meters"]) / 15000
        alt_norm = obs["pos_h_sl_meters"] / 15000

        alpha_beta = th.concat([obs["aero_alpha_rad"], obs["aero_beta_rad"]], dim=1)
        cab, sab = th.cos(alpha_beta), th.sin(alpha_beta)

        phi_theta = th.concat([obs["attitude_phi_rad"], obs["attitude_theta_rad"]], dim=1)
        cpt, spt = th.cos(phi_theta), th.sin(phi_theta)

        rel_bearing = th.atan2(displacement[:,1:2], displacement[:,0:1]) - obs["attitude_psi_deg"]
        cr, sr = th.cos(rel_bearing), th.sin(rel_bearing) # TODO: böyle mi olmalı?

        angular_rates = th.concat([obs["velocities_p_rad_sec"], obs["velocities_q_rad_sec"], obs["velocities_r_rad_sec"]], dim=1)
        mach = obs["velocities_mach"]

        return th.concat([dist_norm, dz_norm, alt_norm, mach, angular_rates, cab, sab, cpt, spt, cr, sr], dim=1)