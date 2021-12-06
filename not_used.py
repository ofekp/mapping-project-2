# ***********************
# please ignore this file
# =======================

def Q2_BenZion(self):
    """
    As suggested by Ben-Zion
    Analysis before adding noise to vf and wz, and while using dead reckoning as the ground truth
    """
    k = 0.3
    sigma_theta = 0.3
    sigma_vf = 0.0
    sigma_wz = 0.0
    is_dead_reckoning = True
    ekf = ExtendedKalmanFilter(self.enu[:, 0:2], self.yaw_vf_wz, self.times, self.sigma_xy, sigma_theta, sigma_vf,
                               sigma_wz, k, is_dead_reckoning, dead_reckoning_start_sec=0.0)
    state_dead_rec, _ = ekf.run()
    is_dead_reckoning = False
    ekf = ExtendedKalmanFilter(state_dead_rec[:, 0:2], self.yaw_vf_wz, self.times, self.sigma_xy, sigma_theta, sigma_vf,
                               sigma_wz, k, is_dead_reckoning, dead_reckoning_start_sec=0.0)
    state_ekf, covs = ekf.run()
    is_dead_reckoning = True
    ekf = ExtendedKalmanFilter(state_dead_rec[:, 0:2], self.yaw_vf_wz, self.times, self.sigma_xy, sigma_theta, sigma_vf,
                               sigma_wz, k, is_dead_reckoning, dead_reckoning_start_sec=0.0)
    state_ekf_dead_rec, _ = ekf.run()
    graphs.plot_trajectory_comparison_dead_reckoning(state_dead_rec, state_ekf, state_ekf_dead_rec)
    graphs.show_graphs()
    # graphs.show_graphs("../Results/Extended Kalman Filter", "ekf_dead_rec_as_gt_predicted_path")
    # let's draw the error
    e_x = state_dead_rec[:, 0].squeeze() - state_ekf[:, 0].squeeze()  # e_x dim is [1, -1]
    e_y = state_dead_rec[:, 1].squeeze() - state_ekf[:, 1].squeeze()  # e_y dim is [1, -1]
    e_yaw = normalize_angles_array(self.yaw_vf_wz[:, 0].squeeze() - state_ekf[:, 2].squeeze())
    cov_graph_x = np.sqrt(covs[:, 0])
    cov_graph_y = np.sqrt(covs[:, 4])
    cov_graph_yaw = np.sqrt(covs[:, 8])
    graphs.plot_error((np.asarray(e_x).squeeze(), cov_graph_x), (np.asarray(e_y).squeeze(), cov_graph_y),
                      (np.asarray(e_yaw).squeeze(), cov_graph_yaw))
    graphs.show_graphs()
    # graphs.show_graphs("../Results/Extended Kalman Filter", "ekf_dead_rec_as_gt_predicted_path_error_and_sigma")
    RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(state_dead_rec, state_ekf)
    print("Extended Kalman Filter - Noisy East/North - R == 0 - dead reckoning as ground truth")
    print("maxE [{:.5f}] meters".format(maxE))
    print("RMSE [{:.5f}]".format(RMSE))
    assert maxE < 1.3