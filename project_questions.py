
import numpy as np
from data_preparation import *
from kalman_filter import KalmanFilter, ExtendedKalmanFilter, ExtendedKalmanFilterSLAM
import graphs
import random
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation

random.seed(11)
np.random.seed(17)


class ProjectQuestions:
    def __init__(self, dataset):
        """
        Initializes:
        - lat: latitude [deg]
        - lon: longitude [deg]
        - yaw: heading [rad]
        - vf: forward velocity parallel to earth-surface [m/s]
        - wz: angular rate around z axis [rad/s]
        - enu - lla converted to enu
        - times - for each frame, how much time has elapsed from the previous frame
        - enu_noise - enu with Gaussian noise (sigma_xy=3 meters)
        """
        self.dataset = dataset
        # build_ENU_from_GPS_trajectory
        self.enu, self.times, self.yaw_vf_wz = build_GPS_trajectory(self.dataset)
        # add noise to the trajectory
        self.sigma_xy = 3
        # TODO(ofekp): I did not use add_gaussian_noise as it requires adding noise one col at a time
        self.enu_noise = self.enu + np.concatenate((np.random.normal(0, self.sigma_xy, (self.enu.shape[0], 2)), np.zeros((self.enu.shape[0], 1))), axis=1)
    
    def Q1(self):
        """
        That function runs the code of question 1 of the project.
        Loads from kitti dataset, set noise to GT-gps values, and use Kalman Filter over the noised values.
        """
        # plot lla
        gps_imu_data = [gpu_imu[0] for gpu_imu in self.dataset.get_gps_imu()]
        lon_vec = np.array([e.lon for e in gps_imu_data]).reshape(-1, 1)
        lat_vec = np.array([e.lat for e in gps_imu_data]).reshape(-1, 1)
        alt_vec = np.array([e.alt for e in gps_imu_data]).reshape(-1, 1)
        graphs.plot_trajectory_and_height(np.concatenate([lon_vec, lat_vec, alt_vec], axis=1),
                                          'Trajectory of the drive - longitude/latitude in degrees',
                                          'Longitude [degrees]',
                                          'Latitude [degrees]',
                                          'Altitude of the drive',
                                          'Frame Index',
                                          'Altitude [meters]')
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "lla_trajectory", overwrite=False)

        # plot enu
        graphs.plot_trajectory_and_height(self.enu,
                                          'Trajectory of the drive - east/north in meters',
                                          'East [meters]',
                                          'North [meters]',
                                          'Up axis for the drive',
                                          'Frame Index',
                                          'Up [meters]')
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "enu_trajectory", overwrite=False)

        # plot enu and enu_noise
        graphs.plot_trajectory_comparison(self.enu, self.enu_noise)
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "comparison_enu_with_and_without_noise", overwrite=False)
        
        # KalmanFilter
        configs = [
            {
                "sigma_n": 2,
                "dead_reckoning": False
            },
        ]

        for config in configs:
            sigma_n = config['sigma_n']
            is_dead_reckoning = config['dead_reckoning']
            kf = KalmanFilter(self.enu_noise, self.times, self.sigma_xy, sigma_n, is_dead_reckoning)
            enu_kf, cov_graph_x, cov_graph_y, final_cov = kf.run()

            graphs.plot_trajectory_comparison(self.enu, self.enu_noise, enu_predicted=enu_kf[:, 0:2])
            # graphs.show_graphs()
            graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_const_vel_predicted_path_sigma_n_{}{}".format(sigma_n, "dead_reckoning" if is_dead_reckoning else ""), overwrite=False)

            RMSE, maxE = KalmanFilter.calc_RMSE_maxE(self.enu, enu_kf)
            print("Kalman Filter - Constant Velocity")
            print("maxE [{}]".format(maxE))
            print("RMSE [{}]".format(RMSE))
            assert maxE < 6.95

            # let's draw the error
            e_x = self.enu[:, 0] - enu_kf[:, 0].squeeze()  # e_x dim is [1, -1]
            e_y = self.enu[:, 1] - enu_kf[:, 1].squeeze()  # e_y dim is [1, -1]
            graphs.plot_error((np.asarray(e_x).squeeze(), cov_graph_x), (np.asarray(e_y).squeeze(), cov_graph_y))
            # graphs.show_graphs()
            graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_error_comparison_x_y_sigma_n_{}{}".format(sigma_n, "dead_reckoning" if is_dead_reckoning else ""), overwrite=False)

            # show the covariance matrix of the state as an ellipse
            fig, ax = plt.subplots()
            ellipse = error_ellipse([0, 0], final_cov)
            draw_ellipse(ax, [ellipse._center[0], ellipse._center[1], ellipse.angle], ellipse.width, ellipse.height, 'b')
            ax.set_aspect('equal', adjustable='box')
            # graphs.show_graphs()
            graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_cov_as_ellipse_sigma_n_{}{}".format(sigma_n, "dead_reckoning" if is_dead_reckoning else ""), overwrite=False)

    def Q2(self):
        # plot yaw, yaw rate and forward velocity
        graphs.plot_yaw_yaw_rate_fv(self.yaw_vf_wz[:, 0], self.yaw_vf_wz[:, 2], self.yaw_vf_wz[:, 1])
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Extended Kalman Filter", "ekf_yaw_yaw_rate_forward_velocity_graph", overwrite=False)

        sigma_theta = 0.0
        sigma_vf = 2.0
        sigma_wz = 0.02
        k = 2
        is_dead_reckoning = False
        ekf = ExtendedKalmanFilter(self.enu_noise, self.yaw_vf_wz, self.times, self.sigma_xy, sigma_theta, sigma_vf, sigma_wz, k, is_dead_reckoning)
        # state_ekf, sigma_x_xy_yx_y_t = ekf.run(locations_noised, times, yaw_vf_wz_noised, do_only_predict=False)  # todo: I did not follow this template...
        state_kf, cov_graph_x, cov_graph_y, cov_graph_yaw, covs = ekf.run()

        RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(self.enu, state_kf)
        print("Extended Kalman Filter - Noisy East/North")
        print("maxE [{}]".format(maxE))
        print("RMSE [{}]".format(RMSE))
        assert maxE < 6.95

        graphs.plot_trajectory_comparison(self.enu, self.enu_noise, enu_predicted=state_kf[:, 0:2])
        graphs.show_graphs()
        # graphs.show_graphs("../Results/Kalman Filter", "ekf_const_vel_predicted_path_sigma_n_{}{}".format(k, "dead_reckoning" if is_dead_reckoning else ""), overwrite=False)

        # let's draw the error
        e_x = self.enu[:, 0].squeeze() - state_kf[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = self.enu[:, 1].squeeze() - state_kf[:, 1].squeeze()  # e_y dim is [1, -1]
        e_yaw = self.yaw_vf_wz[:, 0].squeeze() % (2 * np.pi) - state_kf[:, 2].squeeze() % (2 * np.pi)  # e_yaw dim is [1, -1]
        e_yaw = np.where(e_yaw > 6, e_yaw - 2 * np.pi, e_yaw)  # I had one sample where this occurred
        e_yaw = np.where(e_yaw < -5.5, e_yaw + 2 * np.pi, e_yaw)  # I had a few samples where this occurred
        graphs.plot_error((np.asarray(e_x).squeeze(), cov_graph_x), (np.asarray(e_y).squeeze(), cov_graph_y), (np.asarray(e_yaw).squeeze(), cov_graph_yaw))
        graphs.show_graphs()

        # RMSE, maxE = ekf.calc_RMSE_maxE(locations_gt, locations_ekf)
 
        anim = graphs.build_animation(self.enu[:, 0:2], self.enu_noise[:, 0:2], state_kf[:, 0:2], covs, "Animated trajectory", "East [meters]", "North [meters]", "l0", "l1", "l2")
        graphs.save_animation(anim, "../Results/Extended Kalman Filter", "ekf_predict_animation")

    def plot_odometry(self, sensor_data):
        num_frames = len(sensor_data) // 2
        state = np.array([[0, 0, 0]]).reshape(1, 3)
        for i in range(num_frames):
            curr_odometry = sensor_data[i, 'odometry']
            t = np.array([
                curr_odometry['t'] * np.cos(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['t'] * np.sin(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['r1'] + curr_odometry['r2']
            ]).reshape(3, 1)
            new_pos = state[-1, :].reshape(3, 1) + t
            state = np.concatenate([state, new_pos.reshape(1, 3)], axis=0)
        return state

    def Q3(self):
        landmarks = self.dataset.load_landmarks()
        sensor_data_gt = self.dataset.load_sensor_data()
        state = self.plot_odometry(sensor_data_gt)
        graphs.plot_trajectory(state, "GT trajectory from odometry", "X [meters]", "Y [meters]")
        graphs.show_graphs()

        # add Gaussian noise to the odometry data
        variance_r1_t_r2 = [0.01 ** 2, 0.1 ** 2, 0.01 ** 2]
        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        state_noised = self.plot_odometry(sensor_data_noised)
        graphs.plot_trajectory(state_noised, "GT trajectory from odometry", "X [meters]", "Y [meters]")
        graphs.show_graphs()

        print(sensor_data_gt)

    #     sigma_x_y_theta = #TODO
    #     variance_r_phi = #TODO

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)
        # frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)

    #     graphs.plot_single_graph(mu_arr_gt[:,0] - mu_arr[:,0], "x-$x_n$", "frame", "error", "x-$x_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,0]))
    #     graphs.plot_single_graph(mu_arr_gt[:,1] - mu_arr[:,1], "y-$y_n$", "frame", "error", "y-$y_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,1]))
    #     graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:,2] - mu_arr[:,2]), "$\\theta-\\theta_n$",
    #                              "frame", "error", "$\\theta-\\theta_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,2]))
    #
    #     graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[:,3]),
    #                              "landmark 1: x-$x_n$", "frame", "error [m]", "x-$x_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,3]))
    #     graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:,4]),
    #                              "landmark 1: y-$y_n$", "frame", "error [m]", "y-$y_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,4]))
    #
    #     graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:,5]),
    #                              "landmark 2: x-$x_n$", "frame", "error [m]", "x-$x_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,5]))
    #     graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:,6]),
    #                              "landmark 2: y-$y_n$", "frame", "error [m]", "y-$y_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,6]))
    #
    #     ax.set_xlim([-2, 12])
    #     ax.set_ylim([-2, 12])
    #
    #     ani = animation.ArtistAnimation(fig, frames, repeat=False)
    #     graphs.show_graphs()
    #     # ani.save('im.mp4', metadata={'artist':'me'})
    
    def run(self):
        # self.Q1()
        # self.Q2()
        self.Q3()
        
        
