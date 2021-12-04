
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

font = {'size': 20}
plt.rc('font', **font)


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
        self.sigma_vf = 2
        self.sigma_wz = 0.2
        self.enu_noise = self.enu + np.concatenate((np.random.normal(0, self.sigma_xy, (self.enu.shape[0], 2)), np.zeros((self.enu.shape[0], 1))), axis=1)
        self.yaw_vf_wz_noise = self.yaw_vf_wz + np.concatenate((
                                                                np.zeros((self.yaw_vf_wz.shape[0], 1)),  # no noise in yaw
                                                                np.random.normal(0, self.sigma_vf, (self.yaw_vf_wz.shape[0], 1)),  # noise in forward velocity
                                                                np.random.normal(0, self.sigma_wz, (self.yaw_vf_wz.shape[0], 1))  # noise in angular rate
                                                               ),
                                                               axis=1)

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
        graphs.show_graphs("../Results/Kalman Filter", "lla_trajectory")

        # plot enu
        graphs.plot_trajectory_and_height(self.enu,
                                          'Trajectory of the drive - east/north in meters',
                                          'East [meters]',
                                          'North [meters]',
                                          'Up axis for the drive',
                                          'Frame Index',
                                          'Up [meters]')
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "enu_trajectory")

        # plot enu and enu_noise
        graphs.plot_trajectory_comparison_with_and_without_noise(self.enu, self.enu_noise)
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "comparison_enu_with_and_without_noise")
        
        # KalmanFilter
        configs = [
            {
                "description": "sigma_n too high",
                "sigma_n": 5.0,
                "dead_reckoning": False,
                "result": None
            },
            {
                "description": "sigma_n too low",
                "sigma_n": 0.6,
                "dead_reckoning": False,
                "result": None
            },


            {
                "description": "optimal sigma_n",
                "sigma_n": 1.0,
                "dead_reckoning": False,
                "result": None
            },
            {
                "description": "optimal sigma_n dead reckoning",
                "sigma_n": 1.0,
                "dead_reckoning": True,
                "result": None
            },
        ]

        for config in configs:
            sigma_n = config['sigma_n']
            is_dead_reckoning = config['dead_reckoning']
            kf = KalmanFilter(self.enu_noise, self.times, self.sigma_xy, sigma_n, is_dead_reckoning)
            config['result'] = kf.run()

        graphs.plot_trajectory_comparison(self.enu, configs[0]['result'][0])
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_const_vel_predicted_path_sigma_n_too_high")

        graphs.plot_trajectory_comparison(self.enu, configs[1]['result'][0])
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_const_vel_predicted_path_sigma_n_too_low")

        enu_kf, covs = configs[2]['result']
        enu_dead_rec, _ = configs[3]['result']
        graphs.plot_trajectory_comparison_dead_reckoning(self.enu, enu_kf, enu_dead_rec)
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_const_vel_predicted_path_sigma_n_2_dead_rec")

        RMSE, maxE = KalmanFilter.calc_RMSE_maxE(self.enu, enu_kf)

        # let's draw the error
        e_x = self.enu[:, 0] - enu_kf[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = self.enu[:, 1] - enu_kf[:, 1].squeeze()  # e_y dim is [1, -1]
        graphs.plot_error((np.asarray(e_x).squeeze(), np.sqrt(covs[:, 0].squeeze())), (np.asarray(e_y).squeeze(), np.sqrt(covs[:, 3].squeeze())))
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_error_comparison_x_y_sigma_n_2_dead_rec")

        # show the covariance matrix of the state as an ellipse
        fig, ax = plt.subplots()
        ellipse = error_ellipse([0, 0], covs[-1].reshape(2, 2))
        draw_ellipse(ax, [ellipse._center[0], ellipse._center[1], ellipse.angle], ellipse.width, ellipse.height, 'b')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [meters]", fontsize=20)
        ax.set_ylabel("y [meters]", fontsize=20)
        ax.set_title("Final variance ", fontsize=20)
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_cov_as_ellipse_sigma_n_2_dead_rec")

        print("Kalman Filter - Constant Velocity")
        print("maxE [{:.5f}] meters".format(maxE))
        print("RMSE [{:.5f}]".format(RMSE))
        assert maxE < 5.6

        anim = graphs.build_animation(self.enu[:, 0:2].reshape(-1, 2), enu_dead_rec[:, 0:2].reshape(-1, 2),
                                      enu_kf[:, 0:2].reshape(-1, 2), covs.reshape(-1, 4), "Animated trajectory",
                                      "East [meters]", "North [meters]", "Ground truth", "Kalman filter prediction",
                                      "Dead reckoning (Kalman Gain is 0)")
        graphs.save_animation(anim, "../Results/Kalman Filter", "kf_and_dead_rec_animation")

    def Q2(self):
        # plot yaw, yaw rate and forward velocity
        graphs.plot_yaw_yaw_rate_fv(self.yaw_vf_wz[:, 0], self.yaw_vf_wz[:, 2], self.yaw_vf_wz[:, 1])
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Extended Kalman Filter", "ekf_yaw_yaw_rate_forward_velocity_graph")

        sigma_theta = 0.3
        k = 0.3

        configs = [
            # analysis before adding noise to vf and wz
            {
                "description": "ekf_no_noise_in_vf_wz_zero_R",
                "yaw_vf_wz": self.yaw_vf_wz,
                "sigma_vf": 0.0,
                "sigma_wz": 0.0,
                "result": (None, None, None, None),
                "bestMaxE": 1.3,
                "bestRMSE": 0.52
            },
            {
                "description": "ekf_no_noise_in_vf_wz_non_zero_R",
                "yaw_vf_wz": self.yaw_vf_wz,
                "sigma_vf": 0.5,
                "sigma_wz": 0.5,
                "result": (None, None, None, None),
                "bestMaxE": 4.4,
                "bestRMSE": 0.97
            },
            {
                "description": "ekf_no_noise_in_vf_wz_very_small_R",
                "yaw_vf_wz": self.yaw_vf_wz,
                "sigma_vf": 0.01,
                "sigma_wz": 0.01,
                "result": (None, None, None, None),
                "bestMaxE": 1.11,
                "bestRMSE": 0.38
            },

            # analysis after adding noise to vf and wz
            {
                "description": "ekf_noise_in_vf_wz_bigger_sigma_wz",
                "yaw_vf_wz": self.yaw_vf_wz_noise,
                "sigma_vf": 2.0,
                "sigma_wz": 1.5,
                "result": (None, None, None, None),
                "bestMaxE": 6.03,
                "bestRMSE": 1.54
            },

            {
                "description": "ekf_noise_in_vf_wz_optimal_R",
                "yaw_vf_wz": self.yaw_vf_wz_noise,
                "sigma_vf": 2.0,
                "sigma_wz": 0.2,
                "result": (None, None, None, None),
                "bestMaxE": 4.37,
                "bestRMSE": 1.23,
                "is_animation": True
            },
        ]

        for config in configs:
            yaw_vf_wz = config["yaw_vf_wz"]
            sigma_vf = config["sigma_vf"]
            sigma_wz = config["sigma_wz"]
            ekf = ExtendedKalmanFilter(self.enu_noise, yaw_vf_wz, self.times, self.sigma_xy, sigma_theta, sigma_vf, sigma_wz, k, False)
            state_ekf, covs = ekf.run()
            ekf = ExtendedKalmanFilter(self.enu_noise, yaw_vf_wz, self.times, self.sigma_xy, sigma_theta, sigma_vf, sigma_wz, k, True)
            state_dead_rec, covs_dead_rec = ekf.run()
            config['result'] = (state_ekf, state_dead_rec, covs, covs_dead_rec)

            # draw the trajectories
            graphs.plot_trajectory_comparison_dead_reckoning(self.enu, state_ekf, state_dead_rec)
            graphs.show_graphs("../Results/Extended Kalman Filter", config['description'] + "_predicted_path")

            # draw the error
            e_x = self.enu[:, 0].squeeze() - state_ekf[:, 0].squeeze()  # e_x dim is [1, -1]
            e_y = self.enu[:, 1].squeeze() - state_ekf[:, 1].squeeze()  # e_y dim is [1, -1]
            e_yaw = normalize_angles_array(self.yaw_vf_wz[:, 0].squeeze() - state_ekf[:, 2].squeeze())
            cov_graph_x = np.sqrt(covs[:, 0])
            cov_graph_y = np.sqrt(covs[:, 4])
            cov_graph_yaw = np.sqrt(covs[:, 8])
            graphs.plot_error((np.asarray(e_x).squeeze(), cov_graph_x), (np.asarray(e_y).squeeze(), cov_graph_y),
                              (np.asarray(e_yaw).squeeze(), cov_graph_yaw))
            # graphs.show_graphs()
            graphs.show_graphs("../Results/Extended Kalman Filter", config['description'] + "_error_and_sigma")

            # print the maxE and RMSE
            RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(self.enu, state_ekf)
            print("Extended Kalman Filter for [{}]".format(config['description']))
            print("maxE [{:.5f}] meters".format(maxE))
            print("RMSE [{:.5f}]".format(RMSE))
            assert maxE < config['bestMaxE']
            assert RMSE < config['bestRMSE']

            if 'is_animation' in config and config['is_animation']:
                # this animation shows the covariances of the EKF estimated path
                anim = graphs.build_animation(self.enu[:, 0:2], state_dead_rec[:, 0:2], state_ekf[:, 0:2],
                                              covs[:, [0, 1, 3, 4]], "Animated trajectory", "East [meters]",
                                              "North [meters]", "Ground truth", "Dead reckoning (Kalman Gain is 0)",
                                              "Kalman filter prediction")
                graphs.save_animation(anim, "../Results/Extended Kalman Filter", config['description'] + "_animation_with_ekf_cov")

                # this animation shows the covariances of the dead reckoning estimated path
                anim = graphs.build_animation(self.enu[:, 0:2], state_ekf[:, 0:2], state_dead_rec[:, 0:2],
                                              covs_dead_rec[:, [0, 1, 3, 4]], "Animated trajectory", "East [meters]",
                                              "North [meters]", "Ground truth", "Kalman filter prediction",
                                              "Dead reckoning (Kalman Gain is 0)")
                graphs.save_animation(anim, "../Results/Extended Kalman Filter", config['description'] + "_animation_with_dead_rec_cov")

        # plot vf and wz with and without noise
        graphs.plot_vf_wz_with_and_without_noise(self.yaw_vf_wz, self.yaw_vf_wz_noise)
        graphs.show_graphs("../Results/Extended Kalman Filter", "vf_and_wz_with_and_without_noise")

    def Q2_Bentzi(self):
        """
        As suggested by Bentzi
        analysis before adding noise to vf and wz, and while using dead reckoning as the ground truth
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

    def get_odometry(self, sensor_data):
        """
        Args:
            sensor_data: map from a tuple (frame number, type) where type is either ‘odometry’ or ‘sensor’.
            Odometry data is given as a map containing values for ‘r1’, ‘t’ and ‘r2’ – the first angle, the translation and the second angle in the odometry model respectively.
            Sensor data is given as a map containing:
              - ‘id’ – a list of landmark ids (starting at 1, like in the landmarks structure)
              - ‘range’ – list of ranges, in order corresponding to the ids
              - ‘bearing’ – list of bearing angles in radians, in order corresponding to the ids

        Returns:
            numpy array of of dim [num of frames X 3]
            first two components in each row are the x and y in meters
            the third component is the heading in radians
        """
        num_frames = len(sensor_data) // 2
        state = np.array([[0, 0, 0]], dtype=float).reshape(1, 3)
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
        state = self.get_odometry(sensor_data_gt)
        graphs.plot_trajectory(state, "GT trajectory from odometry data", "X [meters]", "Y [meters]")
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Extended Kalman Filter Slam", "ekf_slam_trajectory_gt")

        # add Gaussian noise to the odometry data
        variance_r1_t_r2 = [0.01 ** 2, 0.1 ** 2, 0.01 ** 2]
        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        state_noised = self.get_odometry(sensor_data_noised)
        graphs.plot_trajectory_with_noise(state, state_noised, "Trajectory from odometry before and after adding noise", "X [meters]", "Y [meters]", "Using ground truth odometry data", "Using noisy odometry data")
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Extended Kalman Filter Slam", "ekf_slam_trajectory_gt_and_noisy")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        landmark1_ind = 3
        landmark2_ind = 6
        sigma_x_y_theta = [0.056, 0.051, 0.015]  # TODO(ofekp): need to play with this and the noise/sigma graphs
        variance_r_phi = [0.1 ** 2, 0.001 ** 2]  # this was given to us in the question, sigma of the sensor data, range and bearing respectively
        ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi, landmark1_ind, landmark2_ind)
        frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)

        # draw the error for x, y and theta
        # draw the error
        e_x = mu_arr_gt[:, 0].squeeze() - mu_arr[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = mu_arr_gt[:, 1].squeeze() - mu_arr[:, 1].squeeze()  # e_y dim is [1, -1]
        e_yaw = normalize_angles_array(mu_arr_gt[:, 2].squeeze() - mu_arr[:, 2].squeeze())
        cov_graph_x = np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 0])
        cov_graph_y = np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 1])
        cov_graph_yaw = np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 2])
        graphs.plot_error((np.asarray(e_x).squeeze(), cov_graph_x), (np.asarray(e_y).squeeze(), cov_graph_y),
                          (np.asarray(e_yaw).squeeze(), cov_graph_yaw))
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Extended Kalman Filter Slam", "ekf_slam_error_and_sigma_x_y_theta")
        RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(mu_arr_gt, mu_arr, start_frame=20)
        print("Extended Kalman Filter SLAM")
        print("maxE [{:.5f}] meters".format(maxE))
        print("RMSE [{:.5f}]".format(RMSE))
        assert maxE < 0.82877
        assert RMSE < 0.30972

        graphs.plot_single_graph(mu_arr_gt[:, 0] - mu_arr[:, 0], "x-$x_n$", "frame", "error", "x-$x_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 0]))
        graphs.plot_single_graph(mu_arr_gt[:, 1] - mu_arr[:, 1], "y-$y_n$", "frame", "error", "y-$y_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 1]))
        graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:, 2] - mu_arr[:, 2]), "$\\theta-\\theta_n$",
                                 "frame", "error", "$\\theta-\\theta_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 2]))
        graphs.show_graphs()

        graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[: ,3]),
                                 "landmark 1: x-$x_n$", "frame", "error [m]", "x-$x_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 3]))
        graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:, 4]),
                                 "landmark 1: y-$y_n$", "frame", "error [m]", "y-$y_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 4]))
        graphs.show_graphs()

        graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:, 5]),
                                 "landmark 2: x-$x_n$", "frame", "error [m]", "x-$x_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 5]))
        graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:, 6]),
                                 "landmark 2: y-$y_n$", "frame", "error [m]", "y-$y_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 6]))
        graphs.show_graphs()

        # draw the error for the first landmark
        for i, landmark_ind in enumerate([landmark1_ind, landmark2_ind]):
            e_x = np.array([float(landmarks[landmark_ind][0])] * mu_arr.shape[0]).squeeze() - mu_arr[:, 3 + 2 * (landmark_ind - 1)].squeeze()  # e_x dim is [1, -1]
            e_y = np.array([float(landmarks[landmark_ind][1])] * mu_arr.shape[0]).squeeze() - mu_arr[:, 4 + 2 * (landmark_ind - 1)].squeeze()  # e_y dim is [1, -1]
            cov_graph_x = np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 3 + 2 * i])
            cov_graph_y = np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:, 4 + 2 * i])
            graphs.plot_error((np.asarray(e_x).squeeze(), cov_graph_x), (np.asarray(e_y).squeeze(), cov_graph_y))
            # graphs.show_graphs()
            graphs.show_graphs("../Results/Extended Kalman Filter Slam", "ekf_slam_error_and_sigma_landmark_ind_{}".format(landmark_ind))
            # RMSE, maxE = ExtendedKalmanFilter.calc_RMSE_maxE(np.array(landmarks[landmark_ind] * mu_arr.shape[0]).reshape(-1, 2), mu_arr[:, [3 + 2 * (landmark_ind - 1), 4 + 2 * (landmark_ind - 1)]], start_frame=20)
            # print(f"Extended Kalman Filter SLAM - landmark id [{landmark_ind}]")
            # print("maxE [{:.5f}] meters".format(maxE))
            # print("RMSE [{:.5f}]".format(RMSE))

        ax.set_xlim([-2, 12])
        ax.set_ylim([-2, 12])
        anim = animation.ArtistAnimation(fig, frames, repeat=False)
        # graphs.show_graphs()
        graphs.save_animation(anim, "../Results/Extended Kalman Filter Slam", "ekf_slam_animation")
    
    def run(self):
        # self.Q1()
        self.Q2()
        # self.Q2_Bentzi()
        # self.Q3()
        
        
