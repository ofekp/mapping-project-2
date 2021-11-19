import os

import numpy as np

from data_preparation import *
from kalman_filter import KalmanFilter
import graphs
import random

random.seed(11)
np.random.seed(17)


class ProjectQuestions:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def Q1(self):
        """
        That function runs the code of question 1 of the project.
        Loads from kitti dataset, set noise to GT-gps values, and use Kalman Filter over the noised values.
        """
        # build_ENU_from_GPS_trajectory
        enu, times, yaw_vf_wz = build_GPS_trajectory(self.dataset)
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
        graphs.plot_trajectory_and_height(enu,
                                          'Trajectory of the drive - east/north in meters',
                                          'East [meters]',
                                          'North [meters]',
                                          'Up axis for the drive',
                                          'Frame Index',
                                          'Up [meters]')
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "enu_trajectory", overwrite=False)

        # the following did not work:
        # graphs.plot_single_graph(locations[: 0:2], "GPS Trajectory ENU", "East [meters]", "North [meters]", "GPS trajectory", is_scatter=True)
        # graphs.show_graphs()

        # add noise to the trajectory
        sigma_x_y = 3
        enu_noise = enu + np.concatenate((np.random.normal(0, sigma_x_y, (enu.shape[0], 2)), np.zeros((enu.shape[0], 1))), axis=1)
        graphs.plot_trajectory_comparison(enu, enu_noise)
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "comparison_enu_with_and_without_noise", overwrite=False)
        
        # KalmanFilter
        # set the initial state
        # u = np.array([enu_noise[0, 0], enu_noise[0, 1], 0, 0])
        enu_predicted = np.array([[enu_noise[0, 0], enu_noise[0, 1], 0, 0]]).reshape(1, 4)
        cov = np.diag([10.0, 10.0, 10.0, 10.0])  # TODO(ofekp): should this be [3, 3, 0, 0]?
        cov_graph_x = [float(cov[0, 0])]
        cov_graph_y = [float(cov[1, 1])]
        # we set R according to the suggestion in the presentation form the class
        delta_t_0 = times[1] - times[0]
        sigma_n = 2  # TODO(ofekp): figure out if this is the only value that should be heuristically set
        R_t = np.diag([0, 0, 1, 1]) * delta_t_0 * sigma_n
        Q_t = np.diag([sigma_x_y, sigma_x_y])
        # since we only measure the position we initialize C_t as follows
        C_t = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]]).reshape(2, 4)
        for i in range(1, enu_noise.shape[0]):
            # state and covariance prediction
            delta_t = times[i] - times[i - 1]
            A = np.matrix([[1.0, 0.0, delta_t, 0.0],
                           [0.0, 1.0, 0.0, delta_t],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
            new_state_prediction = np.dot(A, enu_predicted[-1, :].T).reshape(4, 1)
            cov = np.dot(np.dot(A, cov), A.T) + R_t

            # Kalman gain
            K_t = np.dot(np.dot(cov, C_t.T), np.linalg.pinv(np.dot(np.dot(C_t, cov), C_t.T) + Q_t))

            # correction
            z_t = enu_noise[i, 0:2]
            new_state = new_state_prediction + np.dot(K_t, (z_t.reshape(2, 1) - np.dot(C_t, new_state_prediction)))
            cov = np.dot((np.identity(4) - np.dot(K_t, C_t)), cov)

            # update the predicted path
            enu_predicted = np.concatenate([enu_predicted, new_state.reshape(1, 4)], axis=0)
            cov_graph_x.append(float(cov[0, 0]))
            cov_graph_y.append(float(cov[1, 1]))

        graphs.plot_trajectory_comparison(enu, enu_noise, enu_predicted=enu_predicted[:, 0:2])
        # graphs.show_graphs()
        graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_const_vel_predicted_path", overwrite=False)

        RMSE, maxE = KalmanFilter.calc_RMSE_maxE(enu, enu_predicted)
        assert maxE < 6.95
        print("Kalman Filter - Constant Velocity")
        print("maxE [{}]".format(maxE))
        print("RMSE [{}]".format(RMSE))

        # let's draw the error
        e_x = enu[:, 0] - enu_predicted[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = enu[:, 1] - enu_predicted[:, 1].squeeze()  # e_y dim is [1, -1]
        graphs.plot_error(np.asarray(e_x).squeeze(), np.asarray(e_y).squeeze(), cov_graph_x, cov_graph_y)
        graphs.show_graphs()
        # graphs.show_graphs("../Results/Kalman Filter", "kalman_filter_error_comparison_x_y")


    def Q2(self):
        # sigma_samples = 
        
        # sigma_vf, sigma_omega = 
        
        # build_LLA_GPS_trajectory
        
        # add_gaussian_noise to u and measurments (locations_gt[:,i], sigma_samples[i])
            
        # ekf = ExtendedKalmanFilter(sigma_samples, sigma_vf, sigma_omega)
        # locations_ekf, sigma_x_xy_yx_y_t = ekf.run(locations_noised, times, yaw_vf_wz_noised, do_only_predict=False)
        
        # RMSE, maxE = ekf.calc_RMSE_maxE(locations_gt, locations_ekf)
 
        # build_animation
        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")
        pass
        
    # def Q3(self):
    #     landmarks = self.dataset.load_landmarks()
    #     sensor_data_gt = self.dataset.load_sensor_data()
    #
    #     sigma_x_y_theta = #TODO
    #     variance_r1_t_r2 = #TODO
    #
    #     variance_r_phi = #TODO
    #
    #     sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
    #
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #
    #     ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)
    #
    #     frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)
    #
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
    #     from matplotlib import animation
    #     ani = animation.ArtistAnimation(fig, frames, repeat=False)
    #     graphs.show_graphs()
    #     # ani.save('im.mp4', metadata={'artist':'me'})
    
    def run(self):
        self.Q1()
        # self.Q3()
        
        
