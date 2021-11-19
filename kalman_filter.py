import numpy as np
import matplotlib.pyplot as plt
from utils.plot_state import plot_state
from data_preparation import normalize_angle, normalize_angles_array


class KalmanFilter:
    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """
        e_x = X_Y_GT[:, 0] - X_Y_est[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = X_Y_GT[:, 1] - X_Y_est[:, 1].squeeze()  # e_y dim is [1, -1]
        RMSE = np.sqrt(1 / (e_x.shape[0] - 100) * np.dot(e_x[0, 100:], e_x[0, 100:].T) + np.dot(e_y[0, 100:], e_y[0, 100:].T))
        maxE = np.max(np.abs(e_x) + np.abs(e_y))
        return float(RMSE), maxE


# class ExtendedKalmanFilter:
#
#     #TODO
#     @staticmethod
#     def calc_RMSE_maxE(X_Y_GT, X_Y_est):
#         #TODO
#         return RMSE, maxE


# class ExtendedKalmanFilterSLAM:
#     def __init__(self, sigma_x_y_theta, variance_r1_t_r2, variance_r_phi):
#         self.sigma_x_y_theta = #TODO
#         self.variance_r_phi = #TODO
#         self.R_x = #TODO
#
#     def predict(self, mu_prev, sigma_prev, u, N):
#         # Perform the prediction step of the EKF
#         # u[0]=translation, u[1]=rotation1, u[2]=rotation2
#
#         delta_trans, delta_rot1, delta_rot2 = #TODO
#         theta_prev = #TODO
#
#         F = #TODO
#         G_x = #TODO
#         G = #TODO
#         V = #TODO
#
#         mu_est = #TODO
#         sigma_est = #TODO
#
#         return mu_est, sigma_est
#
#     def update(self, mu_pred, sigma_pred, z, observed_landmarks, N):
#         # Perform filter update (correction) for each odometry-observation pair read from the data file.
#         mu = mu_pred.copy()
#         sigma = sigma_pred.copy()
#         theta = mu[2]
#
#         m = len(z["id"])
#         Z = np.zeros(2 * m)
#         z_hat = np.zeros(2 * m)
#         H = None
#
#         for idx in range(m):
#             j = z["id"][idx] - 1
#             r = z["range"][idx]
#             phi = z["bearing"][idx]
#
#             mu_j_x_idx = 3 + j*2
#             mu_j_y_idx = 4 + j*2
#             Z_j_x_idx = idx*2
#             Z_j_y_idx = 1 + idx*2
#
#             if observed_landmarks[j] == False:
#                 mu[mu_j_x_idx: mu_j_y_idx + 1] = mu[0:2] + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
#                 observed_landmarks[j] = True
#
#             Z[Z_j_x_idx : Z_j_y_idx + 1] = np.array([r, phi])
#
#             delta = mu[mu_j_x_idx : mu_j_y_idx + 1] - mu[0 : 2]
#             q = delta.dot(delta)
#             z_hat[Z_j_x_idx : Z_j_y_idx + 1] = #TODO
#
#             I = np.diag(5*[1])
#             F_j = np.hstack((I[:,:3], np.zeros((5, 2*j)), I[:,3:], np.zeros((5, 2*N-2*(j+1)))))
#
#             Hi = #TODO
#
#             if H is None:
#                 H = Hi.copy()
#             else:
#                 H = np.vstack((H, Hi))
#
#         Q = #TODO
#         S = #TODO
#         K = #TODO
#
#         diff = #TODO
#         diff[1::2] = normalize_angles_array(diff[1::2])
#
#         mu = mu + K.dot(diff)
#         sigma = #TODO
#
#         mu[2] = normalize_angle(mu[2])
#
#         # Remember to normalize the bearings after subtracting!
#         # (hint: use the normalize_all_bearings function available in tools)
#
#         # Finish the correction step by computing the new mu and sigma.
#         # Normalize theta in the robot pose.
#
#
#         return mu, sigma, observed_landmarks
#
#     def run(self, sensor_data_gt, sensor_data_noised, landmarks, ax):
#         # Get the number of landmarks in the map
#         N = len(landmarks)
#
#         # Initialize belief:
#         # mu: 2N+3x1 vector representing the mean of the normal distribution
#         # The first 3 components of mu correspond to the pose of the robot,
#         # and the landmark poses (xi, yi) are stacked in ascending id order.
#         # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution
#
#
#         # init_inf_val = #TODO
#
#         mu_arr = #TODO
#         sigma_prev = #TODO
#
#         # sigma for analysis graph sigma_x_y_t + select 2 landmarks
#         # landmark1_ind=TODO
#         # landmark2_ind=TODO
#
#         Index=[0,1,2,landmark1_ind,landmark1_ind+1,landmark2_ind,landmark2_ind+1]
#         sigma_x_y_t_px1_py1_px2_py2 = sigma_prev[Index,Index].copy()
#
#         observed_landmarks = np.zeros(N, dtype=bool)
#
#         sensor_data_count = int(len(sensor_data_noised) / 2)
#         frames = []
#
#         mu_arr_gt = np.array([[0, 0, 0]])
#
#         for idx in range(sensor_data_count):
#             mu_prev = mu_arr[-1]
#
#             u = sensor_data_noised[(idx, "odometry")]
#             # predict
#             mu_pred, sigma_pred = self.predict(mu_prev, sigma_prev, u, N)
#             # update (correct)
#             mu, sigma, observed_landmarks = self.update(mu_pred, sigma_pred, sensor_data_noised[(idx, "sensor")], observed_landmarks, N)
#
#             mu_arr = np.vstack((mu_arr, mu))
#             sigma_prev = sigma.copy()
#             sigma_x_y_t_px1_py1_px2_py2 = np.vstack((sigma_x_y_t_px1_py1_px2_py2, sigma_prev[Index,Index].copy()))
#
#             delta_r1_gt = sensor_data_gt[(idx, "odometry")]["r1"]
#             delta_r2_gt = sensor_data_gt[(idx, "odometry")]["r2"]
#             delta_trans_gt = sensor_data_gt[(idx, "odometry")]["t"]
#
#             calc_x = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
#             calc_y = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)
#
#             theta = delta_r1_gt + delta_r2_gt
#
#             theta_prev = mu_arr_gt[-1,2]
#             mu_arr_gt = np.vstack((mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))
#
#             frame = plot_state(ax, mu_arr_gt, mu_arr, sigma, landmarks, observed_landmarks, sensor_data_noised[(idx, "sensor")])
#
#             frames.append(frame)
#
#         return frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2
    