import numpy as np
import matplotlib.pyplot as plt
from utils.plot_state import plot_state
from data_preparation import normalize_angle, normalize_angles_array


class KalmanFilter:
    """
    class for the implementation of Kalman filter
    """

    def __init__(self, enu_noise, times, sigma_xy, sigma_n, is_dead_reckoning):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
        """
        self.enu_noise = enu_noise
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_n = sigma_n
        self.is_dead_reckoning = is_dead_reckoning

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        Calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """
        e_x = X_Y_GT[:, 0].squeeze() - X_Y_est[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = X_Y_GT[:, 1].squeeze() - X_Y_est[:, 1].squeeze()  # e_y dim is [1, -1]
        e_x = e_x[100:]
        e_y = e_y[100:]
        # RMSE = np.sqrt(1 / (e_x.shape[0] - 100) * (np.dot(e_x, e_x.T) + np.dot(e_y, e_y.T)))
        RMSE = np.sqrt((1 / (e_x.shape[0] - 100)) * (np.dot(e_x, e_x.T) + np.dot(e_y, e_y.T)))
        maxE = np.max(np.abs(e_x) + np.abs(e_y))
        return float(RMSE), maxE

    def run(self):
        """
        Runs the Kalman filter
        """
        delta_t_0 = self.times[1] - self.times[0]
        enu_kf = np.array([[self.enu_noise[0, 0], self.enu_noise[0, 1], 0, 0]]).reshape(1, 4)
        cov = np.diag([self.sigma_xy ** 2, self.sigma_xy ** 2, 0, 0])  # this has little effect of the final trajectory
        covs = np.array([[cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]]]).reshape(1, 4)
        # we set R according to the suggestion in the presentation form the class
        R_t = np.diag([0.0, 0.0, 1.0, 1.0]) * delta_t_0 * (self.sigma_n ** 2)
        # Q_t is given to us in the question since sigma_xy is given
        Q_t = np.diag([self.sigma_xy ** 2, self.sigma_xy ** 2])
        # since we only measure the position (and not the velocity) we initialize C_t as follows
        C_t = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]]).reshape(2, 4)
        elapsed_time = delta_t_0
        for i in range(1, self.enu_noise.shape[0]):
            # state and covariance prediction
            delta_t = self.times[i] - self.times[i - 1]
            A = np.array([[1.0, 0.0, delta_t, 0.0],
                          [0.0, 1.0, 0.0, delta_t],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]).reshape(4, 4)
            new_state_prediction = np.dot(A, enu_kf[-1, :].T).reshape(4, 1)
            cov = np.dot(np.dot(A, cov), A.T) + R_t

            # Kalman gain
            elapsed_time += delta_t
            if self.is_dead_reckoning and elapsed_time > 5.0:
                K_t = np.zeros((4, 2), dtype=float)
            else:
                K_t = np.dot(np.dot(cov, C_t.T), np.linalg.pinv(np.dot(np.dot(C_t, cov), C_t.T) + Q_t))

            # correction
            z_t = self.enu_noise[i, 0:2]
            new_state = new_state_prediction + np.dot(K_t, (z_t.reshape(2, 1) - np.dot(C_t, new_state_prediction)))
            cov = np.dot((np.identity(4) - np.dot(K_t, C_t)), cov)

            # update the predicted path
            enu_kf = np.concatenate([enu_kf, new_state.reshape(1, 4)], axis=0)
            covs = np.concatenate([covs, np.array([cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]]).reshape(1, 4)], axis=0)
        return enu_kf, covs


class ExtendedKalmanFilter:
    """
    class for the implementation of the extended Kalman filter
    """
    def __init__(self, enu_noise, yaw_vf_wz, times, sigma_xy, sigma_theta, sigma_vf, sigma_wz, k, is_dead_reckoning, dead_reckoning_start_sec=5.0):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            yaw_vf_wz: the yaw, forward velocity and angular change rate to be used (either non noisy or noisy, depending on the question)
            sigma_theta: sigma of the heading
            sigma_vf: sigma of the forward velocity
            sigma_wz: sigma of the angular change rate
            k: hyper parameter to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
            dead_reckoning_start_sec: from what second do we start applying dead reckoning, used for experimentation only
        """
        self.enu_noise = enu_noise
        self.yaw_vf_wz = yaw_vf_wz
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_theta = sigma_theta
        self.sigma_vf = sigma_vf
        self.sigma_wz = sigma_wz
        self.k = k
        self.is_dead_reckoning = is_dead_reckoning
        self.dead_reckoning_start_sec = dead_reckoning_start_sec

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est, start_frame=100):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """
        e_x = X_Y_GT[:, 0].squeeze() - X_Y_est[:, 0].squeeze()  # e_x dim is [1, -1]
        e_y = X_Y_GT[:, 1].squeeze() - X_Y_est[:, 1].squeeze()  # e_y dim is [1, -1]
        e_x = e_x[start_frame:]
        e_y = e_y[start_frame:]
        RMSE = np.sqrt((1 / (e_x.shape[0] - start_frame)) * (np.dot(e_x, e_x.T) + np.dot(e_y, e_y.T)))
        maxE = np.max(np.abs(e_x) + np.abs(e_y))
        return float(RMSE), maxE

    def run(self):
        """
        Runs the extended Kalman filter
        """
        delta_t_0 = self.times[1] - self.times[0]
        state_kf = np.array([[self.enu_noise[0, 0], self.enu_noise[0, 1], self.yaw_vf_wz[0, 0]]]).reshape(1, 3)
        cov = np.diag([self.k * (self.sigma_xy ** 2), self.k * (self.sigma_xy ** 2), self.k * (self.sigma_theta ** 2)])
        covs = np.hstack(cov).reshape(1, 9)
        # we set R according to the suggestion in the presentation form the class
        R_t_tilde = np.diag([self.sigma_vf ** 2, self.sigma_wz ** 2])
        Q_t = np.diag([self.sigma_xy ** 2, self.sigma_xy ** 2])
        # since we only measure the position we initialize C_t as follows
        H_t = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0]]).reshape(2, 3)
        elapsed_time = delta_t_0
        for i in range(1, self.enu_noise.shape[0]):
            # state and covariance prediction
            delta_t = self.times[i] - self.times[i - 1]
            v_t = float(self.yaw_vf_wz[i, 1])
            w_t = float(self.yaw_vf_wz[i, 2])
            factor = v_t / w_t
            g = np.array([
                factor * (np.sin(state_kf[-1, 2] + delta_t * w_t) - np.sin(state_kf[-1, 2])),
                factor * (np.cos(state_kf[-1, 2]) - np.cos(state_kf[-1, 2] + delta_t * w_t)),
                delta_t * w_t
            ])
            new_state_prediction = state_kf[-1, :] + g

            factor2 = v_t / (w_t ** 2)
            V_t = np.array([
                [(1 / w_t) * (np.sin(state_kf[-1, 2] + delta_t * w_t) - np.sin(state_kf[-1, 2])), factor2 * (np.sin(state_kf[-1, 2]) - np.sin(state_kf[-1, 2] + delta_t * w_t)) + delta_t * factor * np.cos(state_kf[-1, 2] + delta_t * w_t)],
                [(1 / w_t) * (np.cos(state_kf[-1, 2]) - np.cos(state_kf[-1, 2] + delta_t * w_t)), factor2 * (np.cos(state_kf[-1, 2] + delta_t * w_t) - np.cos(state_kf[-1, 2])) + delta_t * factor * np.sin(state_kf[-1, 2] + delta_t * w_t)],
                [0, delta_t]
            ])
            G_t = np.array([
                [1, 0, factor * (np.cos(state_kf[-1, 2] + delta_t * w_t) - np.cos(state_kf[-1, 2]))],
                [0, 1, factor * (np.sin(state_kf[-1, 2] + delta_t * w_t) - np.sin(state_kf[-1, 2]))],
                [0, 0, 1],
            ])
            cov = np.dot(np.dot(G_t, cov), G_t.T) + np.dot(np.dot(V_t, R_t_tilde), V_t.T)

            # Kalman gain
            elapsed_time += delta_t
            if self.is_dead_reckoning and elapsed_time >= self.dead_reckoning_start_sec:
                K_t = np.zeros((3, 2), dtype=float)
            else:
                K_t = np.dot(np.dot(cov, H_t.T), np.linalg.pinv(np.dot(np.dot(H_t, cov), H_t.T) + Q_t))

            # correction
            z_t = self.enu_noise[i, 0:2]
            new_state = new_state_prediction.reshape(3, 1) + np.dot(K_t, (z_t.reshape(2, 1) - np.dot(H_t, new_state_prediction).reshape(2, 1)))
            cov = np.dot((np.identity(3) - np.dot(K_t, H_t)), cov)

            # update the predicted path
            state_kf = np.concatenate([state_kf, new_state.reshape(1, 3)], axis=0)
            covs = np.concatenate([covs, np.hstack(cov).reshape(1, 9)], axis=0)
        return state_kf, covs


class ExtendedKalmanFilterSLAM:
    """
    class for the implementation of Kalman filter SLAM algorithm
    """

    def __init__(self, variance_x_y_theta, variance_r1_t_r2, variance_r_phi, landmark1_ind, landmark2_ind):
        """
        Args:
            variance_x_y_theta: variance in x, y and theta respectively
            variance_r1_t_r2: variance in rotation1, translation and rotation2 respectively
            variance_r_phi: variance in the range and bearing
            landmark1_ind: first pick of landmark id, used for analysis
            landmark2_ind: second pick of landmark id, used for analysis
        """
        self.variance_x_y_theta = variance_x_y_theta
        self.variance_r_phi = variance_r_phi
        self.variance_r1_t_r2 = variance_r1_t_r2
        self.R_t_tilde = np.diag(variance_r1_t_r2)
        self.landmark1_ind = landmark1_ind
        self.landmark2_ind = landmark2_ind

    def predict(self, mu_prev, sigma_prev, u, N):
        """
        Perform the prediction step of the EKF SLAM algorithm
        Args:
            mu_prev: previous state
            sigma_prev: previous covariance matrix
            u: the current rotation1, translation and rotation2 describing the motion in the current frame
            N: the total number of landmarks in the whole sequence
        """
        delta_trans, delta_rot1, delta_rot2 = u['t'], u['r1'], u['r2']
        theta_prev = mu_prev[2]

        F = np.hstack((np.identity(3), np.zeros((3, 2 * N))))
        G_x = np.array([
            [0, 0, -delta_trans * np.sin(theta_prev + delta_rot1)],
            [0, 0, delta_trans * np.cos(theta_prev + delta_rot1)],
            [0, 0, 0]
        ]).reshape(3, 3)
        G_t = np.identity(3 + 2 * N) + np.dot(np.dot(F.T, G_x), F)

        V_t = np.array([
            [-delta_trans * np.sin(theta_prev + delta_rot1), np.cos(theta_prev + delta_rot1), 0],
            [delta_trans * np.cos(theta_prev + delta_rot1), np.sin(theta_prev + delta_rot1), 0],
            [1, 0, 1]
        ]).reshape(3, 3)

        t = np.array([
            delta_trans * np.cos(theta_prev + delta_rot1),
            delta_trans * np.sin(theta_prev + delta_rot1),
            delta_rot1 + delta_rot2
        ]).squeeze()
        mu_est = mu_prev + np.dot(F.T, t)
        mu_est[2] = normalize_angle(mu_est[2])
        sigma_est = np.dot(np.dot(G_t, sigma_prev), G_t.T) + np.dot(np.dot(F.T, np.dot(np.dot(V_t, self.R_t_tilde), V_t.T)), F)

        return mu_est, sigma_est

    def update(self, mu_pred, sigma_pred, z, observed_landmarks, N):
        """
        Perform filter update (correction) for each odometry-observation pair read from the data file.
        Args:
            mu_pred: state as calculated by the prediction step
            sigma_pred: covariance matrix as calculated by the prediction step
            z: sensor data, containing the current frame landmarks with range and bearing for each one
            observed_landmarks: array of ids of landmarks we've already seen before
            N: the total number of landmarks in the whole sequence
        """
        mu = mu_pred.copy()
        sigma = sigma_pred.copy()
        theta = mu[2]

        m = len(z["id"])
        Z = np.zeros(2 * m)
        z_hat = np.zeros(2 * m)
        H = None

        for idx in range(m):
            j = z["id"][idx] - 1
            r = z["range"][idx]
            phi = z["bearing"][idx]

            mu_j_x_idx = 3 + j*2
            mu_j_y_idx = 4 + j*2
            Z_j_x_idx = idx*2
            Z_j_y_idx = 1 + idx*2

            if not observed_landmarks[j]:
                mu[mu_j_x_idx: mu_j_y_idx + 1] = mu[0:2] + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
                observed_landmarks[j] = True

            Z[Z_j_x_idx : Z_j_y_idx + 1] = np.array([r, phi])

            delta = mu[mu_j_x_idx : mu_j_y_idx + 1] - mu[0 : 2]
            q = delta.dot(delta)
            z_hat[Z_j_x_idx : Z_j_y_idx + 1] = np.array([np.sqrt(q), np.arctan2(delta[1], delta[0]) - theta])

            I = np.diag(5*[1])
            F_j = np.hstack((I[:,:3], np.zeros((5, 2*j)), I[:,3:], np.zeros((5, 2*N-2*(j+1)))))

            q_sqrt = np.sqrt(q)
            Hi = (1 / q) * np.dot(np.array([[-q_sqrt * delta[0], -q_sqrt * delta[1], 0, q_sqrt * delta[0], q_sqrt * delta[1]], [delta[1], -delta[0], -q, -delta[1], delta[0]]]).reshape(2, -1), F_j)

            if H is None:
                H = Hi.copy()
            else:
                H = np.vstack((H, Hi))

            # standard algorithm for comparison, for further analysis only
            # Q_t = np.diag([0.3, 0.003])  # [2m, 2m]
            # Ki = np.dot(np.dot(sigma, Hi.T), np.linalg.pinv(np.dot(np.dot(Hi, sigma), Hi.T) + Q_t))
            # diff = Z[Z_j_x_idx : Z_j_y_idx + 1] - z_hat[Z_j_x_idx : Z_j_y_idx + 1]
            # diff[1] = normalize_angle(diff[1])
            # mu = np.asarray(mu + Ki.dot(diff).squeeze()).squeeze()
            # sigma = np.dot(np.identity(3 + 2 * N) - np.dot(Ki, Hi), sigma)
            # mu[2] = normalize_angle(mu[2])

        Q_t = np.diag([0.3, 0.003] * m)  # [2m, 2m]
        K = np.dot(np.dot(sigma, H.T), np.linalg.pinv(np.dot(np.dot(H, sigma), H.T) + Q_t))  # K dim is [3+2N, m]

        diff = Z - z_hat
        diff[1::2] = normalize_angles_array(diff[1::2])

        mu = np.asarray(mu + K.dot(diff)).squeeze()
        sigma = np.dot(np.identity(3 + 2 * N) - np.dot(K, H), sigma)

        mu[2] = normalize_angle(mu[2])

        # Remember to normalize the bearings after subtracting!
        # (hint: use the normalize_all_bearings function available in tools)

        # Finish the correction step by computing the new mu and sigma.
        # Normalize theta in the robot pose.

        return mu, sigma, observed_landmarks

    def run(self, sensor_data_gt, sensor_data_noised, landmarks, ax):
        """
        Runs the extended Kalman filter SLAM algorithm
        Args:
            sensor_data_gt: will be used only to build the ground truth trajectory from the non noisy odometry data
            sensor_data_noised: the noisy odometry data and the sensor (landmarks range and bearing) data
            landmarks: list of ids of all the landmarks in the entire sequence
            ax: will be used to build the final animation
        """
        # Get the number of landmarks in the map
        N = len(landmarks)

        # Initialize belief:
        # mu: 2N+3x1 vector representing the mean of the normal distribution
        # The first 3 components of mu correspond to the pose of the robot,
        # and the landmark poses (xi, yi) are stacked in ascending id order.
        # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution

        init_inf_val = 200.0

        mu_arr = np.array([[0] * (3 + 2 * N)], dtype=float).reshape(1, 3 + 2 * N)  # x, y, theta, then (x,y) for each possible landmark
        sigma_prev = np.diag(self.variance_x_y_theta + [init_inf_val] * 2 * N)

        # sigma for analysis graph sigma_x_y_t + select 2 landmarks
        Index = [0, 1, 2, self.landmark1_ind, self.landmark1_ind + 1, self.landmark2_ind, self.landmark2_ind + 1]
        sigma_x_y_t_px1_py1_px2_py2 = sigma_prev[Index, Index].copy()

        observed_landmarks = np.zeros(N, dtype=bool)

        sensor_data_count = int(len(sensor_data_noised) / 2)
        frames = []

        mu_arr_gt = np.array([[0, 0, 0]])

        for idx in range(sensor_data_count):
            mu_prev = mu_arr[-1, :]

            u = sensor_data_noised[(idx, "odometry")]
            # predict
            mu_pred, sigma_pred = self.predict(mu_prev, sigma_prev, u, N)
            # update (correct)
            mu, sigma, observed_landmarks = self.update(mu_pred, sigma_pred, sensor_data_noised[(idx, "sensor")], observed_landmarks, N)

            mu_arr = np.vstack((mu_arr, mu))
            sigma_prev = sigma.copy()
            sigma_x_y_t_px1_py1_px2_py2 = np.vstack((sigma_x_y_t_px1_py1_px2_py2, sigma_prev[Index,Index].copy()))

            delta_r1_gt = sensor_data_gt[(idx, "odometry")]["r1"]
            delta_r2_gt = sensor_data_gt[(idx, "odometry")]["r2"]
            delta_trans_gt = sensor_data_gt[(idx, "odometry")]["t"]

            calc_x = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
            calc_y = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)

            theta = delta_r1_gt + delta_r2_gt

            theta_prev = mu_arr_gt[-1,2]
            mu_arr_gt = np.vstack((mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))

            frame = plot_state(ax, mu_arr_gt, mu_arr, sigma, landmarks, observed_landmarks, sensor_data_noised[(idx, "sensor")])

            frames.append(frame)

        return frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2
