from accReader import *
import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from dataGenerator import *
from helper import *
import time


class KalmanFilter:
    def __init__(self):
        # feed in 2d simulation data/
        data = genData2d()
        self.z = data[0]

        # feed in real positional data.
        self.positions_X = data[1]
        self.positions_Y = data[2]
        self.positions_Z = data[3]

        # keep track of all predictedPositions
        self.predictedX = []

        # Initial Conditions
        # same timestep as defined in dataGenerator.
        self.t = timesteps

        # Process / Estimation Errors
        # make these more realistic, this is just for development purposes.
        # x, y and z errors.
        self.error_est_x = 1
        self.error_est_v_x = 1
        self.error_est_a_x = 1
        self.error_est_y = 1
        self.error_est_v_y = 1
        self.error_est_a_y = 1
        self.error_est_z = 1
        self.error_est_v_z = 1
        self.error_est_a_z = 1

        # angle errors.
        self.error_est_angle = 1
        self.error_est_angle_dot = 1
        self.error_est_roll = 1
        self.error_est_roll_dot = 1
        self.error_est_pitch = 1
        self.error_est_pitch_dot = 1
        self.sigma_a = 1

        # Noise values
        # dummy values, find better vals.
        self.noiseX = 1
        self.noiseY = 1
        self.noiseZ = 1
        self.noiseA_x = 1
        self.noiseA_y = 1
        self.noiseA_z = 1

        self.noiseYaw = 1
        self.noiseYaw_dot = 1
        self.noiseRoll = 1
        self.noiseRoll_dot = 1
        self.noisePitch = 1
        self.noisePitch_dot = 1

        # Setup numpy arrays for use throughout
        self.X = numpy.zeros(shape=9)
        self.W = numpy.zeros(shape=6)

        # Create H matrices
        self.H = numpy.zeros(shape=(6,9))
        self.H_angle = numpy.zeros(shape=(6,6))

        # Create the G matrices.
        self.G_dotted = numpy.zeros(shape=(9,9))
        self.G_angle_dotted = numpy.zeros(shape=(6,6))

    def prediction2d(self, X):
        """Predict position of drone using taylor expansion to 2nd order as integral approximation."""
        X = self.A.dot(X)
        return X


    def predictAngle(self, W):
        """Predict angle from taylor expansion to 1st order as integral approximation."""
        W = self.B.dot(W)
        return W

    def covariance2d(self, P_x, P_y, P_z, P_x_dot, P_y_dot, P_z_dot, P_x_ddot, P_y_ddot, P_z_ddot):
        """Create covariance matrix of x, y and z."""
        cov_matrix = numpy.array([[P_x, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, P_y, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, P_z, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, P_x_dot, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, P_y_dot, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, P_z_dot, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, P_x_ddot, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, P_y_ddot, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, P_z_ddot]])
        return cov_matrix


    def covarianceAngle(self, P_yaw, P_yaw_dot, P_roll, P_roll_dot, P_pitch, P_pitch_dot):
        """Create covariance matrix of angles."""
        cov_matrix = numpy.array([[P_yaw, 0, 0, 0, 0, 0],
                                  [0, P_yaw_dot, 0, 0, 0, 0],
                                  [0, 0, P_roll, 0, 0, 0],
                                  [0, 0, 0, P_roll_dot, 0, 0],
                                  [0, 0, 0, 0, P_pitch, 0],
                                  [0, 0, 0, 0, 0, P_pitch_dot]])
        return cov_matrix


    def preSimulationSetup(self):
        # Initial Estimation Covariance Matrix - 2d case
        self.P = self.covariance2d(self.error_est_x, self.error_est_y, self.error_est_z, self.error_est_v_x,
                              self.error_est_v_y, self.error_est_v_z, self.error_est_a_x, self.error_est_a_y,
                              self.error_est_a_z)
        self.P_angle = self.covarianceAngle(self.error_est_angle, self.error_est_angle_dot, self.error_est_roll,
                                       self.error_est_roll_dot, self.error_est_pitch, self.error_est_pitch_dot)

        # A and B matrices used to take state matrix to the next timestep.
        self.A = numpy.array([[1, 0, 0, self.t, 0, 0, (self.t ** 2) / 2, 0, 0],
                         [0, 1, 0, 0, self.t, 0, 0, (self.t ** 2) / 2, 0],
                         [0, 0, 1, 0, 0, self.t, 0, 0, (self.t ** 2) / 2],
                         [0, 0, 0, 1, 0, 0, self.t, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, self.t, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, self.t],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.B = numpy.array([[1, self.t, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, self.t, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, self.t],
                         [0, 0, 0, 0, 0, 1]])

        # initial state vectors
        # x looks like [x, y, z, xdot, ydot, zdot, xddot, yddot, zddot]
        # y looks like [yaw, yawdot, roll, rolldot, pitch, pitchdot]

        # Transform initial input into drone reference frame.
        (x_init, y_init, z_init) = transformAngles(self.z[0][3], self.z[0][4], self.z[0][5], self.z[0][6], self.z[0][8],
                                                   self.z[0][10], True)
        self.X = numpy.array([self.z[0][0], self.z[0][1], self.z[0][2], 0, 0, 0, x_init, y_init, z_init])
        self.W = numpy.array([self.z[0][6], self.z[0][7], self.z[0][8], self.z[0][9], self.z[0][10], self.z[0][11]])




        # Noise matrices for position and angle.
        self.R = numpy.array([[self.noiseX, 0, 0, 0, 0, 0],
                         [0, self.noiseA_x, 0, 0, 0, 0],
                         [0, 0, self.noiseY, 0, 0, 0],
                         [0, 0, 0, self.noiseA_y, 0, 0],
                         [0, 0, 0, 0, self.noiseZ, 0],
                         [0, 0, 0, 0, 0, self.noiseA_z]])
        self.R_angle = numpy.array([[self.noiseYaw, 0, 0, 0, 0, 0],
                               [0, self.noiseYaw_dot, 0, 0, 0, 0],
                               [0, 0, self.noiseRoll, 0, 0, 0],
                               [0, 0, 0, self.noiseRoll_dot, 0, 0],
                               [0, 0, 0, 0, self.noisePitch, 0],
                               [0, 0, 0, 0, 0, self.noisePitch_dot]])

        self.K_angle_identity = numpy.identity(6)
        self.K_identity = numpy.identity(9)

    def runSimulation(self):
        # Loop through data and do kalman filter iteration for each data point.
        for data in self.z[1:]:
            # Choose correct H matrix used for converting input data into same shape as state matrix.
            if (data[0] == 0):
                self.H[:] = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1]]
            else:
                self.H[:] = [[1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1]]
            if (data[6] == 0):
                self.H_angle[:] = [[0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1]]
            else:
                self.H_angle[:] = [[1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]]

            # 	Predict angle
            self.W = self.predictAngle(self.W)

            # todo - split this up into two matrixes that can be dotted together if poss. Final result will be this though.
            self.G_angle_dotted = [[self.t**4, self.t**3, 0, 0, 0, 0],
                                   [self.t**3, self.t**2, 0, 0, 0, 0],
                                   [0, 0, self.t**4, self.t**3, 0, 0],
                                   [0, 0, self.t**3, self.t**2, 0, 0],
                                   [0, 0, 0, 0, self.t**4, self.t**3],
                                   [0, 0, 0, 0, self.t**3, self.t**2]]
            Q = self.G_angle_dotted * self.sigma_a ** 2
            self.P_angle = self.B.dot(self.P_angle).dot(self.B.T) + Q

            # Calculate Kalman gain
            S = self.H_angle.dot(self.P_angle).dot(self.H_angle.T) + self.R_angle
            K = self.P_angle.dot(self.H_angle.T).dot(inv(S))
            Y = data[6:] - self.H_angle.dot(self.W)
            # Update state matrix
            self.W = self.W + K.dot(Y)
            # Update covariance matrix.
            self.P_angle = (self.K_angle_identity - K.dot(self.H_angle)).dot(self.P_angle)
            # Obtain current angles.
            yaw = self.W[0]
            roll = self.W[2]
            pitch = self.W[4]

            # Predict positions
            self.X = self.prediction2d(self.X)

            # 	Handling X state vector
            # todo - split this up into two matrixes that can be dotted together if poss. Final result will be this though.
            self.G_dotted = [[(self.t**4)/4, 0, 0, (self.t**3)/2, 0, 0, (self.t**2)/2, 0, 0],
                             [0 ,(self.t**4)/4, 0, 0, (self.t**3)/2, 0, 0, (self.t**2)/2, 0],
                             [0, 0 ,(self.t**4)/4, 0, 0, (self.t**3)/2, 0, 0, (self.t**2)/2],
                             [(self.t**3)/2, 0, 0, self.t**2, 0, 0, self.t, 0, 0],
                             [0, (self.t**3)/2, 0, 0, self.t**2, 0, 0, self.t, 0],
                             [0, 0, (self.t**3)/2, 0, 0, self.t**2, 0, 0, self.t],
                             [(self.t**2)/2, 0, 0, self.t, 0, 0, 1, 0, 0],
                             [0, (self.t**2)/2, 0, 0, self.t, 0, 0, 1, 0],
                             [0, 0, (self.t**2)/2, 0, 0, self.t, 0, 0, 1]]

            # calculate Kalman gain.
            Q = self.G_dotted * self.sigma_a ** 2
            self.P = self.A.dot(self.P).dot(self.A.T) + Q
            S = self.H.dot(self.P).dot(self.H.T) + self.R
            K = self.P.dot(self.H.T).dot(inv(S))

            # Transform accerelation vector back to inerial reference frame.
            (data[3], data[4], data[5]) = transformAngles(data[3], data[4], data[5], yaw, roll, pitch, True)

            Y = data[:6] - self.H.dot(self.X)

            # Update state vector
            self.X = self.X + K.dot(Y)

            # Save the result for comparison later
            self.predictedX.append(self.X)

            # Update the covariance matrix
            self.P = (self.K_identity - K.dot(self.H)).dot(self.P)

    def postSimulationAnalysis(self):

        predictedPositionsX = [x[0] for x in self.predictedX]
        predictedPositionsY = [x[1] for x in self.predictedX]
        predictedPositionsZ = [x[2] for x in self.predictedX]

        # find out precision of kalman filter
        precision = precisionAnalysis(predictedPositionsX, predictedPositionsY, predictedPositionsZ, self.positions_X,
                                      self.positions_Y, self.positions_Z)
        print "Precision is ", precision, " m"

        # print the results.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(predictedPositionsX, predictedPositionsY, predictedPositionsZ, 'o', label = "kalman")
        ax.plot(self.positions_X, self.positions_Y, self.positions_Z, label="real")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.text2D(0.1, 0.95, "Kalman Filter predictions and true positions", transform=ax.transAxes)
        ax.legend()
        pyplot.show()


kf = KalmanFilter()
kf.preSimulationSetup()
kf.runSimulation()
kf.postSimulationAnalysis()
