from __future__ import division
from accReader import *
import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from dataGenerator import *
from helper import *
import csv
import math
import pandas as pd
import time
import helper

# pump up the frequency


class KalmanFilter:
    def __init__(self, timesteps, gradient, cutoff, accNoise, rtkNoise, errAngle, errAngleRate, errRTK, errAcc):
        self.simMode = False
        # feed in 3d simulation data/
        if (self.simMode):
            # data = genData2d(timesteps)
            # data = genRandomFlightData(timesteps, gradient, cutoff, rtkNoise)

            # file = open('dataRotatingArmSim.csv')
            file = open('groundData.csv')
            df = pd.DataFrame.from_csv(file, index_col=0)
            # print df.head()
            # using groundData
            times, timeIntervals = helper.getTimes(df)
            self.times = timeIntervals
            # using circleSim
            # times = numpy.arange(0, 100, 0.0065)
            # self.times = [0.0065] * len(times)
            data = circleSim(times)
            # self.times = timeIntervals
            # print numpy.shape(data[0])
            self.z = data[0]
            # print self.z[:5]
            self.positions_X = data[1]
            self.positions_Y = data[2]
            self.positions_Z = data[3]
            # print self.z[:10]

            # Initial Conditions
            # same timestep as defined in dataGenerator.
            self.t = 0

        # feed in actual data
        else:
            file = open('groundData.csv')
            # file = open('dataRotatingArmSim.csv')
            df = pd.DataFrame.from_csv(file, index_col=0)
            # print df.head()
            # df = pd.DataFrame.from_csv(file)
            self.z = df.values.tolist()[:]

            # print self.z[:5]
            self.t = self.z[0][-1]
            # print self.t


        

        # keep track of all predictedPositions
        self.predictedX = []
        self.covariances = []
        self.predictedYaw = []


        # Process / Estimation Errors
        # make these more realistic, this is just for development purposes.
        # x, y and z errors.

        # todo - warm start P
        errorPos = 10
        errorAcc = 10
        errorVel = 1

        # errorPos = 0.0001
        # errorAcc = 0.00
        self.error_est_x = errorPos
        self.error_est_v_x = errorVel
        self.error_est_a_x = errorAcc
        self.error_est_y = errorPos
        self.error_est_v_y = errorVel
        self.error_est_a_y = errorAcc
        self.error_est_z = errorPos
        self.error_est_v_z = errorVel
        self.error_est_a_z = errorAcc

        # angle errors.
        angleError = errAngle
        angleDot = errAngleRate
        self.error_est_angle = angleError
        self.error_est_angle_dot = angleDot
        self.error_est_roll = angleError
        self.error_est_roll_dot = angleDot
        self.error_est_pitch = angleError
        self.error_est_pitch_dot = angleDot

        self.sigma_a = 0.1

        if self.simMode == False:
            noisePos = errRTK
            noiseAcc = errAcc
            # noisePos = 0.0001
            # noiseAcc = 1

        else:
            noisePos = 0.0001
            noiseAcc = 1

        # Noise values
        # dummy values, find better vals.
        self.noiseX = noisePos
        self.noiseY = noisePos
        self.noiseZ = noisePos
        self.noiseA_x = noiseAcc
        self.noiseA_y = noiseAcc
        self.noiseA_z = noiseAcc


        # z = .1
        # w = 0.01
        self.noiseYaw = angleError
        self.noiseYaw_dot = angleDot
        self.noiseRoll = angleError
        self.noiseRoll_dot = angleDot
        self.noisePitch = angleError
        self.noisePitch_dot = angleDot

        # Setup numpy arrays for use throughout
        self.X = numpy.zeros(shape=9)
        self.W = numpy.zeros(shape=6)

        # Create H matrices
        self.H = numpy.zeros(shape=(6,9))
        self.H_angle = numpy.zeros(shape=(6,6))

        # Create the G matrices.
        self.G_dotted = numpy.zeros(shape=(9,9))
        self.G_angle_dotted = numpy.zeros(shape=(6,6))

    def prediction2d(self, X, t):
        """Predict position of drone using taylor expansion to 2nd order as integral approximation."""
        self.A = self.createA(t)
        X = self.A.dot(X)
        return X


    def predictAngle(self, W, t):
        # todo - add time as in input here!
        """Predict angle from taylor expansion to 1st order as integral approximation."""
        self.B = self.createB(t)
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

    def createA(self, t):
        return numpy.array([[1, 0, 0, t, 0, 0, (t ** 2) / 2, 0, 0],
                            [0, 1, 0, 0, t, 0, 0, (t ** 2) / 2, 0],
                            [0, 0, 1, 0, 0, t, 0, 0, (t ** 2) / 2],
                            [0, 0, 0, 1, 0, 0, t, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, t, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, t],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def createB(self, t):

        return numpy.array([[1, t, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, t, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, t],
                              [0, 0, 0, 0, 0, 1]])



    def preSimulationSetup(self):
        # Initial Estimation Covariance Matrix - 2d case
        self.P = self.covariance2d(self.error_est_x, self.error_est_y, self.error_est_z, self.error_est_v_x,
                              self.error_est_v_y, self.error_est_v_z, self.error_est_a_x, self.error_est_a_y,
                              self.error_est_a_z)
        self.P_angle = self.covarianceAngle(self.error_est_angle, self.error_est_angle_dot, self.error_est_roll,
                                       self.error_est_roll_dot, self.error_est_pitch, self.error_est_pitch_dot)

        # A and B matrices used to take state matrix to the next timestep.
        self.A = self.createA(self.t)

        self.B = self.createB(self.t)

        # initial state vectors
        # x looks like [x, y, z, xdot, ydot, zdot, xddot, yddot, zddot]
        # y looks like [yaw, yawdot, roll, rolldot, pitch, pitchdot]

        if self.simMode == False:
            for i in self.z:
                if i[0] == 0: continue
                xStart = i[0]
                yStart = i[1]
                zStart = i[2]

        # Transform initial input into drone reference frame.
        (x_init, y_init, z_init) = transformAngles(self.z[0][3], self.z[0][4], self.z[0][5], self.z[0][6], self.z[0][8],
                                                   self.z[0][10], True)

        if self.simMode == False:
            self.X = numpy.array([xStart, yStart, zStart, 1.05, 0, 0, x_init, y_init, z_init])
        else:
            self.X = numpy.array([self.z[0][0], self.z[0][1], self.z[0][2], 0, 0, 0, x_init, y_init, z_init])
        self.W = numpy.array([self.z[0][6], self.z[0][7], self.z[0][8], self.z[0][9], self.z[0][10], self.z[0][11]])




        # Noise matrices for position and angle.
        self.R = numpy.array([[self.noiseX, 0, 0, 0, 0, 0],
                              [0, self.noiseY, 0, 0, 0, 0],
                              [0, 0, self.noiseZ, 0, 0, 0],
                              [0, 0, 0, self.noiseA_x, 0, 0],
                              [0, 0, 0, 0, self.noiseA_y, 0],
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
        # counter = 0
        # Loop through data and do kalman filter iteration for each data point.
        for n, data in enumerate(self.z[1:]):
            if self.simMode:
                self.t = self.times[n]

            # if data[0] == 0: continue
            # Choose correct H matrix used for converting input data into same shape as state matrix.
            if not self.simMode:
                self.t = data[-1]
                # print self.t
                # self.t = 0.01
                # if data[0] == 0: continue
            # set to first index so x_pos can be 0 everywhere
            # if (data[0] == 0):
            # print data
            # time.sleep(1)
            if (data[1] == 0):
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
                # print "rate used"
                self.H_angle[:] = [[0, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1]]
            else:
                # print "angle used"
                self.H_angle[:] = [[1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1]]



            # 	Predict angle
            self.W = self.predictAngle(self.W, self.t)
            self.W[0] = self.W[0] % (2 * numpy.pi)
            self.W[2] = self.W[2] % (2 * numpy.pi)
            self.W[4] = self.W[4] % (2 * numpy.pi)

            # todo - split this up into two matrixes that can be dotted together if poss. Final result will be this though.
            # todo - use indexes instead of whole matrix
            self.G_angle_dotted[:] = [[self.t**4, self.t**3, 0, 0, 0, 0],
                                   [self.t**3, self.t**2, 0, 0, 0, 0],
                                   [0, 0, self.t**4, self.t**3, 0, 0],
                                   [0, 0, self.t**3, self.t**2, 0, 0],
                                   [0, 0, 0, 0, self.t**4, self.t**3],
                                   [0, 0, 0, 0, self.t**3, self.t**2]]
            Q = self.G_angle_dotted * self.sigma_a ** 2
            self.P_angle = self.B.dot(self.P_angle).dot(self.B.T) + Q

            # todo - the angles being read out here are wrong. This needs fixing.
            # Calculate Kalman gain
            S = self.H_angle.dot(self.P_angle).dot(self.H_angle.T) + self.R_angle
            K = self.P_angle.dot(self.H_angle.T).dot(inv(S))
            # if (self.simMode):
            #     Y = data[6:] - self.H_angle.dot(self.W)
            # else:
            #     Y = data[6:12] - self.H_angle.dot(self.W)
            Y = data[6:12] - self.H_angle.dot(self.W)
            # Update state matrix
            self.W = self.W + K.dot(Y)
            # Update covariance matrix.
            self.P_angle = (self.K_angle_identity - K.dot(self.H_angle)).dot(self.P_angle)



            # Obtain current angles.
            yaw = self.W[0]
            roll = self.W[2]
            pitch = self.W[4]



            self.predictedYaw.append(yaw)

            # if counter < 10:
            #     print yaw, roll, pitch
            #
            # counter += 1


            # Predict positions
            self.X = self.prediction2d(self.X, self.t)

            # 	Handling X state vector
            # todo - split this up into two matrixes that can be dotted together if poss. Final result will be this though.
            self.G_dotted[:] = [[(self.t**4)/4, 0, 0, (self.t**3)/2, 0, 0, (self.t**2)/2, 0, 0],
                                [0 ,(self.t**4)/4, 0, 0, (self.t**3)/2, 0, 0, (self.t**2)/2, 0],
                                [0, 0 ,(self.t**4)/4, 0, 0, (self.t**3)/2, 0, 0, (self.t**2)/2],
                                [(self.t**3)/2, 0, 0, self.t**2, 0, 0, self.t, 0, 0],
                                [0, (self.t**3)/2, 0, 0, self.t**2, 0, 0, self.t, 0],
                                [0, 0, (self.t**3)/2, 0, 0, self.t**2, 0, 0, self.t],
                                [(self.t**2)/2, 0, 0, self.t, 0, 0, 1, 0, 0],
                                [0, (self.t**2)/2, 0, 0, self.t, 0, 0, 1, 0],
                                [0, 0, (self.t**2)/2, 0, 0, self.t, 0, 0, 1]]

            # calculate Kalman gain.
            # todo - keeping stationary and take rms
            # todo - rtk & accelerometer
            Q = self.G_dotted * self.sigma_a ** 2
            self.P = self.A.dot(self.P).dot(self.A.T) + Q
            S = self.H.dot(self.P).dot(self.H.T) + self.R
            K = self.P.dot(self.H.T).dot(inv(S))
            # if counter < 5:
            #     # print yaw, roll, pitch
            #     print "In real drone's ref phrame: ", data[3], data[4], data[5]
            #     print "Angles: ", yaw, roll, pitch

            # Transform accerelation vector back to inerial reference frame.
            # if (yaw != 0 or roll != 0 or pitch != 0):
            #     print "There is an angle!"
            data[3], data[4], data[5] = transformAngles(data[3], data[4], data[5], yaw, roll, pitch, True)

            # if counter < 5:
            #     # print yaw, roll, pitch
            #     print "Accelerations real: ", data[3], data[4], data[5]
            #     counter += 1

            # print yaw, roll, pitch

            Y = data[:6] - self.H.dot(self.X)

            # Update state vector
            self.X = self.X + K.dot(Y)

            # Save the result for comparison later
            self.predictedX.append(self.X)
            self.covariances.append(self.P)

            # Update the covariance matrix
            self.P = (self.K_identity - K.dot(self.H)).dot(self.P)

    def postSimulationAnalysis(self):

        predictedPositionsX = [x[0] for x in self.predictedX]
        predictedPositionsY = [x[1] for x in self.predictedX]
        predictedPositionsZ = [x[2] for x in self.predictedX]

        # xerror, yerror, zerror = self.findErrors(self.covariances)

        # find out precision of kalman filter
        precision = precisionAnalysis(predictedPositionsX, predictedPositionsY, predictedPositionsZ, self.positions_X,
                                      self.positions_Y, self.positions_Z)

        precision2d = precisionAnalysis2d(predictedPositionsX, predictedPositionsY, self.positions_X, self.positions_Y)
        print "Precision in 2d is", precision2d, "m"
        print "Precision is ", precision, " m"

        # print the results.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(predictedPositionsX, predictedPositionsY, predictedPositionsZ, 'o', ms =2, label = "kalman")

        ax.plot(self.positions_X, self.positions_Y, self.positions_Z, label="real")

        # for i in numpy.arange(0, len(predictedPositionsX)):
        #     ax.plot([predictedPositionsX[i] + xerror[i], predictedPositionsX[i] - xerror[i]], [predictedPositionsY[i], predictedPositionsY[i]], [predictedPositionsZ[i], predictedPositionsZ[i]], marker="_")
        #     ax.plot([predictedPositionsX[i], predictedPositionsX[i]], [predictedPositionsY[i] + yerror[i], predictedPositionsY[i] - yerror[i]], [predictedPositionsZ[i], predictedPositionsZ[i]], marker="_")
        #     ax.plot([predictedPositionsX[i], predictedPositionsX[i]], [predictedPositionsY[i], predictedPositionsY[i]], [predictedPositionsZ[i] + zerror[i], predictedPositionsZ[i] - zerror[i]], marker="_")

        ax.set_xlabel('X axis/m')
        ax.set_ylabel('Y axis/m')
        ax.set_zlabel('Z axis/m')
        ax.text2D(0.1, 0.95, "Kalman Filter predictions and true positions", transform=ax.transAxes)
        ax.legend()
        pyplot.show()


    def postSimulation2d(self):
        predictedPositionsX = [x[0] for x in self.predictedX]
        predictedPositionsY = [x[1] for x in self.predictedX]

        precision2d = precisionAnalysis2d(predictedPositionsX, predictedPositionsY, self.positions_X, self.positions_Y)
        print "Precision in 2d is", precision2d, "m"
        # return precision2d

        xError, yError = self.findErrors(self.covariances)
        #
        # # length = 6500
        #
        # # print the results.
        # plt.errorbar(predictedPositionsX[:], predictedPositionsY[:], xError[:], yError[:], label="Kalman")
        plt.plot(predictedPositionsX, predictedPositionsY, label="Predicted motion")
        plt.plot(self.positions_X, self.positions_Y, label="Actual motion")
        plt.legend()
        plt.title("Section of random walk simulation shown in 2d with error bars")
        plt.xlabel("X relative position/m")
        plt.ylabel("Y relative position/m")
        plt.show()
        plt.plot(self.predictedYaw)
        plt.show()

    def getPredictions(self):
        self.predictedPositionsX = [x[0] for x in self.predictedX]
        self.predictedPositionsY = [x[1] for x in self.predictedX]
        self.predictedPositionsZ = [x[2] for x in self.predictedX]

    def postAnalysis(self):
        self.predictedPositionsX = [x[0] for x in self.predictedX]
        self.predictedPositionsY = [x[1] for x in self.predictedX]
        self.predictedPositionsZ = [x[2] for x in self.predictedX]

        # differences = [(i - (3.8 * j + 11.8))**2 for i in self.predictedPositionsY for j in self.predictedPositionsX]
        # print "difference is ", numpy.mean(differences)

        # xErrors = [x[0][0] for x in self.covariances]
        # yErrors = [y[1][1] for y in self.covariances]

        rtkX = []
        rtkY = []
        errorX = []
        errorY = []


        for i in self.z:
            # if i[0] != 0:
            #     rtkX.append(i[0])
            if i[1] != 0:
                rtkX.append(i[0])
                rtkY.append(i[1])

        xActual = numpy.mean(rtkX)

        xPredActual = numpy.mean(self.predictedPositionsX)

        # print len(self.predictedPositionsX)
        # rtkDetlas = [abs(i - xActual) for i in rtkX]
        # print "mean of rtk deltas: ", numpy.mean(rtkDetlas)
        # print max(self.predictedPositionsX[200:]) - min(self.predictedPositionsX[200:])
        # print "std of rtk deltas: ", numpy.std(rtkDetlas)
        # deltas = [abs(i - xPredActual) for i in self.predictedPositionsX]
        # print "mean of deltas: ", numpy.mean(deltas)
        # print "std of deltas: ", numpy.std(deltas)

        # rtkX = [0 for i in range(len(rtkY))]
        # print "mean X is", numpy.mean(rtkX)
        # print len(rtkX), len(rtkY)
        # print the results.
        # plt.errorbar(self.predictedPositionsX, self.predictedPositionsY, xErrors, yErrors, label="kalman predictsion")
        plt.plot(self.predictedPositionsX[:], self.predictedPositionsY[:], label="Kalman predictions")
        plt.plot(rtkX[:], rtkY[:], 'o', ms=1, label="RTK")
        # plt.xlim(-2, -1)
        # def equation(x):
        #     return [3.8 * i + 11.8 for i in x]
        # # plt.plot(rtkX, equation(rtkX), label="equation")
        plt.xlabel("Relative position East/m")
        plt.ylabel("Relative position North/m")
        plt.legend()
        # plt.xlim(-1, 1)
        pyplot.show()

        # def func(x, a, b, c, d):
        #     return a * numpy.cos(b * x - c) + d
        #
        #
        # times = numpy.arange(0, 16, 0.1)
        # popt = [-0.3, 2.6, 12, 1.6076921935]
        # plt.plot(rtkY)
        # plt.plot(func(times, *popt))
        # plt.show()


    def getAccuracy(self):
        self.predictedPositionsX = [x[0] for x in self.predictedX]
        self.predictedPositionsY = [x[1] for x in self.predictedX]

        deltas = []
        for i, val in enumerate(self.predictedPositionsX):
            delta = abs(numpy.sqrt((val+3.45)**2 + (self.predictedPositionsY[i]-2.41)**2) - 0.35)
            deltas.append(delta)
        # deltas = [abs(numpy.sqrt(x**2 + y**2)) - 0.275 for x in self.predictedPositionsX for y in self.predictedPositionsY]
        return numpy.mean(deltas)


    def findDifferencesRealData(self):
        if self.simMode:
            raise Exception("This only works for real data - should not be used in simulation.")
        rtkX = []
        rtkY = []
        time = 0
        times = []
        for i, val in enumerate(self.z):
            time += val[-1]
            if val[0] != 0:
                times.append(time)
                rtkX.append(val[0])
                rtkY.append(val[1])
        def func(x, a, b, c, d):
            return a * numpy.cos(b * x - c) + d
        popt = [-0.3, 2.6, 12, 1.6076921935]
        differences = []
        time = 0
        yAccs = []
        timesAccs = []
        trueYs = []
        # print len(self.z), len(self.predictedPositionsX)
        for i, val in enumerate(self.z[1:]):
            time += val[-1]
            trueY = func(time, *popt)
            trueYs.append(trueY)
            yAccs.append(val[4])
            timesAccs.append(time)
            # if i == len(self.z): break
            difference = ((self.predictedPositionsX[i] - -3.39981724138)**2 + (self.predictedPositionsY[i] - trueY)**2)**0.5
            differences.append(difference)
        # print numpy.mean(differences)
        # print time
        timeArray = numpy.arange(0, 12, 0.1)
        plt.plot(timesAccs, self.predictedYaw, label="predictionsYaws")
        # plt.plot(times, rtkY, label="rtkY")
        # plt.plot(timeArray, func(timeArray, *popt), "r", label="Model")
        plt.xlabel("Time/s")
        plt.ylabel("Position/m")
        plt.legend()

        # plt.plot(timesAccs, yAccs)
        plt.show()

        # plt.plot(timesAccs, self.predictedPositionsX, label="predictionsY")
        # plt.plot(times, rtkX, label="rtkX")
        # # plt.plot(timeArray, func(timeArray, *popt), "r", label="Model")
        # plt.xlabel("Time/s")
        # plt.ylabel("Position/m")
        # plt.legend()
        #
        # # plt.plot(timesAccs, yAccs)
        # plt.show()

    def findErrors(self, covariances):
        xError = []
        yError = []
        zError = []
        for n, i in enumerate(covariances):
            # print n
            # xError.append(numpy.sqrt(i[0][0]/(n+1)))
            # yError.append(numpy.sqrt(i[1][1]/(n+1)))
            # zError.append(numpy.sqrt(i[2][2]/(n+1)))

            xError.append(numpy.sqrt(i[0][0]))
            yError.append(numpy.sqrt(i[1][1]))
            zError.append(numpy.sqrt(i[2][2]))

        return xError, yError






def findBestError():
    errors = numpy.linspace(0.000001, 1, 20)
    currentAccAngle = 0
    bestErrorAngle = 0
    for i in errors:
        kf = KalmanFilter(1, 1, 1, 1, 1, i, 1, 1, 1)
        kf.preSimulationSetup()
        kf.runSimulation()
        acc = kf.getAccuracy()
        print acc
        if (currentAccAngle == 0 or acc < currentAccAngle):
            currentAccAngle = acc
            bestErrorAngle = i

    currentAccRate = 0
    bestErrorAngleRate = 0
    for i in errors:
        kf = KalmanFilter(1, 1, 1, 1, 1, bestErrorAngle, i, 1, 1)
        kf.preSimulationSetup()
        kf.runSimulation()
        acc = kf.getAccuracy()
        print acc
        if (currentAccRate == 0 or acc < currentAccRate):
            currentAccRate = acc
            bestErrorAngleRate = i

    currentAccRTK = 0
    bestErrorRTK = 0
    for i in errors:
        kf = KalmanFilter(1, 1, 1, 1, 1, bestErrorAngle, bestErrorAngleRate, i, 1)
        kf.preSimulationSetup()
        kf.runSimulation()
        acc = kf.getAccuracy()
        print acc
        if (currentAccRTK == 0 or acc < currentAccRTK):
            currentAccRTK = acc
            bestErrorRTK = i

    currentAccAcc = 0
    bestErrorAcc = 0
    for i in errors:
        kf = KalmanFilter(1, 1, 1, 1, 1, bestErrorAngle, bestErrorAngleRate, bestErrorRTK, i)
        kf.preSimulationSetup()
        kf.runSimulation()
        acc = kf.getAccuracy()
        print acc
        if (currentAccAcc == 0 or acc < currentAccAcc):
            currentAccAcc = acc
            bestErrorAcc = i

    return (bestErrorAngle, bestErrorAngleRate, bestErrorRTK, bestErrorAcc)

# errors = findBestError()
errors = [1.0, 0.001, 0.0001, 0.10526405263157894]
# (1.0, 9.9999999999999995e-07, 9.9999999999999995e-07, 1.0)
print errors

kf = KalmanFilter(0.001, 3, 0.1, 0.1, 0.1, *errors)
kf.preSimulationSetup()
kf.runSimulation()
if kf.simMode:
    # kf.postSimulationAnalysis()
    kf.postSimulation2d()
else:
    kf.postAnalysis()
    kf.findDifferencesRealData()
    print "RMS accuracy of predictions is ", kf.getAccuracy(), " m"


# predictions = []
# # accNoises = numpy.arange(0.15, 0.5, 0.02)
# rtkNoises = numpy.arange(0, 0.02, 0.002)
# # for accNoise in accNoises:
# for rtkNoise in rtkNoises:
#     # print "Running for ", accNoise
#     # kf = KalmanFilter(0.0001, 3, 0.1, accNoise)
#     print "Running for ", rtkNoise
#     kf = KalmanFilter(0.0001, 3, 0.1, 0.2, rtkNoise, *errors)
#     kf.preSimulationSetup()
#     kf.runSimulation()
#     kf.getPredictions()
#     precision = precisionAnalysis(kf.predictedPositionsX, kf.predictedPositionsY, kf.predictedPositionsZ, kf.positions_X,
#                                   kf.positions_Y, kf.positions_Z)
#     predictions.append(precision)
#
# plt.plot(rtkNoises, predictions)
# plt.xlabel("RTK noise m/s^2")
# plt.ylabel("RMS accuracy of Kalman filter")
# plt.title("Accuracy of Kalman filter for various RTK noise values.")
# plt.show()


# print transformAngles(-2.8821, 0.8061, 0, -0.2, 0, 0, True)

# timestepsList = numpy.linspace(0.0001, 1)
# precisions = []
#
# for timestep in timestepsList:
#     kf = KalmanFilter(timestep)
#     kf.preSimulationSetup()
#     kf.runSimulation()
#     kf.getPredictions()
#     precision = precisionAnalysis(kf.predictedPositionsX, kf.predictedPositionsY, kf.predictedPositionsZ, kf.positions_X,
#                                   kf.positions_Y, kf.positions_Z)
#     precisions.append(precision)
#
# plt.plot(timestepsList, precisions)
# plt.xlabel("Time-step size /s")
# plt.ylabel("RMS accuracy of Kalman filter")
# plt.title("Accuracy of Kalman filter for various timestep sizes.")
# plt.show()
#
# gradients = [3]
# cutoffs = numpy.arange(0, 0.1, 0.01)
# precisionsGradients = []
#
#
# for gradient in gradients:
#     precisions = []
#     for cutoff in cutoffs:
#         kf = KalmanFilter(0.001, gradient, cutoff)
#         kf.preSimulationSetup()
#         kf.runSimulation()
#         # kf.getPredinjctions()
#         # precision = precisionAnalysis(kf.predictedPositionsX, kf.predictedPositionsY, kf.predictedPositionsZ,
#         #                               kf.positions_X, kf.positions_Y, kf.positions_Z)
#         precision = kf.postSimulation2d()
#         precisions.append(precision)
#         print cutoff, precision
#     precisionsGradients.append(precisions)
#
# plt.plot(cutoffs, precisionsGradients[0], label="gradient = 1")
# # plt.plot(cutoffs, precisionsGradients[1], label="gradient = 2")
# # plt.plot(cutoffs, precisionsGradients[2], label="gradient = 3")
# plt.legend()
# plt.xlabel("Cutoff / f")
# plt.ylabel("RMS accuracy of Kalman filter")
# plt.title("Accuracy of Kalman filter for various cutoff frequencies and gradients.")
# plt.show()