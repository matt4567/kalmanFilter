from accReader import *
import numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from dataGenerator import *
from helper import *
import time


# feed in 2d simulation data/
data = genData2d()
z = data[0]

# feed in real positional data.
positions_X = data[1]
positions_Y = data[2]
positions_Z = data[3]


# keep track of all predictedPositions
predictedX = []


def prediction2d(x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot):
    """Predict position of drone using taylor expansion to 2nd order as integral approximation."""
    X = numpy.array([x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])
    return A.dot(X)


def predictAngle(yaw, yaw_dot, roll, roll_dot, pitch, pitch_dot):
    """Predict angle from taylor expansion to 1st order as integral approximation."""
    W = numpy.array([yaw, yaw_dot, roll, roll_dot, pitch, pitch_dot])
    return B.dot(W)


def covariance2d(P_x, P_y, P_z, P_x_dot, P_y_dot, P_z_dot, P_x_ddot, P_y_ddot, P_z_ddot):
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


def covarianceAngle(P_yaw, P_yaw_dot, P_roll, P_roll_dot, P_pitch, P_pitch_dot):
    """Create covariance matrix of angles."""
    cov_matrix = numpy.array([[P_yaw, 0, 0, 0, 0, 0],
                              [0, P_yaw_dot, 0, 0, 0, 0],
                              [0, 0, P_roll, 0, 0, 0],
                              [0, 0, 0, P_roll_dot, 0, 0],
                              [0, 0, 0, 0, P_pitch, 0],
                              [0, 0, 0, 0, 0, P_pitch_dot]])
    return cov_matrix


# Initial Conditions
# same timestep as defined in dataGenerator.
t = timesteps

# Process / Estimation Errors
# todo - make these more realistic, this is just for development purposes.
# x, y and z errors.
error_est_x = 1
error_est_v_x = 1
error_est_a_x = 1
error_est_y = 1
error_est_v_y = 1
error_est_a_y = 1
error_est_z = 1
error_est_v_z = 1
error_est_a_z = 1

# angle errors.
error_est_angle = 1
error_est_angle_dot = 1
error_est_roll = 1
error_est_roll_dot = 1
error_est_pitch = 1
error_est_pitch_dot = 1

sigma_a = 1

# Initial Estimation Covariance Matrix - 2d case
P = covariance2d(error_est_x, error_est_y, error_est_z, error_est_v_x, error_est_v_y, error_est_v_z, error_est_a_x, error_est_a_y, error_est_a_z)
P_angle = covarianceAngle(error_est_angle, error_est_angle_dot, error_est_roll, error_est_roll_dot, error_est_pitch, error_est_pitch_dot)

# A and B matrices used to take state matrix to the next timestep.
A = numpy.array([[1, 0, 0, t, 0, 0, (t ** 2) / 2, 0, 0],
                 [0, 1, 0, 0, t, 0, 0, (t ** 2) / 2, 0],
                 [0, 0, 1, 0, 0, t, 0, 0, (t ** 2) / 2],
                 [0, 0, 0, 1, 0, 0, t, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, t, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, t],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])

B = numpy.array([[1, t, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, t, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, t],
                 [0, 0, 0, 0, 0, 1]])

# initial state vectors
# x looks like [x, y, z, xdot, ydot, zdot, xddot, yddot, zddot]
# y looks like [yaw, yawdot, roll, rolldot, pitch, pitchdot]

# Transform initial input into drone reference frame.
(x_init, y_init, z_init) = transformAngles(z[0][3], z[0][4], z[0][5], z[0][6], z[0][8], z[0][10], True)
X = numpy.array([z[0][0], z[0][1], z[0][2], 0, 0, 0, x_init, y_init, z_init])
W = numpy.array([z[0][6], z[0][7], z[0][8], z[0][9], z[0][10], z[0][11]])


# Noise values
# todo - dummy values, find better vals.
noiseX = 1
noiseY = 1
noiseZ = 1
noiseA_x = 1
noiseA_y = 1
noiseA_z = 1

noiseYaw = 1
noiseYaw_dot = 1
noiseRoll = 1
noiseRoll_dot = 1
noisePitch = 1
noisePitch_dot = 1

# Noise matrices for position and angle.
R = numpy.array([[noiseX, 0, 0, 0, 0, 0],
                 [0, noiseA_x, 0, 0, 0, 0],
                 [0, 0, noiseY, 0, 0, 0],
                 [0, 0, 0, noiseA_y, 0, 0],
                 [0, 0, 0, 0, noiseZ, 0],
                 [0, 0, 0, 0, 0, noiseA_z]])
R_angle = numpy.array([[noiseYaw, 0, 0, 0, 0, 0],
                       [0, noiseYaw_dot, 0, 0, 0, 0],
                       [0, 0, noiseRoll, 0, 0, 0],
                       [0, 0, 0, noiseRoll_dot, 0, 0],
                       [0, 0, 0, 0, noisePitch, 0],
                       [0, 0, 0, 0, 0, noisePitch_dot]])

# Loop through data and do kalman filter iteration for each data point.
for data in z[1:]:
    # Choose correct H matrix used for converting input data into same shape as state matrix.
    if (data[0] == 0):
        H = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    else:
        H = numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    if (data[6] == 0):
        H_angle = numpy.array([[0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1]])
    else:
        H_angle = numpy.array([[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])

    # 	Predict angle
    W = predictAngle(W[0], W[1], W[2], W[3], W[4], W[5])

    # todo - split this up into two matrixes that can be dotted together if poss. Final result will be this though.
    G_angle_dotted = numpy.array([[t**4, t**3, 0, 0, 0, 0],
                                  [t**3, t**2, 0, 0, 0, 0],
                                  [0, 0, t**4, t**3, 0, 0],
                                  [0, 0, t**3, t**2, 0, 0],
                                  [0, 0, 0, 0, t**4, t**3],
                                  [0, 0, 0, 0, t**3, t**2]])
    Q = G_angle_dotted * sigma_a ** 2
    P_angle = B.dot(P_angle).dot(B.T) + Q

    # Calculate Kalman gain
    S = H_angle.dot(P_angle).dot(H_angle.T) + R_angle
    K = P_angle.dot(H_angle.T).dot(inv(S))
    Y = data[6:] - H_angle.dot(W)
    # Update state matrix
    W = W + K.dot(Y)
    # Update covariance matrix.
    P_angle = (numpy.identity(len(K)) - K.dot(H_angle)).dot(P_angle)
    # Obtain current angles.
    yaw = W[0]
    roll = W[2]
    pitch = W[4]

    # Predict positions
    X = prediction2d(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8])

    # 	Handling X state vector
    # todo - split this up into two matrixes that can be dotted together if poss. Final result will be this though.
    G_dotted = numpy.array([[(t**4)/4, 0, 0, (t**3)/2, 0, 0, (t**2)/2, 0, 0],
                            [0 ,(t**4)/4, 0, 0, (t**3)/2, 0, 0, (t**2)/2, 0],
                            [0, 0 ,(t**4)/4, 0, 0, (t**3)/2, 0, 0, (t**2)/2],
                            [(t**3)/2, 0, 0, t**2, 0, 0, t, 0, 0],
                            [0, (t**3)/2, 0, 0, t**2, 0, 0, t, 0],
                            [0, 0, (t**3)/2, 0, 0, t**2, 0, 0, t],
                            [(t**2)/2, 0, 0, t, 0, 0, 1, 0, 0],
                            [0, (t**2)/2, 0, 0, t, 0, 0, 1, 0],
                            [0, 0, (t**2)/2, 0, 0, t, 0, 0, 1]])

    # calculate Kalman gain.
    Q = G_dotted * sigma_a ** 2
    P = A.dot(P).dot(A.T) + Q
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(inv(S))

    # Transform accerelation vector back to inerial reference frame.
    (data[3], data[4], data[5]) = transformAngles(data[3], data[4], data[5], yaw, roll, pitch, True)

    Y = data[:6] - H.dot(X)

    # Update state vector
    X = X + K.dot(Y)

    # Save the result for comparison later
    predictedX.append(X)

    # Update the covariance matrix
    P = (numpy.identity(len(K)) - K.dot(H)).dot(P)

predictedPositionsX = [x[0] for x in predictedX]
predictedPositionsY = [x[1] for x in predictedX]
predictedPositionsZ = [x[2] for x in predictedX]

# find out precision of kalman filter
precision = precisionAnalysis(predictedPositionsX, predictedPositionsY, predictedPositionsZ, positions_X, positions_Y, positions_Z)
print "Precision is ", precision, " m"

# print the results.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(predictedPositionsX, predictedPositionsY, predictedPositionsZ, 'o', label = "kalman")
ax.plot(positions_X, positions_Y, positions_Z, label="real")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.text2D(0.1, 0.95, "Kalman Filter predictions and true positions", transform=ax.transAxes)
ax.legend()
pyplot.show()
