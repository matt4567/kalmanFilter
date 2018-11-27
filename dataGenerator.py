import numpy
import matplotlib.pyplot as plt
from helper import *

timesteps = 0.001


def genData2d():
    """Simulate data from drone moving in circle in x and y with radius 10 and accelerating in the positive vertical direction."""
    radius = 10
    # array of timesteps.
    timestepsArray = numpy.arange(0, 10 * numpy.pi, timesteps)
    # real positions in x, y and z direction.
    positions_X = numpy.sin(timestepsArray) * radius * 1/2
    positions_Y = numpy.cos(timestepsArray) * radius
    positions_Z = (timestepsArray ** 2)/2
    # data array setup to return
    data = numpy.zeros((1, 12))

    for index, val in enumerate(timestepsArray):
        # angular rate is 0.2 radians per second
        angleIncrease = 0.002
        angularRate = angleIncrease / timesteps
        angularRate = angularRate + numpy.random.normal(0, 0.1)

        # setup angles
        roll = angleIncrease * index + numpy.random.normal(0, 0.01)
        yaw = angleIncrease * index + numpy.random.normal(0, 0.01)
        pitch = angleIncrease * index + numpy.random.normal(0, 0.01)

        # setup accelerations in x, y and z directions.
        x_ddot = - radius * 1/2 * sin(val)
        y_ddot = - radius * cos(val)
        z_ddot = 1

        # transform acceleration vecctor to drones reference frame.
        (accData_x, accData_y, accData_z) = transformAngles(x_ddot, y_ddot, z_ddot, yaw, roll, pitch, False)

        # add noise.
        accData_x += numpy.random.normal(0, 0.0008)
        accData_y += numpy.random.normal(0, 0.0008)
        accData_z += numpy.random.normal(0, 0.0008)

        # RTK and angle data for one in every 10 timesteps. Add the angular rate and acceleration data for every timestep.
        if (index % 10 == 0):
            RTK_x = numpy.sin(val) * radius *1/2 + numpy.random.normal(0, .1)
            RTK_y = numpy.cos(val) * radius + numpy.random.normal(0, .1)
            RTK_z = (val ** 2)/2 + numpy.random.normal(0, .1)
            data = numpy.append(data, [[RTK_x, RTK_y, RTK_z, accData_x, accData_y, accData_z, yaw, angularRate, roll, angularRate, pitch, angularRate]], axis=0)
        else:
            data = numpy.append(data, [[0, 0, 0, accData_x, accData_y, accData_z, 0, angularRate, 0, angularRate, 0, angularRate]], axis=0)

    return (data[1:], positions_X, positions_Y, positions_Z)


def cos(x):
    return numpy.cos(x)


def sin(x):
    return numpy.sin(x)
