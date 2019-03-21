from __future__ import division
import numpy
import matplotlib.pyplot as plt
import random
import time
from helper import *
from scipy.fftpack import rfft, irfft, fftfreq

def genData2d(timesteps):
    """Simulate data from drone moving in circle in x and y with radius 10 and accelerating in the positive vertical direction."""
    radius = 2
    # array of timesteps.
    timestepsArray = numpy.arange(0, 10 * numpy.pi, timesteps)
    # real positions in x, y and z direction.
    positions_X = numpy.sin(timestepsArray) * radius * 1/2
    positions_Y = numpy.cos(timestepsArray) * radius
    positions_Z = (timestepsArray)/2
    # data array setup to return
    data = numpy.zeros((1, 12))

    for index, val in enumerate(timestepsArray):
        # angular rate is 0.2 radians per second
        angleIncrease = 0.002
        angularRate = angleIncrease / timesteps
        angularRate = angularRate + numpy.random.normal(0, 0.01)

        # setup angles
        roll = angleIncrease * index + numpy.random.normal(0, 0.01)
        yaw = angleIncrease * index + numpy.random.normal(0, 0.01)
        pitch = angleIncrease * index + numpy.random.normal(0, 0.01)

        # setup accelerations in x, y and z directions.
        x_ddot = - radius * 1/2 * sin(val)
        y_ddot = - radius * cos(val)
        z_ddot = 0

        # transform acceleration vecctor to drones reference frame.
        (accData_x, accData_y, accData_z) = transformAngles(x_ddot, y_ddot, z_ddot, yaw, roll, pitch, False)

        # add noise.
        accData_x += numpy.random.normal(0, 0.0008)
        accData_y += numpy.random.normal(0, 0.0008)
        accData_z += numpy.random.normal(0, 0.0008)

        # RTK and angle data for one in every 10 timesteps. Add the angular rate and acceleration data for every timestep.
        if (index % 10 == 0):
            RTK_x = numpy.sin(val) * radius *1/2 + numpy.random.normal(0, .01)
            RTK_y = numpy.cos(val) * radius + numpy.random.normal(0, .01)
            RTK_z = (val)/2 + numpy.random.normal(0, .01)
            data = numpy.append(data, [[RTK_x, RTK_y, RTK_z, accData_x, accData_y, accData_z, yaw, angularRate, roll, angularRate, pitch, angularRate]], axis=0)
        else:
            data = numpy.append(data, [[0, 0, 0, accData_x, accData_y, accData_z, 0, angularRate, 0, angularRate, 0, angularRate]], axis=0)

    return (data[1:], positions_X, positions_Y, positions_Z)


def cos(x):
    return numpy.cos(x)


def sin(x):
    return numpy.sin(x)

def genRandomFlightData(timesteps, gradient, cutoff, rtkNoise):

    # wind force
    # find acceleration
    # find velocities



    accType = "Mems"
    xs = []
    ys = []
    zs = []

    xs_noise = []
    ys_noise = []
    zs_noise = []
    start = 0
    finish = 100

    times = numpy.arange(start, finish, timesteps)

    x = 0
    y = 0
    z = 0

    x_velocity = 0
    y_velocity = 0
    z_velocity = 0

    x_accs = []
    y_accs = []
    z_accs = []

    roll = []
    roll_dot = []
    pitch = []
    pitch_dot = []
    yaw = []
    yaw_dot = []


    counterIn = 0
    counterOut = 0
    pi = numpy.pi

    def angleEquation(timeStep, phase):
        return numpy.sin((timeStep/25) * numpy.pi + phase)

    def angleRateEquation(timeStep, phase):
        return numpy.pi/25 * numpy.cos((timeStep/25) * numpy.pi + phase)


    for i, val in enumerate(times):
        x_vel_old = x_velocity
        y_vel_old = y_velocity
        z_vel_old = z_velocity
        x_velocity += numpy.random.normal(0, 2 * timesteps) + numpy.sin(1 *numpy.pi * i) + numpy.sin(3 *numpy.pi * i)
        y_velocity += numpy.random.normal(0, 2 * timesteps) + numpy.sin(1 *numpy.pi * i) + numpy.sin(3 *numpy.pi * i)
        z_velocity += numpy.random.normal(0, 2 * timesteps) + numpy.sin(1 *numpy.pi * i) + numpy.sin(3 *numpy.pi * i)
        #
        # x_velocity += numpy.sin(1 * pi *i) + numpy.sin(3 *pi * i)
        # y_velocity += numpy.sin(1 * pi * i) + numpy.sin(3 * pi * i)
        # z_velocity += numpy.sin(1 * pi * i) + numpy.sin(3 * pi * i)
        x = x + x_velocity * timesteps
        y = y + y_velocity * timesteps
        z = z + z_velocity * timesteps
        if (i % 10 == 0):
            xs_noise.append(x + numpy.random.normal(0, rtkNoise))
            ys_noise.append(y + numpy.random.normal(0, rtkNoise))
            zs_noise.append(z + numpy.random.normal(0, rtkNoise))
        else:
            xs_noise.append(0)
            ys_noise.append(0)
            zs_noise.append(0)

        xs.append(x)
        ys.append(y)
        zs.append(z)

        roll.append(angleEquation(val, numpy.pi/2))
        pitch.append(angleEquation(val, numpy.pi))
        yaw.append(angleEquation(val, 0))

        roll_dot.append(angleRateEquation(val, numpy.pi/2))
        pitch_dot.append(angleRateEquation(val, numpy.pi))
        yaw_dot.append(angleRateEquation(val, 0))

        x_acc = ((x_velocity - x_vel_old) / timesteps) + numpy.random.normal(0, .001)
        y_acc = ((y_velocity - y_vel_old) / timesteps) + numpy.random.normal(0, .001)
        z_acc = ((z_velocity - z_vel_old) / timesteps) + numpy.random.normal(0, .001)

        x_acc, y_acc, z_acc = transformAngles(x_acc, y_acc, z_acc, yaw[-1], roll[-1], pitch[-1], False)

        if (accType == "Mems"):
            x_accs.append(x_acc)
            y_accs.append(y_acc)
            z_accs.append(z_acc)

        # else:
            # if (len(x_accs) > 0):
            #     if (abs(((x_acc - x_accs[-1]) / timesteps)) < 1 or abs(((y_acc - y_accs[-1]) / timesteps)) < 1):
            #         x_accs.append(0)
            #         y_accs.append(0)
            #         counterOut+=1
            #     else:
            #         x_accs.append(x_acc)
            #         y_accs.append(y_acc)
                    # counterIn+=1

            # else:
            #     x_accs.append(x_acc)
            #     y_accs.append(y_acc)

    x_highpass = highPassFilter(x_accs, False, timesteps, gradient, cutoff)
    y_highpass = highPassFilter(y_accs, False, timesteps, gradient, cutoff)
    z_highpass = highPassFilter(z_accs, False, timesteps, gradient, cutoff)

    x = numpy.array(xs_noise)
    y = numpy.array(ys_noise)
    z = numpy.array(zs_noise)
    acc_x = numpy.array(x_highpass)
    acc_y = numpy.array(y_highpass)
    acc_z = numpy.array(z_highpass)
    zero = numpy.zeros(len(x))

    # print yaw[:10]

    data = numpy.column_stack((x, y, z, acc_x, acc_y, acc_z, yaw, yaw_dot, roll, roll_dot, pitch, pitch_dot))
    return (data, xs, ys, zs)

def circleSim(times):

    rate = 10 * 2 * numpy.pi / 21.02
    radius = 0.35
    def rtkEquation(val, phase):
        # radius = 0.35
        x = radius * numpy.sin(rate * val)
        y = radius * numpy.cos(rate * val - phase)
        return (x, y)

    def yawEquation(val):
        gradient = 2 * numpy.pi / 2 * numpy.pi
        return ((numpy.pi * 2) - val * rate) % (2 * numpy.pi)
#     times = np.arange(0, 100, timesteps)
    x = []
    y = []
    z = []

    ax = []
    ay = []
    az = []

    yaw = []
    roll = []
    pitch = []

    yawRate = []
    rollRate = []
    pitchRate = []

    xsReal = []
    ysReal = []
    zsReal = []

    for i, val in enumerate(times):
    	if i % 10 == 0:
            x_pos, y_pos = rtkEquation(val, numpy.pi)
            x.append(x_pos)
            y.append(y_pos)
            yaw.append(yawEquation(val))

        else:
            x.append(0)
            y.append(0)
            yaw.append(0)

        ay.append(rate ** 2 * radius)
        yawRate.append(-rate)
        xsReal.append(rtkEquation(val, numpy.pi)[0])
        ysReal.append(rtkEquation(val, numpy.pi)[1])

    zeros = numpy.zeros(len(x))
    data = numpy.column_stack((x, y, zeros, zeros, ay, zeros, yaw, yawRate, zeros, zeros, zeros, zeros))
    return (data, xsReal, ysReal, zsReal)





    # Wind drift attempt
    # for i in range(len(zs)):
    #     velocities = (numpy.random.normal(1, 2), numpy.random.normal(1, 2))
    #     force = [0.5 * 1.225 * velocities[0]**2 * 0.025, 0.5 * 1.225 * velocities[1]**2 * 0.025]
    #     x_acc, y_acc = force[0] / 10, force[1] / 10
    #     x_velocity, y_velocity = x_velocity + x_acc * timesteps, y_velocity + y_acc * timesteps
    #     x, y = x + x_velocity * timesteps + x_acc * timesteps**2, y + y_velocity * timesteps + y_acc * timesteps**2
    #     if (i % 10 == 0):
    #         xs_noise.append(x + numpy.random.normal(0.01))
    #         ys_noise.append(y + numpy.random.normal(0.01))
    #         zs_noise.append(zs[i] + numpy.random.normal(0.01))
    #     else:
    #         zs_noise.append(0)
    #         xs_noise.append(0)
    #         ys_noise.append(0)
    #     xs.append(x)
    #     ys.append(y)
    #     x_accs.append(x_acc + numpy.random.normal(0, 0.001))
    #     y_accs.append(y_acc + numpy.random.normal(0, 0.001))
    #
    # zero = numpy.zeros(len(xs))
    #
    #
    # data = numpy.column_stack((xs_noise, ys_noise, zs_noise, x_accs, y_accs, zero, zero, zero, zero, zero, zero, zero))
    # return data, xs, ys, zs


# print numpy.shape(genRandomFlightData(0.1))
# print genRandomFlightData(0.1)[:10]











