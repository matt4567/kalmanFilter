from __future__ import division
import numpy
import matplotlib.pyplot as plt


def cos(x):
    return numpy.cos(x)


def sin(x):
    return numpy.sin(x)

def getTimes(df):
    time = 0
    timeIntervals = []
    times = []
    for index, row in df.iterrows():
        times.append(time)
        timeIntervals.append(row['dt'])
        time += row['dt']
    return times, timeIntervals


def transformAngles(x, y, z, yaw, roll, pitch, inverse):
    """Transform the angular vector to the drones reference frame and back to the inertial reference frame."""
    # Translation vectors in x, y and z.
    x_translation = numpy.array([[1, 0, 0],
                                 [0, cos(pitch), -sin(pitch)],
                                 [0, sin(pitch), cos(pitch)]])
    y_translation = numpy.array([[cos(roll), 0, sin(roll)],
                                 [0, 1, 0],
                                 [-sin(roll), 0, cos(roll)]])
    z_translation = numpy.array([[cos(yaw), - sin(yaw), 0],
                                 [sin(yaw), cos(yaw), 0],
                                 [0, 0, 1]])
    # Built translation matrix (or transpose if we are undoing)
    R = z_translation.dot(y_translation).dot(x_translation)
    if (inverse):
        R = R.T
    # do relevant translation
    vector = numpy.array([x,y,z])
    translatedVals =  R.dot(vector)

    # final translated accelerations.
    x_prime = translatedVals[0]
    y_prime = translatedVals[1]
    z_prime = translatedVals[2]

    return (x_prime, y_prime, z_prime)


def precisionAnalysis(x_kal, y_kal, z_kal,  x_true, y_true, z_true):
    """Measure rms displacement between predicted and actual positions."""
    precisions = []
    for i, val in enumerate(x_kal):
        diffX = val - x_true[i]
        diffY = y_kal[i] - y_true[i]
        diffZ = z_kal[i] - z_true[i]
        precision = (diffX ** 2 + diffY ** 2 + diffZ ** 2) ** (0.5)
        precisions.append(precision)

    return numpy.mean(precisions)

def precisionAnalysis2d(x_kal, y_kal,  x_true, y_true):
    """Measure rms displacement between predicted and actual positions in only x and y."""
    precisions = []
    for i, val in enumerate(x_kal):
        diffX = val - x_true[i]
        diffY = y_kal[i] - y_true[i]
        precision = (diffX ** 2 + diffY ** 2) ** (0.5)
        precisions.append(precision)

    return numpy.mean(precisions)

def highPassFilter(acceleration, isOn, timesteps, gradient, cutoff):
    """High pass filter for filtering accelerations"""
    if (isOn == False):
        return acceleration

    print gradient, cutoff
    # plt.plot(acceleration)
    # plt.show()


    fourier = [1] * 100000
    def equation(x):
        return numpy.sin(x) + numpy.sin(2 * x) + numpy.sin(3 * x) + numpy.sin(4 * x)

    xs = numpy.arange(0, 4 * numpy.pi, 0.01)
    sample = equation(xs)
    fourier = numpy.fft.fft(sample)
    freq = numpy.fft.fftfreq(len(fourier), 0.01)
    # freq = numpy.fft.fftshift(freq)
    fourier_old = fourier.copy()

    for i in range(len(freq)):
        # if (freq[i] < cutoff and freq[i] > -cutoff):
        filteredVal = filter(freq[i], 3, 1 / 3)
        fourier[i] = filteredVal * fourier[i]

    ones = [1] * 10000
    freqones = numpy.fft.fftfreq(len(ones), 0.01)
    for i in range(len(freqones)):
        # if (freq[i] < cutoff and freq[i] > -cutoff):
        filteredVal = filter(freqones[i], gradient, cutoff)
        ones[i] = filteredVal * ones[i]
    #
    results = numpy.fft.ifft(fourier)
    plt.subplot(411)
    plt.title("before")
    plt.xlim(-3, 3)
    plt.plot(freq, fourier_old)
    plt.plot(freqones, ones)
    plt.subplot(412)
    plt.title("after")
    plt.xlim(-3, 3)
    plt.plot(freq, fourier)
    plt.subplot(413)
    plt.title("before plot")
    plt.plot(equation(xs))
    plt.subplot(414)
    plt.title("after plot")
    plt.plot(results)
    plt.show()

    # print len(acceleration)
    # acceleration = [1] * len(acceleration)
    fourier = numpy.fft.fft(acceleration)
    freq = numpy.fft.fftfreq(len(fourier), timesteps)
    # numpy.fft.fftshift(freq)
    fourier_old = fourier.copy()
    # plt.plot(freq, fourier)
    # plt.show()
    # cutoff = filterFreq
    # todo - this needs looking at.
    # todo - maybe you want an absolute value
    # fourier[:] = 1
    for i in range(len(freq)):
        # if (freq[i] < cutoff and freq[i] > -cutoff):
        filteredVal = filter(freq[i], gradient, cutoff)
        fourier[i] = filteredVal * fourier[i]
    # fourier[0] = 0
    # results = numpy.fft.ifft(fourier)
    # plt.subplot(411)
    # plt.title("before")
    # plt.xlim(-3, 3)
    # # plt.ylim(0,2)
    # plt.plot(freq, fourier_old)
    # plt.plot(freqones, ones)
    # plt.subplot(412)
    # plt.title("after")
    # plt.xlim(-3, 3)
    # plt.plot(freq, fourier)
    # plt.subplot(413)
    # plt.title("before plot")
    # plt.plot(acceleration)
    # plt.subplot(414)
    # plt.title("after plot")
    # plt.plot(results)
    # plt.show()
    return numpy.fft.ifft(fourier)

# def filter(val, gradient, cutoff):
#     if (val <= cutoff and val >= -cutoff):
#         return 0
#     elif (val >= cutoff * gradient or val <- -cutoff * gradient):
#         return 1
#     elif (val < cutoff and ):
#         return numpy.tanh(gradient * (val - cutoff))
#     elif (val < -cutoff):
#         return numpy.tanh(-gradient * (val + cutoff))

def filter(i, gradient, cutoff):
    if (i < cutoff and i > -cutoff): return 0
    else:return 1
    if (i >= cutoff * 5 *gradient or i <= -cutoff * 5 *gradient):
        return 1
    elif (i <= cutoff and i >= 0):
        return 0
    elif (i >= -cutoff and i <= 0):
        return 0
    elif (i > 0):
        return numpy.tanh(gradient * (i - cutoff))
    else:
        return numpy.tanh(-gradient * (i + cutoff))
def filterold(i, gradient, cutoff):
    if (i >= 5 * cutoff * gradient or i <= -5 * cutoff * gradient):
        return 1
    elif (i <= cutoff and i >= 0):
        return 0
    elif (i >= -cutoff and i <= 0):
        return 0
    elif (i > 0):
        return numpy.tanh(gradient * (i - cutoff))
    else:
        return numpy.tanh(-gradient * (i + cutoff))




# print transformAngles(0, 1, 0, 3 * numpy.pi/2, 0, 0, True)
# print transformAngles(0.1, 0, 0, 0.244, 0, 0, False)