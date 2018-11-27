import numpy


def cos(x):
    return numpy.cos(x)


def sin(x):
    return numpy.sin(x)


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
