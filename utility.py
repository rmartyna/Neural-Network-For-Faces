import math


def sigmoid(value):
    return 1/(1 + math.exp(-value))


def calculate_error(output, target):
    error = 0
    for o, t in zip(output, target):
        error += 0.5 * (o - t) * (o - t)
    return error



