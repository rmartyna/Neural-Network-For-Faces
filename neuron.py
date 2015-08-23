import utility
import random
import constants


class Neuron(object):
    def __init__(self, input_size):
        self.weights = []
        for i in range(input_size + 1):
            self.weights.append(random.uniform(constants.WEIGHT_MIN, constants.WEIGHT_MAX))

    def output(self, input_val):
        result = self.weights[0]
        for w, i in zip(self.weights[1:], input_val):
            result += w*i
        return utility.sigmoid(result)

    def __str__(self):
        return "Weights: " + str(self.weights)
