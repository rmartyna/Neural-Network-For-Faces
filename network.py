import random
import os
import neuron
import constants
import load_data


class Network(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.hidden_layer = []
        for i in range(hidden_size):
            self.hidden_layer.append(neuron.Neuron(input_size))
        self.output_layer = []
        for i in range(output_size):
            self.output_layer.append(neuron.Neuron(hidden_size))
        self.learning_rate = learning_rate

        self.file_names = None
        self.training_ind = None
        self.test_ind = None
        self.add_data()
        self.add_data()

    # Split examples in ratio: 70% training, 30% test
    def add_data(self):
        self.file_names = [os.path.join(constants.DATA_PATH, name) for name in os.listdir(constants.DATA_PATH)]
        files_number = len(self.file_names)
        files_indexes = list(range(files_number))
        random.shuffle(files_indexes)
        self.training_ind = files_indexes[:(files_number//10)*7]
        self.test_ind = files_indexes[(files_number//10)*7:]

    def train_example(self, input_val, target):
        hidden_layer_output = self.calculate_hidden_output(input_val)
        output_layer_output = self.calculate_output_output(hidden_layer_output)
        output_layer_error = self.calculate_output_error(output_layer_output, target)
        self.update_output_weights(hidden_layer_output, output_layer_error)
        hidden_layer_error = self.calculate_hidden_error(hidden_layer_output, output_layer_error)
        self.update_hidden_weights(input_val, hidden_layer_error)

    def calculate_hidden_output(self, input_val):
        hidden_layer_output = []
        for n in self.hidden_layer:
            hidden_layer_output.append(n.output(input_val))
        return hidden_layer_output

    def calculate_output_output(self, hidden_val):
        output_layer_output = []
        for n in self.output_layer:
            output_layer_output.append(n.output(hidden_val))
        return output_layer_output

    def calculate_hidden_error(self, hidden_layer_output, output_layer_error):
        hidden_layer_error = []
        for h_index in range(len(self.hidden_layer)):
            out_error = 0
            for o_index in range(len(self.output_layer)):
                out_error += output_layer_error[o_index] * self.output_layer[o_index].weights[1 + h_index]
            hidden_layer_error.append(hidden_layer_output[h_index] * (1 - hidden_layer_output[h_index]) * out_error)
        return hidden_layer_error

    @staticmethod
    def calculate_output_error(output_layer_output, target):
        output_layer_error = []
        for o, r in zip(output_layer_output, target):
            output_layer_error.append(o * (1 - o) * (r - o))
        return output_layer_error

    def update_hidden_weights(self, input_val, hidden_layer_error):
        for hidden_index in range(len(self.hidden_layer)):
            self.hidden_layer[hidden_index].weights[0] += \
                hidden_layer_error[hidden_index]*self.learning_rate
        for input_index in range(len(input_val)):
            for hidden_index in range(len(self.hidden_layer)):
                self.hidden_layer[hidden_index].weights[1+input_index] += \
                    hidden_layer_error[hidden_index]*input_val[input_index]*self.learning_rate

    def update_output_weights(self, hidden_layer_output, output_layer_error):
        for output_index in range(len(self.output_layer)):
            self.output_layer[output_index].weights[0] += \
                output_layer_error[output_index]*self.learning_rate
        for hidden_index in range(len(hidden_layer_output)):
            for output_index in range(len(self.output_layer)):
                self.output_layer[output_index].weights[1+hidden_index] += \
                    output_layer_error[output_index]*hidden_layer_output[hidden_index]*self.learning_rate

    @staticmethod
    def assign_target_value(test_type, file_name):
        target = None
        if test_type == constants.TEST_SUNGLASSES:
            if "_sunglasses" in file_name:
                target = [constants.MAX_TARGET]
            else:
                target = [constants.MIN_TARGET]
        elif test_type == constants.TEST_DIRECTION:
            if "_up" in file_name:
                target = [constants.MAX_TARGET, constants.MIN_TARGET, constants.MIN_TARGET, constants.MIN_TARGET]
            elif "_straight" in file_name:
                target = [constants.MIN_TARGET, constants.MAX_TARGET, constants.MIN_TARGET, constants.MIN_TARGET]
            elif "_left" in file_name:
                target = [constants.MIN_TARGET, constants.MIN_TARGET, constants.MAX_TARGET, constants.MIN_TARGET]
            else:
                target = [constants.MIN_TARGET, constants.MIN_TARGET, constants.MIN_TARGET, constants.MAX_TARGET]
        elif test_type == constants.TEST_EXPRESSION:
            if "_happy" in file_name:
                target = [constants.MAX_TARGET, constants.MIN_TARGET, constants.MIN_TARGET, constants.MIN_TARGET]
            elif "_sad" in file_name:
                target = [constants.MIN_TARGET, constants.MAX_TARGET, constants.MIN_TARGET, constants.MIN_TARGET]
            elif "_angry" in file_name:
                target = [constants.MIN_TARGET, constants.MIN_TARGET, constants.MAX_TARGET, constants.MIN_TARGET]
            else:
                target = [constants.MIN_TARGET, constants.MIN_TARGET, constants.MIN_TARGET, constants.MAX_TARGET]
        return target

    @staticmethod
    def check_if_correct_output(test_type, target, output):
        if test_type == constants.TEST_SUNGLASSES:
            if (output[0] < constants.AVERAGE_TARGET and target[0] == constants.MIN_TARGET) or \
                    (output[0] >= constants.AVERAGE_TARGET and target[0] == constants.MAX_TARGET):
                return True
            else:
                return False
        elif test_type == constants.TEST_DIRECTION or test_type == constants.TEST_EXPRESSION:
            if (target[0] == constants.MAX_TARGET and max(output) == output[0]) \
                or (target[1] == constants.MAX_TARGET and max(output) == output[1]) \
                or (target[2] == constants.MAX_TARGET and max(output) == output[2]) \
                    or (target[3] == constants.MAX_TARGET and max(output) == output[3]):
                return True
            else:
                return False
        raise Exception("Should not go here!")

    def train(self, test_type):
        print("Initial network:\n" + str(self))
        for time in range(constants.ITERATION_TRAIN):
            ind = random.randint(0, len(self.training_ind) - 1)
            file_no = self.training_ind[ind]
            target = self.assign_target_value(test_type, self.file_names[file_no])
            data = load_data.reduce_dimension(load_data.parse_file(self.file_names[file_no]))
            input_val = []
            for d in data:
                input_val.extend(d)
            # 255 is maximum gray-scale value
            input_val = [i/255.0 for i in input_val]
            self.train_example(input_val, target)
        print("Finished network:\n" + str(self))

    def test(self, test_type):
        correct = 0
        for ind in range(len(self.test_ind)):
            file_no = self.test_ind[ind]
            target = self.assign_target_value(test_type, self.file_names[file_no])
            data = load_data.reduce_dimension(load_data.parse_file(self.file_names[file_no]))
            input_val = []
            for d in data:
                input_val.extend(d)
            # 255 is maximum gray-scale value
            input_val = [i/255.0 for i in input_val]
            output = self.calculate_output_output(self.calculate_hidden_output(input_val))
            if self.check_if_correct_output(test_type, target, output):
                correct += 1
        print("Correctly classified: " + str(correct) + "/" + str(len(self.test_ind)))

    def __str__(self):
        result = ["Output Layer:\n"]
        for n in self.output_layer:
            result.append(str(n) + "\n")
        result.append("\n")
        return "".join(result)
