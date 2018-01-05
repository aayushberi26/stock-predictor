import random

from math_lib import (
    dot_product,
    sigmoid,
    sigmoid_prime
)

class Neuron:
    '''
    This is a class representing an individual neuron
    '''
    def __init__(self, num_inputs):
        self.output = 0
        # add one to inputs for bias
        self.weights = [random.random() for i in range(num_inputs+1)]
        self.delta = 0

    def __repr__(self):
        neuron_str = ""
        neuron_str += "Weights: " + str(self.weights) + "\n"
        neuron_str += "Output: " + str(self.output) + "\n"
        neuron_str += "Delta: " + str(self.delta)
        return neuron_str

    def set_output(self, input_vector):
        self.output = sigmoid(dot_product(input_vector, self.weights))


    def update_weights(self):
        pass

class NeuronLayer:
    '''
    This is a class representing a layer of neurons
    '''
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [Neuron(num_inputs) for i in range(num_neurons)]

    def __repr__(self):
        layer_str = ""
        for neuron in self.neurons:
            layer_str += "Neuron: \n" + str(neuron) + "\n\n"
        return layer_str

    def layer_outputs(self, input_vector):
        # add bias unit at beginning
        input_vector = [1] + input_vector
        for neuron in self.neurons:
            neuron.set_output(input_vector)
            # print (str(neuron) + "\n")
        return [neuron.output for neuron in self.neurons]

    def layer_delta(self, prev_layer):
        errors = list()
        for i in  range(len(self.neurons)):
            error = 0.0
            for neuron in prev_layer.neurons:
                error += (neuron.weights[i+1] * neuron.delta)
            errors.append(error)
        for index, neuron in enumerate(self.neurons):
            neuron.delta = errors[index] * sigmoid_prime(neuron.output)

    def output_delta(self, expected):
        for index, neuron in enumerate(self.neurons):
            neuron.delta = (expected[index] - neuron.output) * sigmoid_prime(neuron.output)

    def learn(self, input_vector, learning_rate):
        for neuron in self.neurons:
            for index,input_val in enumerate(input_vector):
                neuron.weights[index+1] += learning_rate * neuron.delta * input_val
            neuron.weights[0] += learning_rate * neuron.delta
