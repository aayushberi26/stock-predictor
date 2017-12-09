import random

from math_lib import *

class Neuron:
    '''
    This is a class representing an individual neuron
    '''
    def __init__(self, num_inputs):
        self.output = 0
        # add one to inputs for bias
        self.weights = [random.random() for i in range(num_inputs+1)]

    def __repr__(self):
        neuron_str = ""
        neuron_str += "Weights: " + str(self.weights) + "\n"
        neuron_str += "Output " + str(self.output)
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