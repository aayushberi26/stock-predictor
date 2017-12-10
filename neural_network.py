from neuron_models import (
    NeuronLayer
)

class NeuralNetwork:
    '''
    This is a class representing a neural network
    '''
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = []
        for index,size in enumerate(hidden_layer_sizes):
            if index == 0:
                self.hidden_layers.append(
                    NeuronLayer(num_inputs, size)
                )
            else:
                self.hidden_layers.append(
                    NeuronLayer(hidden_layer_sizes[index-1], size)
                )
        self.output_layer = NeuronLayer(hidden_layer_sizes[-1], num_outputs)

    def __repr__(self):
        network_str = ""
        network_str += "Inputs: " + str(self.num_inputs) + "\n"
        network_str += "Outputs: " + str(self.num_outputs) + "\n"
        network_str += "Hidden Layers: \n"
        for index,hidden_layer in enumerate(self.hidden_layers):
            network_str += "\nLayer: " + str(index+1) + "\n"
            network_str += str(hidden_layer) + "\n"
        network_str += "Output Layer \n"
        network_str += str(self.output_layer)
        return network_str

    def forward_propagate(self, input_vector):
        # First we iterate through the hidden layers and for each hidden layer
        # we go through the neurons for that hidden layer and apply the function
        for hidden_layer in self.hidden_layers:
            # print (str(hidden_layer))
            input_vector = hidden_layer.layer_outputs(input_vector)
        return self.output_layer.layer_outputs(input_vector)

    def backward_propagate(self, expected):
        # First we iterate through the output layer and calculate the delta
        # Then we move inwards from outer to inner hidden layer and apply
        # the function for calculating error
        self.output_layer.output_delta(expected)
        previous_layer = self.output_layer
        for hidden_layer in  reversed(self.hidden_layers):
            hidden_layer.layer_delta(previous_layer)
            previous_layer = hidden_layer
    
    def update_weights(self, input_vector, learning_rate):
        for index,hidden_layer in enumerate(self.hidden_layers):
            if index == 0:
                # don't want to include the bias vector
                input_vector = input_vector[:-1]
            else:
                input_vector = [neuron.output for neuron in self.hidden_layers[index-1].neurons]
            hidden_layer.learn(input_vector, learning_rate)
        input_vector = [neuron.output for neuron in self.hidden_layers[-1].neurons]
        self.output_layer.learn(input_vector, learning_rate)

    def train(self, training_set, learning_rate, n_epochs):
        for epoch in range(n_epochs):
            sum_error = 0
            for entry in training_set:
                outputs = self.forward_propagate(entry[:-1])
                # set as single output vector with the true output
                if self.num_outputs == 1:
                    expected = [entry[-1]]
                # multiclass
                else:
                    expected = [0 for i in range(self.num_outputs)]
                    expected[entry[-1]] = 1
                errors = [(expected[i] - outputs[i])**2 for i in range(len(expected))]
                sum_error += sum(errors)
                self.backward_propagate(expected)
                self.update_weights(entry, learning_rate)
            print("Epoch=%d, Learning Rate=%.3f, error=%.3f" % (epoch, learning_rate, sum_error))

if __name__ == "__main__":
    dataset = import_data("MSFTSimple.csv")
    data_len = len(dataset)
    training_set = dataset[:round(0.6*data_len)]
    cross_validation_set = dataset[round(0.6*data_len):round(0.8*data_len)]
    test_set = dataset[round(0.8*data_len):]
    n_inputs = len(training_set[0]) - 1
    n_outputs = 2
    nn = NeuralNetwork(n_inputs, n_outputs, [2])
    nn.train(training_set, 0.5, 20)
    # for layer in nn.hidden_layers:
    #     print(str(layer))
    # print(str(nn.output_layer))
