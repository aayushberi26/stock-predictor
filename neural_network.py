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

    def predict(self, entry):
        outputs = self.forward_propagate(entry[:-1])
        # return output class that network is most confident of
        return outputs.index(max(outputs))

if __name__ == "__main__":
#     one_layer = [
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.01,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.1,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.25,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.5,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.75,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.9,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 1.0,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.01,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.1,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.25,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.5,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.75,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.9,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 1.0,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.01,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.1,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.25,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.5,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.75,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.9,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 1.0,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.01,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.1,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.25,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.5,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.75,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 0.9,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26],
#          'learning_rate': 1.0,
#          'epochs': 10000}
#     ]

#     two_layer = [
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.01,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.1,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.25,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.5,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.75,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.9,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 1.0,
#          'epochs': 10},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.01,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.1,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.25,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.5,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.75,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.9,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 1.0,
#          'epochs': 100},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.01,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.1,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.25,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.5,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.75,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.9,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 1.0,
#          'epochs': 1000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.01,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.1,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.25,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.5,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.75,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 0.9,
#          'epochs': 10000},
#         {'num_inputs': 50,
#          'num_outputs': 2,
#          'hidden_layers': [26, 26],
#          'learning_rate': 1.0,
#          'epochs': 10000}
#     ]

    # for params in one_layer:
    #     nn = NeuralNetwork(params['num_inputs'], params['num_outputs'], params['hidden_layers'])
    #     nn.train(dataset, params['learning_rate'], params['epochs'])
    #     params['model'] = nn
    # ]

    # for params in two_layer:
    #     nn = NeuralNetwork(params['num_inputs'], params['num_outputs'], params['hidden_layers'])
    #     nn.train(dataset, params['learning_rate'], params['epochs'])
    #     params['model'] = nn
    # ]



    dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = 2
    nn = NeuralNetwork(n_inputs, n_outputs, [26, 26])
    nn.train(dataset, 0.5, 10000)
    for row in dataset:
        prediction = nn.predict(row)
        print('Expected=%d, Got=%d' % (row[-1], prediction))
    for layer in nn.hidden_layers:
        print(str(layer))
    print(str(nn.output_layer))
