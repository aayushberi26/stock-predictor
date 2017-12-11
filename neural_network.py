from neuron_models import (
    NeuronLayer
)
import vectors
import sys

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
                    expected[int(entry[-1])] = 1
                errors = [(expected[i] - outputs[i])**2 for i in range(len(expected))]
                sum_error += sum(errors)
                self.backward_propagate(expected)
                self.update_weights(entry, learning_rate)
            print("Epoch=%d   Learning Rate=%.3f   error=%.3f ," % (epoch, learning_rate, sum_error))

    def predict(self, entry):
        outputs = self.forward_propagate(entry[:-1])
        # return output class that network is most confident of
        return outputs.index(max(outputs))

if __name__ == "__main__":
    one_layer = [
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.01,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.1,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.25,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.5,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.75,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.9,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 1.0,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.01,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.1,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.25,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.5,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.75,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.9,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 1.0,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.01,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.1,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.25,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.5,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.75,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.9,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 1.0,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.01,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.1,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.25,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.5,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.75,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 0.9,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26],
         'learning_rate': 1.0,
         'epochs': 10000}
    ]

    two_layer = [
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.01,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.1,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.25,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.5,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.75,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.9,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 1.0,
         'epochs': 10},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.01,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.1,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.25,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.5,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.75,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.9,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 1.0,
         'epochs': 100},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.01,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.1,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.25,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.5,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.75,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.9,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 1.0,
         'epochs': 1000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.01,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.1,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.25,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.5,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.75,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 0.9,
         'epochs': 10000},
        {'num_inputs': 50,
         'num_outputs': 2,
         'hidden_layers': [26, 26],
         'learning_rate': 1.0,
         'epochs': 10000}
    ]

 #    dataset = [[2.7810836,2.550537003,0],
	# [1.465489372,2.362125076,0],
	# [3.396561688,4.400293529,0],
	# [1.38807019,1.850220317,0],
	# [3.06407232,3.005305973,0],
	# [7.627531214,2.759262235,1],
	# [5.332441248,2.088626775,1],
	# [6.922596716,1.77106367,1],
	# [8.675418651,-0.242068655,1],
	# [7.673756466,3.508563011,1]]
 #    n_inputs = len(dataset[0]) - 1
 #    n_outputs = 2
 #    nn = NeuralNetwork(n_inputs, n_outputs, [26, 26])
 #    nn.train(dataset, 0.5, 10000)
    saveout = sys.stdout

    dataset = vectors.import_data("MSFTSimple.csv")
    data_len = len(dataset)
    training_set = dataset[:round(0.6*data_len)]
    cross_validation_set = dataset[round(0.6*data_len):round(0.8*data_len)]
    test_set = dataset[round(0.8*data_len):]
    n_inputs = len(training_set[0]) - 1
    n_outputs = 2
    learning_rates = [0.1, 0.25, 0.5]
    epochs = [100, 1000]
    hidden_layers = [([26], "one_layer"), ([26, 26], "two_layer")]
    for learning_rate in learning_rates:
        for epoch in epochs:
            for hidden_layer in hidden_layers:
                base_filename = hidden_layer[1] + "_epoch_" + str(epoch) + "_lrate_" + str(learning_rate) + ".txt"
                nn = NeuralNetwork(n_inputs, n_outputs, hidden_layer[0])
                train_filename = "train_" + base_filename

                f_train = open("output/"+train_filename, 'w')
                sys.stdout = f_train

                nn.train(training_set, learning_rate, epoch)

                sys.stdout = saveout
                f_train.close()

                # store training output in train_filename
                cross_validation_filename = "cross_val_" + base_filename
                f_cross_validation = open("output/"+cross_validation_filename, 'w')
                sys.stdout = f_cross_validation
                wrong_cross_validate = 0
                for row in cross_validation_set:
                    prediction = nn.predict(row)
                    if prediction != row[-1]:
                        wrong += 1
                    print('Expected=%d   Got=%d ,' % (row[-1], prediction))
                    print("Wrong: " + str(wrong_cross_validate) + "/" + str(len(cross_validation_set)))

                sys.stdout = saveout
                f_cross_validation.close()

                    # store cross_validation output in cross_validation_filename
                testset_filename = "test_" + base_filename
                f_test = open("output/"+testset_filename, 'w')
                sys.stdout = f_test
                wrong_test = 0
                for row in test_set:
                    prediction = nn.predict(row)
                    if prediction != row[-1]:
                        wrong += 1
                    print('Expected=%d   Got=%d ,' % (row[-1], prediction))
                    print("Wrong: " + str(wrong_test) + "/" + str(len(test_set)))

                sys.stdout = saveout
                f_test.close()
                    # store test set output in testset_filename



    # for params in one_layer[0:1]:
    #     nn = NeuralNetwork(params['num_inputs'], params['num_outputs'], params['hidden_layers'])
    #     nn.train(training_set, params['learning_rate'], params['epochs'])
    #     params['model'] = nn
    #     for row in cross_validation_set:
    #         prediction = nn.predict(row)
    #         print('Expected=%d, Got=%d' % (row[-1], prediction))
    #     for layer in nn.hidden_layers:
    #         print(str(layer))
    #     print(str(nn.output_layer))

    # dataset = [[0, 0, 0],[0, 1, 1],[1, 0, 1],[1, 1, 0]]
    # nn = NeuralNetwork(2, 2, [5])
    # nn.train(dataset, 0.2, 10000)
    # wrong = 0
    # for row in dataset:
    #     prediction = nn.predict(row)
    #     if prediction != row[-1]:
    #         wrong += 1
    #     print('Expected=%d, Got=%d' % (row[-1], prediction))
    #     print ("Wrong: "  + str(wrong / len(dataset)))

    # for params in two_layer:
    #     nn = NeuralNetwork(params['num_inputs'], params['num_outputs'], params['hidden_layers'])
    #     nn.train(dataset, params['learning_rate'], params['epochs'])
    #     params['model'] = nn
    # ]

    # for row in dataset:
    #     prediction = nn.predict(row)
    #     print('Expected=%d, Got=%d' % (row[-1], prediction))