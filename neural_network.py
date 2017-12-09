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
            input_vector = hidden_layer.layer_outputs(input_vector)
        return self.output_layer.layer_outputs(input_vector)

if __name__ == "__main__":
    network = NeuralNetwork(5, 1, [2,1])
    print ("Before")
    print (network)
    network.forward_propagate([0.2,0.4,10,20,50])
    print("After")
    print (network)