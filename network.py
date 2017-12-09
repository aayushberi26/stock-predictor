import random

class NeuralNetwork:
    '''
    This is a class representing a neural network
    '''
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = []
        for index,size in enumerate(hidden_layer_sizes):
            # inputs already have bias unit in them
            print ('index' + str(index))
            print ('size' + str(size))
            if index == 0:
                self.hidden_layers.append([{'weights': [random.random() for i in range(self.num_inputs)]} for j in range(size)])
            else:
                self.hidden_layers.append([{'weights': [random.random() for i in range(hidden_layer_sizes[index-1])]} for j in range(size)])

        self.output_layer = [{'weights': [random.random() for i in range(hidden_layer_sizes[-1])]} for j in range(num_outputs)]
    
if __name__ == "__main__":
    network = NeuralNetwork(5, 1, [2,1])
    print (network.hidden_layers)
    print (network.output_layer)


