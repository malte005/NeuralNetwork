import numpy
import scipy.special


# neural network class definition
class neuralNetwork:

    #
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.iNodes = input_nodes
        self.hNodes = hidden_nodes
        self.oNodes = output_nodes

        self.lr = learningrate

        # link weight matrices
        self.wInputHidden = numpy.random.normal(0.0, pow(self.hNodes, -0, 5), (self.hNodes, self.iNodes))
        self.wHiddenOutput = numpy.random.normal(0.0, pow(self.oNodes, -0, 5), (self.oNodes, self.hNodes))

        # sigmoid function
        self.activationFunction = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, input_list, target_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wInputHidden, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activationFunktion(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wHiddenOutput, hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activationFunktion(final_inputs)

        # handle ERRORS
        # output layer error is (target - actual)
        output_errors = target_list - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.wHiddenOutput.T, output_errors)

        # update weights for the links between the hidden and output layers
        self.wHiddenOutput += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                  numpy.transpose(hidden_outputs))
        # update weights for links between input and hidden layers
        self.wInputHidden += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                 numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wInputHidden, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.wHiddenOutput, hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs
