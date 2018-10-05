import matplotlib.pyplot as mat
import numpy

from class_neuralNetwork import NeuralNetwork

# 28 * 28 pixels
input_nodes = 28 * 28
# random due to 10 output nodes
hidden_nodes = 100
# digits 0 t 9
output_node = 10

learning_rate = 0.5

nn = NeuralNetwork(input_nodes, hidden_nodes, output_node, learning_rate)

# read train data csv
training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# show amount of train datasets
print('Train datasets: ' + str(len(training_data_list)))

# show example of test data
# all_values = training_data_list[0].split(',')
# image_array = numpy.asfarray(all_values[1:]).reshape(28,28)
# img = mat.imshow(image_array, cmap='Greys', interpolation='None')
# mat.show(img)

## train the neurral network
for record in training_data_list:
    # split dataset by comma
    all_values = record.split(',')
    # scale input to range 0.01 to 1.00
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create target output values
    targets = numpy.zeros(output_node) + 0.01
    # all_values[0] is the target label for the digit
    targets[int(all_values[0])] = 0.99
    nn.train(inputs, targets)

    pass
print('Training done =)')


# test trained network
test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# get first record
all_values = test_data_list[1].split(',')


image_array = numpy.asfarray(all_values[1:]).reshape(28,28)
mat.show(mat.imshow(image_array, cmap='Greys', interpolation='None'))

# outputs
print('Digit: ' + all_values[0])
print(nn.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01))
