'''
Name: Daniel Zabelin
class: YB1
This neural network predicts if a given person is a male or a female.
Males and females have 2 features: height and weight
Each person is represented by a row in the inputs array
The outputs:
male = 0
female = 1

resources:
https://github.com/miloharper/multi-layer-neural-network/blob/master/main.py
https://github.com/vzhou842/neural-network-from-scratch
'''


from numpy import exp, array, random, dot,sum
import matplotlib.pyplot as plt 

#neron layer:
class NeuronLayer():
	
	#initializing the neural network with random synaptic waeights and random bias
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.bias = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

#neural netwrok:
class NeuralNetwork():
	
	#initializing the neural network with the nueron layers:
    def __init__(self, layer1, layer2):
        self.layer1 = layer1 #hidden layer
        self.layer2 = layer2 #output layer
        self.error_history = [] #error history for graph
        self.iteration_list = [] #iterations history for graph

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    #training the nural network with number of training iterations as iterations_count, training set outputs as outputs and training set inputs as inputs
    def train(self, inputs, outputs, iterations_count):
        for iteration in range(iterations_count):
            output_from_layer_1, output_from_layer_2 = self.think(inputs)
            self.back_probegation(inputs,outputs,output_from_layer_1,output_from_layer_2)
            self.iteration_list.append(iteration)
	
    def back_probegation(self,training_set_inputs,training_set_outputs,output_from_layer_1,output_from_layer_2):
		#calculating the errors and bias of each layer:
        layer2_error = training_set_outputs - output_from_layer_2
        layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
        layer2_bias_delta = sum(layer2_delta * self.__sigmoid_derivative(training_set_outputs))
        layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
        layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
        layer1_bias_delta = sum(layer1_delta * self.__sigmoid_derivative(training_set_outputs))
		
		#calculating adjusments:
        layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
        layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

		#assaining adjusments:
        self.layer1.synaptic_weights += 0.1*layer1_adjustment
        self.layer1.bias+=0.1*layer1_bias_delta
        self.layer2.synaptic_weights += 0.1*layer2_adjustment
        self.layer2.bias+=0.1*layer2_bias_delta
        
        #adding the error to history for the graph
        self.error_history.append(((layer2_error) ** 2).mean())
			
    # The neural network "thinks" (also feed farward):
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

#Seed the random number generator
random.seed(1)

#Create layer 1 (2 neurons, each with 2 inputs)
layer1 = NeuronLayer(2, 2)

#Create layer 2 (a single neuron with 2 inputs)
layer2 = NeuronLayer(1, 2)

#Combine the layers to create a neural network
neural_network = NeuralNetwork(layer1, layer2)

#assigning training set inputs and expected outputs:
training_set_inputs = array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
training_set_outputs = array([[1,0,0,1]]).T

#training the neural network with 10000 iterations and with the assigned inputs and outputs
neural_network.train(training_set_inputs, training_set_outputs, 10000)

#Test the neural network with a new situation.
#testing emily: 128 pounds, 63 inches
hidden_state, output = neural_network.think(array([-7, -3]))
if output > 0.5:
	print("its a female",output)
elif output < 0.5:
	print("its a male",output)
else:
	print("i cant predict the gender:",output)
#testing frank: 155 pounds, 68 inches
hidden_state, output = neural_network.think(array([20, 2]))
if output > 0.5:
	print("its a female",output)
elif output < 0.5:
	print("its a male",output)
else:
	print("i cant predict the gender:",output)

#printing the 
plt.figure(figsize=(15,5))
plt.plot(neural_network.iteration_list, neural_network.error_history)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
