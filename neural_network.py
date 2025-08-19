import numpy as np
import pandas as pd

class Neural_Network:
	def __init__(self, sizes, train_data, train_solution, test_data, test_solution, epochs=1000, learning_rate=0.01):

		self.layers = []
		self.biases = []
		self.weights = []
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.train_data = train_data
		self.train_solution = train_solution
		self.test_data = test_data
		self.test_solution = test_solution
		self.epochs = epochs
		self.learning_rate = learning_rate

		self.generate_Layers()

	def generate_Layers(self):
		for i in range(self.num_layers):
			self.layers.append(np.zeros(self.sizes[i]))
	
	def generate_Weights(self):
		for i in range(self.num_layers - 1):
			self.weights.append(np.random.default_rng([self.sizes[i], self.sizes[i + 1]]))

	def generate_Biases(self):
		for i in range(self.num_layers - 1):
			self.biases.append(np.random.default_rng([self.sizes[i + 1]]))

	def load_weights(self):
		with open("./Saved_Models/weights.pkl", "rb") as file:
			self.weights = pd.read_pickle(file)

	def load_biases(self):
		with open("./Saved_Models/biases.pkl", "rb") as file:
			self.biases = pd.read_pickle(file)

	def relu(self, x):
		return np.maximum(0, x)

	def relu_derivative(self, x):
		return np.where(x > 0, 1, 0)

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0, keepdims=True)

	def feedforward(self, data):
		self.layers[0] = data

		for i in range(1, self.num_layers):
			# Calculate the weighted sum of inputs plus bias
			self.layers[i] = np.dot(self.layers[i], self.weights[i - 1]) + self.biases[i - 1]

			# Apply activation function except for the last layer
			if i != self.num_layers - 1:
				self.layers[i] = self.relu(self.layers[i])