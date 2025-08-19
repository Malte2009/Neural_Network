import numpy as np
import pandas as pd

from neural_network import Neural_Network

train_data = np.array(pd.read_csv("./Dataset/mnist_train.csv"))
tain_solution = train_data.T[0]
train_data = train_data.T[1:]

test_data = np.array(pd.read_csv("./Dataset/mnist_test.csv"))
test_solution = test_data.T[0]
test_data = test_data.T[1:]

Neural_Network = Neural_Network([784, 30, 10], train_data, tain_solution, test_data, test_solution, epochs=1000, learning_rate=0.01)

if input("Use existing parameters? (y/n): ").lower() == "y":
    Neural_Network.load_weights()
    Neural_Network.load_biases()
else:
    Neural_Network.generate_Weights()
    Neural_Network.generate_Layers()