# layers.py
import numpy as np

# add drop out, adam optimizer

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((1, output_dim))

        # values saved for backward function
        self.A_prev = None  # input to the layer
        self.Z = None       # linear output
        self.A = None       # activated output
        self.dB = None      # gradient of the bias 
        self.dW = None      # gradient of the weight

    def forward(self, A, function_type='relu'): # A is the input (tensor) at the beginning of the layer
        self.A_prev = A
        self.Z = np.dot(A, self.weights) + self.biases # Z is a linear output

        # Activation function
        if function_type == 'relu':
            self.A = np.maximum(0, self.Z)
        elif function_type == 'linear':
            self.A = self.Z
        else:
            raise ValueError(f"Unknown activation: {function_type}")
        return self.A # A is a non linear output

    def loss_function(self, Y_pred, Y_true):
        loss = np.mean((Y_true - Y_pred) ** 2)
        return loss

    def backward(self, A=None, Y=None, function_type='relu'):
        if Y is not None:  # output layer
            m = self.A_prev.shape[0]  # number of samples
            dA = (2 / m) * (A - Y)
        else:  # hidden layer
            dA = A  # here A is actually dA from next layer

        # compute dZ; activation function
        if function_type == 'relu':
            dZ = dA * (self.Z > 0)
        elif function_type == 'linear':
            dZ = dA
        else:
            raise ValueError(f"Unknown activation: {function_type}")

        self.dW = np.dot(self.A_prev.T, dZ)
        self.dB = np.sum(dZ, axis=0, keepdims=True)

        # Gradient to previous layer
        dA_prev = np.dot(dZ, self.weights.T)
        return dA_prev

    
    def update_parameters(self, learning_rate, optimizer_type='sgd'):
        if optimizer_type == 'sgd':
            self.weights -= learning_rate * self.dW
            self.biases  -= learning_rate * self.dB
        else:
            raise ValueError(f"Unknown activation: {optimizer_type}")