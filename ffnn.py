# ffnn.py
import numpy as np
from layers import DenseLayer

def model(x_train, y_train, x_test, learning_rate, epochs):
    # 1. Network dimensions
    input_dim = x_train.shape[1]
    n_hidden1 = 20
    n_hidden2 = 16
    n_hidden3 = 12
    output_dim = y_train.shape[1]

    # 2. Initialize layers
    layer1 = DenseLayer(input_dim, n_hidden1)
    layer2 = DenseLayer(n_hidden1, n_hidden2)
    layer3 = DenseLayer(n_hidden2, n_hidden3)
    output_layer = DenseLayer(n_hidden3, output_dim)

    # 3. Training loop
    for epoch in range(epochs):
        # --- Forward pass ---
        A1 = layer1.forward(x_train, 'relu')

        A2 = layer2.forward(A1, 'relu')

        A3 = layer3.forward(A2, 'relu')

        A_out = output_layer.forward(A3, 'linear')

        # --- Compute loss ---
        loss = output_layer.loss_function(A_out, y_train)

        # --- Backward pass ---
        dA_out = output_layer.backward(A_out, y_train, 'linear')
        dA3 = layer3.backward(dA_out, None, 'relu')
        dA2 = layer2.backward(dA3, None, 'relu')
        _ = layer1.backward(dA2, None, 'relu')

        # --- Update parameters ---
        layer1.update_parameters(learning_rate)
        layer2.update_parameters(learning_rate)
        layer3.update_parameters(learning_rate)
        output_layer.update_parameters(learning_rate)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # --- Test predictions ---
    A1_test = layer1.forward(x_test, 'relu')

    A2_test = layer2.forward(A1_test, 'relu')

    A3_test = layer3.forward(A2_test, 'relu')

    y_pred = output_layer.forward(A3_test,  'linear')

    return y_pred
