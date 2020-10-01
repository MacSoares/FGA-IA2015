import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Inicialização randomica de pesos e bias
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

print("Pesos iniciais da camada escondida: ", end="")
print(*hidden_weights)
print("Bias inicial da camada escondida: ", end="")
print(*hidden_bias)
print("Pesos iniciais: ", end="")
print(*output_weights)
print("Bias inicial: ", end="")
print(*output_bias)


# Treinamento
for _ in range(epochs):
    # Propagação de cálculo
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Atualizando pesos e bias pós calculo
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("Pesos finais da camada escondida: ", end="")
print(*hidden_weights)
print("Bias final da camada escondida: ", end="")
print(*hidden_bias)
print("Pesos finais: ", end="")
print(*output_weights)
print("Bias final: ", end="")
print(*output_bias)

print("\nResultado calculado pela rede neural: ", end="")
print(*predicted_output)

print("\nResultado esperado de calculo da rede neural: ", end="")
print(*expected_output)
