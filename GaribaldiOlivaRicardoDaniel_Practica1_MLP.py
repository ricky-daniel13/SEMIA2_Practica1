import numpy as np
import pandas as pd

# Cargar datos de entrenamiento desde el archivo CSV
train_data = pd.read_csv('.\Practica1\XOR_trn.csv', header=None)
X_train = train_data.iloc[:, :-1].values  # Características
y_train = train_data.iloc[:, -1].values   # Etiquetas

# Cargar datos de prueba desde el archivo CSV
test_data = pd.read_csv('.\Practica1\XOR_tst.csv', header=None)
X_test = test_data.iloc[:, :-1].values    # Características
y_test = test_data.iloc[:, -1].values     # Etiquetas



# - Apartir de aqui, el codigo no me pertenece -
# Definir una función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inicializar los pesos y sesgos de la red neuronal
input_size = X_train.shape[1]
hidden_size = 4  # Número de neuronas en la capa oculta
output_size = 1
learning_rate = 0.01

# Inicialización de pesos y sesgos
np.random.seed(0)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

# Entrenamiento de la red neuronal
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Cálculo del error
    error = y_train.reshape(-1, 1) - predicted_output

    # Backpropagation
    d_predicted_output = error * (predicted_output * (1 - predicted_output))
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * (hidden_output * (1 - hidden_output))

    # Actualización de pesos y sesgos
    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Evaluación de la red neuronal en datos de prueba
threshold = 0.5

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

for i in range(len(X_test)):
    # prediction = predice(xtest[i], hidPesos, hidBias, outPesos, outBias)

    hidden_input = np.dot(X_test[i], weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    print(f"Input: {X_test[i]}, Predicted: {predicted_output}, real: {y_test[i]}")


hidden_input = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_output = sigmoid(hidden_input)
output_layer_input = np.dot(hidden_output, weights_hidden_output) + bias_output
predicted_output = sigmoid(output_layer_input)

for i in range(len(X_test)):
    # prediction = predice(xtest[i], hidPesos, hidBias, outPesos, outBias)

    print(f"Input: {X_test[i]}, Predicted: {predicted_output[i]}, real: {y_test[i]}")

# Umbral para la clasificación binaria (0.5)

predicted_labels = (predicted_output > threshold).astype(int)

# Calcular la precisión en datos de prueba
accuracy = np.mean(predicted_labels == y_test.reshape(-1, 1)) * 100
print(f'Precisión en datos de prueba: {accuracy:.2f}%')
