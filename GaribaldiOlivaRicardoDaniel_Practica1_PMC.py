import numpy as np
import pandas as pd


# Función de activación sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoide_derivada(x):
    return x * (1 - x)

def predice(x, hidPesos, hidBias, outPesos, outBias):
    hidden_layer_input = np.dot(x, hidPesos) + hidBias
    hidden_layer_output = sigmoide(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, outPesos) + outBias
    output_layer_output = sigmoide(output_layer_input)
    return output_layer_output


#Perceptron multicapa
def perceptron(x, y, apren, generaciones ):

    inputSize = 2
    hiddenSize = 3
    outputSize = 1

    hidPesos = np.random.uniform(size=(inputSize, hiddenSize))
    hidBias = np.random.uniform(size=(1, hiddenSize))

    outPesos = np.random.uniform(size=(hiddenSize, outputSize))
    outBias = np.random.uniform(size=(1, outputSize))

    for epoch in range(generaciones):

        hidden_layer_input = np.dot(x, hidPesos) + hidBias
        hidLayOutput = sigmoide(hidden_layer_input)

        outLayInput = np.dot(hidLayOutput, outPesos) + outBias
        outLayOutput = sigmoide(outLayInput)

        #print("OutLayOutput: ")
        #print(outLayOutput)

        error = y - outLayOutput

        #print("Outpesos: ")
        #print(outPesos)

        #print("Outpesos traspuesto: ")
        #print(outPesos.T)

        

        # Backpropagation
        d_output = error * sigmoide_derivada(outLayOutput)

        #print("Error: ")
        #print(error)

        #print("d_output: ")
        #print(d_output)
        errorHid = d_output.dot(outPesos.T)
        d_hidden_layer = errorHid * sigmoide_derivada(hidLayOutput)

        # Update weights and biases
        outPesos += hidLayOutput.T.dot(d_output) * apren
        outBias += np.sum(d_output, axis=0, keepdims=True) * apren

        hidPesos += x.T.dot(d_hidden_layer) * apren
        hidBias += np.sum(d_hidden_layer, axis=0, keepdims=True) * apren
    
    return hidPesos, hidBias, outPesos, outBias

# Entradas y Salidas

#Cargar CSV


# Especifica la ubicación del archivo CSV
archivo_csv = '.\Practica1\XOR_trn.csv'
archivo_csv_pruebas = '.\Practica1\XOR_tst.csv'

datos = pd.read_csv(archivo_csv, header=None)
datostest = pd.read_csv(archivo_csv_pruebas, header=None)
print(datos.head())

x = np.array(datos.iloc[:, :-1])
y = np.array(datos.iloc[:, 2],ndmin=2).T

print(x.shape)
print(y.shape)
yyy= np.array([[0], [1], [1], [0]])
print(yyy.shape)

# Entrenar el perceptron

apren = 0.3
generaciones = 1000
hidPesos, hidBias, outPesos, outBias = perceptron(x, y, apren, generaciones)

xtest = np.array(datostest.iloc[:, :-1])
ytest = np.array(datostest.iloc[:, -1])

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

# Probar el perceptron
for i in range(len(xtest)):
    prediction = predice(xtest[i], hidPesos, hidBias, outPesos, outBias)
    print(f"Input: {xtest[i]}, Predicted: {prediction}, real: {ytest[i]}")
