import numpy as np
import pandas as pd

def predice(x, pesos, bias):
    suma = np.dot(x, pesos) + bias
    # Funcion activacion
    return 1 if suma >= 0 else 0


#Perceptron simple de una capa
def perceptron(x, y, apren, generaciones ):
    pesos = np.random.rand(x.shape[1])
    bias = 0
    for epoch in range(generaciones):
        for i in range(len(x)):
            pred = predice(x[i], pesos, bias)
            error = y[i] - pred
            # Update weights and bias
            pesos += apren * error * x[i]
            bias += apren * error
    return pesos, bias

# Entradas y Salidas

#Cargar CSV


# Especifica la ubicación del archivo CSV
archivo_csv = '.\Practica1\XOR_trn.csv'
archivo_csv_pruebas = '.\Practica\XOR_tst.csv'

datos = pd.read_csv(archivo_csv, header=None)
datostest = pd.read_csv(archivo_csv_pruebas, header=None)
# print(datos.head())

x = np.array(datos.iloc[:, :-1])
y = np.array(datos.iloc[:, -1])

# Entrenar el perceptron
apren = 0.1
generaciones = 100
trained_weights, trained_bias = perceptron(x, y, apren, generaciones)

xtest = np.array(datostest.iloc[:, :-1])
ytest = np.array(datostest.iloc[:, -1])

# Probar el perceptron
for i in range(len(xtest)):
    prediction = predice(xtest[i], trained_weights, trained_bias)
    print(f"Input: {xtest[i]}, Predicted: {prediction}, real: {ytest[i]}")
