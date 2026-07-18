'''
En esta demostración, cargamos un conjunto de datos, 
lo dividimos en características y etiquetas, creamos 
un modelo de aprendizaje automático, lo entrenamos y 
lo usamos para realizar predicciones e imprimirlas 
en la consola. Espero que esta demostración les sea 
útil. Ahora, pasaremos a la siguiente demostración.
'''

# Importar librerías necesarias
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 

# Cargar Dataset y mostrar las primeras filas
iris_data = pd.read_csv('iris.csv')
iris_data.head()

# Separar datos en características (x) y etiquetas (y)
x = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']

# Separar datos para entrenamiento y pruebas.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Padronizar las características
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Crear al model de ml
model = LogisticRegression()

# Entrenar el modelo
model.fit(x_train_scaled, y_train)

# Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy del modelo: {accuracy}")   

# Muestrear nuevos datos para la predicción
new_data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [6.3, 2.9, 5.6, 1.8],
                     [4.9, 3.0, 1.4, 0.2]])

# Padronizar los nuevos datos (Ignorar el aviso)
new_data_scaled = scaler.transform(new_data)

# Hacer predicciones
predictions = model.predict(new_data_scaled)

# Imprimir las clases predichas
print("Prediciones", predictions)
