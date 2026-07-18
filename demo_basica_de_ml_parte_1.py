'''
Proceso de Machine Learning

- Cargar datos
- Preprocesar datos
- Entrenar modelo
- Evaluar modelo
- Hacer predicciones
'''

# Importar librerías necesarias
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Cargar Dataset y mostrar las primeras filas
iris_data = pd.read_csv('iris.csv')
iris_data.head()

# Separar datos en características (x) y etiquetas (y)
x = iris_data.drop(columns=['Id', 'Species'])
y = iris_data['Species']
# COLOCAR LINHA EM NOVA CÉLULA
x.head()

# Crear al model de ml
model = LogisticRegression()

# Entrenar el modelo
model.fit(x.values, y)

# Predecir usando el modelo entrenado
predictions = model.predict([[4.6,3.6,1.5,0.2]])

# Imprimir la predicción
print(predictions)
