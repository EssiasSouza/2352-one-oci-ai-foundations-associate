# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generar datos sinteticos (circulos)
x, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=0)

# Representa los puntos con sus etiquetas.
plt.figure(figsize=(8,6))
plt.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.RdYlGn, edgecolors='k', marker='o', s=50)
plt.xlabel('Característica 1 (x[:,0])')
plt.ylabel('Característica 2 (x[:,1])')
plt.title('Generar datos sinteticos (make_circles Dataset)')
plt.colorbar(label='Etiqueta de clase')
plt.show()
