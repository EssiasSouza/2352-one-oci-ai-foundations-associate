# Bloc [4] Importaciones de nuevas librerías
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interactive

# Bloc [5] Función para actualizar y mostrar el gráfico
def update_plot(hidden_layer_size):

    # Generar datos sintéticos (círculo)
    # X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=0)

    # Crear un clasificador Perceptrón Multicapa (MLP)
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation='relu',
        max_iter=3000,
        random_state=1
    )

    # Entrenar el clasificador con los datos
    clf.fit(x, y)

    # Crear una cuadrícula de puntos para la visualización
    # Estos son arreglos unidimensionales de 100 valores cada uno,
    # que representan las coordenadas X e Y de la cuadrícula.
    x_vals = np.linspace(x[:, 0].min() - 0.1, x[:, 0].max() + 0.1, 100)
    y_vals = np.linspace(x[:, 1].min() - 0.1, x[:, 1].max() + 0.1, 100)

    # Los arreglos X_plane e Y_plane tendrán tamaño 100x100,
    # representando una cuadrícula de 10.000 puntos.
    X_plane, Y_plane = np.meshgrid(x_vals, y_vals)

    # Convierte la cuadrícula en una matriz 2D (grid_points)
    # de forma (10000, 2), donde cada fila representa un punto.
    grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))

    # Predecir las clases para todos los puntos de la cuadrícula
    # (para dibujar la frontera de decisión)
    Z = clf.predict(grid_points)

    # Z.reshape(X_plane.shape) transforma Z
    # en una matriz de 100x100.
    Z = Z.reshape(X_plane.shape)

    # Predecir las clases de los datos originales
    y_pred = clf.predict(x)

    # Limpiar el gráfico anterior
    plt.clf()

    # Dibujar la frontera de decisión
    # Se utiliza para visualizar las probabilidades
    # o etiquetas del modelo sobre toda la cuadrícula.
    plt.contourf(
        X_plane,
        Y_plane,
        Z,
        levels=[-0.5, 0.5, 1.5],
        cmap=plt.cm.RdYlGn,
        alpha=0.6
    )

    # Dibujar los puntos originales con sus clases predichas
    # Separar los puntos según la clase predicha
    class_0 = y_pred == 0      # Índices de los puntos clasificados como clase 0
    class_1 = y_pred == 1      # Índices de los puntos clasificados como clase 1

    plt.scatter(
        x[class_0, 0],
        x[class_0, 1],
        c='red',
        edgecolors='k',
        marker='o',
        s=50,
        label='Predicted Class 0'
    )

    plt.scatter(
        x[class_1, 0],
        x[class_1, 1],
        c='green',
        edgecolors='k',
        marker='o',
        s=50,
        label='Predicted Class 1'
    )

    # Agregar etiquetas y título
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(
        f'Límite de decisión y etiquetas predichas (tamaño de la capa oculta)={hidden_layer_size})'
    )
    plt.legend()
    plt.show()


# Bloc [6] Crear un control deslizante para el tamaño de la capa oculta
hidden_layer_size_slider = widgets.IntSlider(
    value=1,
    min=1,
    max=10,
    step=1,
    description='Tamaño de la capa oculta'
)

# Bloc [7] Crear un widget interactivo
interactive_plot = interactive(
    update_plot,
    hidden_layer_size=hidden_layer_size_slider
)

# Bloc [8] Crear un slider para "hidden layer size"
hidden_layer_size_slider = widgets.IntSlider(value=1, min=1, max=10, step=1, description='Hidden Layer Size:')


# Bloc [9] Crear un widget interactivo para ajustar el número de neuronas en la capa oculta
Interactive_plot = interactive(update_plot, hidden_layer_size=hidden_layer_size_slider)

# Bloc [10] Mostrar los widgets
display(interactive_plot)
