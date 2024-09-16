# LINEAR REGRESION
Este proyecto implementa un modelo de regresión lineal simple utilizando Python para predecir el precio de un automóvil en función de su kilometraje (kilómetros)
El objetivo es mostrar cómo se puede entrenar un modelo de regresión lineal desde cero y utilizarlo para realizar predicciones, además de visualizar los resultados.

### Caracteristicas Principales
- ***Entrenamiendo del modelo:*** El modelo se entrena usando descenso de gradiente, actualizando los parámetros theta0 y theta1 en cada iteración. El proceso de entrenamiento minimiza la función de costo para obtener el mejor ajuste posible a los datos.
- ***Normalización de Datos:*** Los datos del conjunto se normalizan utilizando el método de normalización Z-score para mejorar el rendimiento durante el entrenamiento.
- ***Predicción del Modelo:*** Después de entrenar el modelo, se puede predecir el precio de un auto en función del kilometraje ingresado por el usuario.
- ***Cálculo del Error:*** El proyecto calcula manualmente el Error Cuadrático Medio (MSE) para evaluar el rendimiento del modelo.
- ***Visualización:*** El proyecto genera dos gráficos clave:
    - Gráfico de Dispersión con la Línea de Regresión: Muestra los puntos de datos (precio vs. kilometraje) junto con la línea de regresión lineal.
    - Progresión del Descenso de Gradiente: Muestra la evolución de los parámetros theta0 y theta1 a lo largo de las iteraciones, lo que permite observar cómo el modelo converge

[img](https://github.com/johnconh/Linear-Regresion/blob/main/progresion.png)
