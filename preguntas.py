"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
  
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = ____[____].____
    X = ____[____].____

    # Imprima las dimensiones de `y`
    print(____.____)

    # Imprima las dimensiones de `X`
    print(____.____)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.reshape(____, ____)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.reshape(____, ____)

    # Imprima las nuevas dimensiones de `y`
    print(____.____)

    # Imprima las nuevas dimensiones de `X`
    print(____.____)
    """
    df = pd.read_csv('gm_2008_region.csv')
    y = df['life'].to_numpy()
    X = df['fertility'].to_numpy()
    print(y.shape)
    print(X.shape)

    y_reshaped = y.reshape(-1,1)
    X_reshaped = X.reshape(-1,1)
    print(y_reshaped.shape)
    print(X_reshaped.shape)

def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
 

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Imprima las dimensiones del DataFrame
    print(____.____)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    print(____)

    # Imprima la media de la columna `life` con 4 decimales.
    print(____)

    # Imprima el tipo de dato de la columna `fertility`.
    print(____)

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    print(____)
"""
    df = pd.read_csv('gm_2008_region.csv')
    print(df.shape)
    print(round(float(df['life'].corr(df['fertility'])),4))
    print(round(float(df['life'].mean()),4))
    print(type(df['fertility']))
    print(round(float(df['GDP'].corr(df['life'])),4))

def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = ____

    # Asigne a la variable los valores de la columna `life`
    y_life = ____

    # Importe LinearRegression
    from ____ import ____

    # Cree una instancia del modelo de regresión lineal
    reg = ____

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = ____(
        ____,
        ____,
    ).reshape(____, _____)

    # Entrene el modelo usando X_fertility y y_life
    reg.fit(____, ____)

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print(____.score(____, ____).round(____))
 """
    from numpy.core.function_base import linspace
    from sklearn.linear_model import LinearRegression
    import numpy as np

    df = pd.read_csv('gm_2008_region.csv')

    X_fertility = df['fertility'].to_numpy()[:,np.newaxis]
    y_life = df['life'].copy().to_numpy()

    reg = LinearRegression()

    prediction_space = np.linspace(X_fertility.min(),X_fertility.max(),139)[:,np.newaxis]

    reg.fit(X_fertility,y_life)

    y_pred = reg.predict(prediction_space)

    print(reg.score(X_fertility,y_life).round(4))

def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
   

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from ____ import ____

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = ____

    # Asigne a la variable los valores de la columna `life`
    y_life = ____

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = ____(
        ____,
        ____,
        test_size=____,
        random_state=____,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = ____

    # Entrene el clasificador usando X_train y y_train
    ____.fit(____, ____)

    # Pronostique y_test usando X_test
    y_pred = ____

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(____(____, ____))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
"""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('gm_2008_region.csv')

    X_fertility = df['fertility'].to_numpy()[:,np.newaxis]
    y_life = df['life'].to_numpy()

    (X_train, X_test, y_train, y_test,) = train_test_split(
                                                 X_fertility,
                                                 y_life,
                                                 test_size=0.2,
                                                 random_state=53,)

    linearRegression = LinearRegression()
    linearRegression.fit(X_train,y_train)
    y_pred = linearRegression.predict(X_test)

    print('R^2: {:6.4f}'.format(linearRegression.score(X_test,y_test)))
    rmse = mean_squared_error(y_test,y_pred, squared = False)
    print('Root Mean Squared Error: {:6.4f}'.format(rmse))
