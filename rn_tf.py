import numpy as np
import os
from scipy import stats

# TensorFlow
import tensorflow as tf
from keras import activations
 
print(tf.__version__)

def circulo(num_datos=100, R=1, minimo=0, maximo=1, latitud=0, longitud=0):
    pi = np.pi

    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size=num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size=num_datos) * 2 * pi * 10

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    x = np.round(x + longitud, 3)
    y = np.round(y + latitud, 3)

    df = np.column_stack([x, y])
    return df

N = 250

datos_nuevayork = circulo(num_datos=N, R=1.5, latitud=40.71427, longitud=-74.00597)
datos_cracovia = circulo(num_datos=N, R=1, latitud=50.0614300, longitud=19.9365800)
X = np.concatenate([datos_nuevayork, datos_cracovia])
X = np.round(X, 3)
print ('X : ', X)

y = [0] * N + [1] * N
y = np.array(y).reshape(len(y), 1)
print ('y : ', y)

train_end = int(0.6 * len(X))
#print (train_end)
test_start = int(0.8 * len(X))
#print (test_start)
X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=4, input_shape=[2], activation=activations.relu, name='relu1'),
                                           tf.keras.layers.Dense(units=8, activation=activations.relu, name='relu2'),
                                           tf.keras.layers.Dense(units=1, activation=activations.sigmoid, name='sigmoid')])
linear_model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError)
print(linear_model.summary())

linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
w = linear_model.layers[0].get_weights()[0]
b = linear_model.layers[0].get_weights()[1]
print('W 0', w)
print('b 0', b)
w = linear_model.layers[1].get_weights()[0]
b = linear_model.layers[1].get_weights()[1]
print('W 1', w)
print('b 1', b)
w = linear_model.layers[2].get_weights()[0]
b = linear_model.layers[2].get_weights()[1]
print('W 2', w)
print('b 2', b)

print('predict city 1 : nuevayork')
nuevayork_matrix = tf.constant([ [42.3584300, -71.0597700],
                           [38.8951100,-77.0363700 ], 
                           [39.9523300, -75.1637900] ], tf.float32)

#print(linear_model.predict([[-43.598 -28.107][-46.268 -14.62 ] [-45.154 -3.249] [-46.52 -21.315][-41.719 -10.532][-48.291 -28.376]] ))   
print(linear_model.predict(nuevayork_matrix).tolist() )   
print('predict city 2 : cracovia')
cracovia_matrix = tf.constant([ [50.0343700, 19.2103700],
                           [51.7500000, 19.4666700] ], tf.float32)
print(linear_model.predict(cracovia_matrix).tolist() )