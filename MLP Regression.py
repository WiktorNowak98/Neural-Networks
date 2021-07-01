import numpy as np
import matplotlib as _plt
import matplotlib.pylab as plt
import sklearn
import sklearn_evaluation
import math
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
# Definicja naszej sieci
siec = MLPRegressor(
    hidden_layer_sizes=(13,), activation='logistic', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=5000, shuffle=True,
    random_state=0, tol=0.001, verbose=False, warm_start=False)

# Generowanie danych od 0 do 2 co 0.001 i utworzenie z nich tablicy

x = np.arange(0, 2, 0.001).reshape(-1, 1)
X = x
Amplituda = 20

# Generowanie szumu

szum = np.random.normal(0, 2, x.shape)

# Utworzenie funkcji oraz jej zaszumionej wersji

y = (Amplituda * np.sin(2*x) * np.cos(4*x) + szum).ravel()
y_ref = (Amplituda * np.sin(2*x) * np.cos(4*x)).ravel()
Y = y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

siec.fit(X_train, y_train)
y_pred = siec.predict(X)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=10, c='b', marker='s', label='Dane z dodanym szumem')
ax1.scatter(x, y_pred, s=10, c='r', marker='o', label='Dokonana aproksymacja')
plt.legend()

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)

y_pred = siec.predict(X_test)
ax1.scatter(x, y_ref, s=10, c='b', marker="s", label='Funkcja bez zaszumienia')
ax1.scatter(X_test[:, 0], y_pred, s=10, c='r', marker="o", label='Dokonana aproksymacja')
plt.legend()
plt.show()