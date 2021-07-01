import numpy as np
import matplotlib as _plt
import matplotlib.pylab as plt
import sklearn
import sklearn_evaluation
import math
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn_evaluation.plot import grid_search
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warning = 'ignore'

def przetwarzanie_danych(dane, wyjscie, k):
    X = []
    Y = []
    for i in range(len(dane) - k - 1):
        x_i_mat = np.array(dane[i:(i + k)])
        x_i = x_i_mat.reshape(x_i_mat.shape[0] * x_i_mat.shape[1])
        y_i = np.array(dane[(i + k):(i + k + 1)][wyjscie])
        X.append(x_i)
        Y.append(y_i)
    return np.array(X), np.array(Y)

# Wczytywanie danych giełdowych, 5 kolumn, definicja wejścia i wyjścia.

dane_gieldowe = pd.read_csv('WYNN_data.csv')

wskazniki = dane_gieldowe[['close', 'low', 'high', 'volume', 'open']]
X, y = przetwarzanie_danych(wskazniki, 'close', 10)

# Standaryzacja danych.

sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)

# Poszukiwanie optymalnego rozwiązanie problemu

warstwy = [50, 100]
eta = [0.01, 0.001, 0.0001]
solver = ['sgd', 'lbfgs']

for k in solver:
    for i in warstwy:
        for j in eta:

            mlp = MLPRegressor(
                hidden_layer_sizes=i, activation='tanh', solver=k,
                learning_rate='adaptive', learning_rate_init=j, max_iter=1000,
                verbose=0)
            mlp.fit(X_train, y_train)
            # Utworzenie kwantów czasowych aby narysować wykres.
            czas = np.arange(0, len(y)).reshape(-1, 1)
            y_pred = mlp.predict(X_std)
            print("Training: warstwa: %d, eta: %.4f, solver: %s" % (i, j, k))
            print("Wyniki treningu: %f\n Wyniki testów %f" % (mlp.score(X_std, y), mlp.score(X_test, y_test)))
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(111)
            plt.plot(czas, y, 'r', label='Odczytane dane spółki')
            plt.plot(czas, y_pred, 'b', label='Dokonana aproksymacja')
            plt.title('solver: %s, glebokosc warstwy ukrytej: %d, eta: %.4f' % (k, i, j))
            plt.legend()
            plt.show()