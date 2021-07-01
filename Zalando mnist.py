import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

import sklearn
import pandas as pd

from sklearn import datasets, svm
from sklearn import neural_network
from sklearn import datasets
from sklearn import metrics

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn_evaluation.plot import grid_search
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Wczytanie danych testowych oraz treningowych z plików csv

Test_data = pd.read_csv('fashion-mnist_test.csv')
Train_data = pd.read_csv('fashion-mnist_train.csv')
Test_data_values = Test_data.values
Train_data_values = Train_data.values

# Podział danych na etykiety oraz dane właściwe

etykiety_testowe = Test_data_values[:, 0]
etykiety_treningowe = Train_data_values[:, 0]
dane_testowe = Test_data_values[:, 1::]
dane_treningowe = Train_data_values[:, 1::]

# Standaryzacja danych

sc = StandardScaler()
sc.fit(dane_testowe)
sc.fit(dane_treningowe)

print("Dlugosci tablic z danymi")
print(len(etykiety_testowe))
print(len(dane_testowe))
print(len(etykiety_treningowe))
print(len(dane_treningowe))

dane_testowe_std = sc.transform(dane_testowe)
dane_treningowe_std = sc.transform(dane_treningowe)

# Utworzenie gorącej jedynki (One hot encoder'a)

ohe = OneHotEncoder()
tstl = etykiety_testowe.reshape(-1, 1)
trnl = etykiety_treningowe.reshape(-1, 1)

ohe.fit(tstl)
ohe.fit(trnl)

etykiety_testowe_ohe = ohe.transform(tstl).toarray()
etykiety_treningowe_ohe = ohe.transform(trnl).toarray()

# Użyte solvery, głębokości warstwy ukrytej oraz wielkości współczynnika uczenia

_warstwy = [20, 40, 100]
_eta = [0.1, 0.01, 0.001]
_solver = ['sgd', 'lbfgs']

for k in _solver:
    for i in _warstwy:

        loss_sgd = []

        for j in _eta:

            # Utworzenie Multi layer perceptronu o zadanej głębokości warstwy ukrytej, ecie oraz solverze

            mlp = MLPClassifier(hidden_layer_sizes=(i,), max_iter=500, solver=k, verbose=0, random_state=1, learning_rate_init=j)

            # Ustalenie funkcji aktywacji jako softmax

            mlp.out_activation = 'softmax'

            # Wypisanie wyników uczenia oraz testów

            print("Training: warstwa: %d, eta: %.3f, solver: %s" % (i, j, k))
            mlp.fit(dane_treningowe_std, etykiety_treningowe)
            print("Wyniki uczenia: %f\n Wyniki testów %f" % (mlp.score(dane_treningowe_std, etykiety_treningowe), mlp.score(dane_testowe_std, etykiety_testowe)))

            if k == 'sgd':
                loss_sgd.append(mlp.loss_curve_)

        # Wyznaczenie wykresów funkcji strat dla sgd

        if k == 'sgd':
            plt.plot(loss_sgd[0], label='eta = 0.1')
            plt.plot(loss_sgd[1], label='eta = 0.01')
            plt.plot(loss_sgd[2], label='eta = 0.001')
            plt.ylabel('Strata obliczeniowa')
            plt.xlabel('Iteracja')
            plt.title('Solver: sgd, Ilosc warstw: %d' % i)
            plt.legend()
            plt.grid()
            plt.show()
