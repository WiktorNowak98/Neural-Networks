import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import datasets

# srodek_ciezkosci = sum(wszystkie_dane) / ilość_danych
# dane_znormalizowane_i = srodek_ciezkosci - dane_i / norma(dane)

def Normalizacja_danych(dane):

    srodek_ciezkosci = 0
    s = 0

    for i in range(len(dane)):
        s += dane[i]
    srodek_ciezkosci = s / len(dane)
    dane_std = []
    for i in range(len(dane)):
        element_normalny = srodek_ciezkosci - dane[i]
        dane_std.append(element_normalny / np.linalg.norm(element_normalny))

    return dane_std

def JaneKohonen(dane, klasy, std = True, alfa = 0.5, miara_podob = 1, mod_alfa = 1, iteracje = 50, C = 2, C2 = 3):

    # 1) Definicja p wektorów reprezentantów w1, w2, ... , wp. Współczynnik alfa należy (0.1, 0.7)

    wektory_rep = []
    for i in range(klasy):
        wektory_rep.append(np.ones_like(dane[0]) / np.linalg.norm(np.ones_like(dane[0])))
    wektory_rep_tab = np.array(wektory_rep)

    # 2) Normalizacja danych lub jej brak

    if std:
        dane_std = Normalizacja_danych(dane)
    else:
        dane_std = dane

    # 3) Wybór elementów do modyfikacji

    alfa_pom = alfa
    for a in range(iteracje):
        tablica_do_modyfikacji = []

        if miara_podob == 1:
            for i in range(len(dane_std)):
                tab_max = []
                for j in range(klasy):
                    tab_max.append(np.dot(wektory_rep_tab[j], dane_std[i]))
                _max = np.where(tab_max == np.amax(tab_max))
                tablica_do_modyfikacji.append(_max[0][0])

        if miara_podob == 2:
            for i in range(len(dane_std)):
                tab_min = []
                for j in range(klasy):
                    tab_min.append(np.linalg.norm(wektory_rep_tab[j] - dane_std[i]))
                _min = np.where( tab_min == np.amin(tab_min))
                tablica_do_modyfikacji.append(_min[0][0])

        if miara_podob == 3:
            for i in range(len(dane_std)):
                tab_min = []
                for j in range(klasy):
                    suma = 0
                    for k in range(len(wektory_rep_tab[0])):
                        suma += abs(wektory_rep_tab[j][k] - dane_std[i][k])
                    tab_min.append(math.sqrt(suma))
                _min = np.where(tab_min == np.amin(tab_min))
                tablica_do_modyfikacji.append(_min[0][0])

        indeksy_do_modyfikacji = np.array(tablica_do_modyfikacji)

        # 4) Modyfikacja wektorów

        for i in range(len(dane_std)):
            wektory_rep_tab[indeksy_do_modyfikacji[i]] = wektory_rep_tab[indeksy_do_modyfikacji[i]] + alfa_pom * (dane_std[i] - wektory_rep_tab[indeksy_do_modyfikacji[i]])
            wektory_rep_tab[indeksy_do_modyfikacji[i]] = (wektory_rep_tab[indeksy_do_modyfikacji[i]]) / (np.linalg.norm(wektory_rep_tab[indeksy_do_modyfikacji[i]]))
            if mod_alfa == 1:
                alfa_pom = alfa / (iteracje - a)
            if mod_alfa == 2:
                alfa_pom = alfa * math.exp(-C * a)
            if mod_alfa == 3:
                alfa_pom = C/(C2 + a)

    return wektory_rep_tab

#----------------------- Dane Iris Normalizowane -----------------------#

iris = datasets.load_iris()
dane = iris.data
dane_std = Normalizacja_danych(dane)
dane_std = np.array(dane_std)

liczba_klas = 3
mod_alfa = 1
alfa = 0.5
miara_podobienstwa = 2
iteracje = 100
wektory_rep_tab = JaneKohonen(dane, liczba_klas, alfa, miara_podobienstwa, mod_alfa, iteracje)

ctr = [0, 0, 0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = zip(ctr, ctr, ctr)
u, v, w = zip(wektory_rep_tab[0, 0:3], wektory_rep_tab[1, 0:3], wektory_rep_tab[2, 0:3])
ax.quiver(x, y, z, u, v, w)
ax.scatter(dane_std[:, 0], dane_std[:, 1], dane_std[:, 2])
plt.show()

#----------------------- Dane Iris Nienormalizowane -----------------------#

std = False
wektory_rep_tab = JaneKohonen(dane, liczba_klas, std)

ctr = [0, 0, 0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = zip(ctr, ctr, ctr)
u, v, w = zip(wektory_rep_tab[0, 0:3], wektory_rep_tab[1, 0:3], wektory_rep_tab[2, 0:3])
ax.quiver(x, y, z, u, v, w)
ax.scatter(dane[:, 0], dane[:, 1], dane[:, 2])
plt.show()

#----------------------- Dane Giełdowe Normalizowane-----------------------#

dane_gieldowe = pd.read_csv('WYNN_data.csv')
wskazniki = dane_gieldowe[['close', 'low', 'high', 'open']]
dane_gieldowe = np.array(wskazniki)
dane_gieldowe_std = Normalizacja_danych(dane_gieldowe)
dane_gieldowe_std = np.array(dane_gieldowe_std)

liczba_klas = 5
wektory_rep_tab = JaneKohonen(dane_gieldowe, liczba_klas)

ctr = [0, 0, 0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = zip(ctr, ctr, ctr, ctr, ctr)
u, v, w = zip(wektory_rep_tab[0, 0:3], wektory_rep_tab[1, 0:3], wektory_rep_tab[2, 0:3], wektory_rep_tab[3, 0:3], wektory_rep_tab[4, 0:3])
ax.quiver(x, y, z, u, v, w)
ax.scatter(dane_gieldowe_std[:, 0], dane_gieldowe_std[:, 1], dane_gieldowe_std[:, 2])
plt.show()

#----------------------- Dane Giełdowe Nienormalizowane -----------------------#

std = False
wektory_rep_tab = JaneKohonen(dane_gieldowe, liczba_klas, std)

ctr = [0, 0, 0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = zip(ctr, ctr)
u, v, w = zip(wektory_rep_tab[0, 0:3], wektory_rep_tab[1, 0:3])
ax.quiver(x, y, z, u, v, w)
ax.scatter(dane_gieldowe[:, 0], dane_gieldowe[:, 1], dane_gieldowe[:, 2])
plt.show()
'''