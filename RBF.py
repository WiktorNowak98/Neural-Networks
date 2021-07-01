import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def Normalizacja_danych(dane):
    s = 0
    for i in range(len(dane)):
        s += dane[i]
    srodek_ciezkosci = s / len(dane)
    dane_std = []
    for i in range(len(dane)):
        element_normalny = srodek_ciezkosci - dane[i]
        dane_std.append(element_normalny / np.linalg.norm(element_normalny))

    return dane_std

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

def JaneKohonen(dane, klasy, std = True, alfa = 0.5, miara_podob = 1, mod_alfa = 1, iteracje = 50, C = 2, C2 = 3):

    # 1) Definicja p wektorów reprezentantów w1, w2, ... , wp. Współczynnik alfa należy (0.1, 0.7)
    rgen = np.random.RandomState(3)
    wektory_rep = []
    for i in range(klasy):
        wtemp = rgen.normal(loc=0.0, scale=0.01,size=len(dane[0]))
        wektory_rep.append(wtemp / np.linalg.norm(wtemp))
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

# Wyznaczenie promienia funkcji aktywacji na podstawie rozpietosci zbiorów
# Szukamy najwiekszej odleglosci pomiedzy elementami danych wejsciowych aby poznac rozpietosc zbioru

def Promien_funkcji_aktywacji(centra_danych):
    max = 0
    for i in range(len(centra_danych)):
        for j in range(len(centra_danych)):
            d = np.linalg.norm(centra_danych[i] - centra_danych[j])
            if (max < d):
                max = d
    d = max
    l = 3
    promien = d/l
    return promien

# Wyznaczenie elementów macierzy Teta
# Dla kazdego elementu danych dla kazdego centrum wyznaczamy wartosc aktywacji funkcji

def Policz_teta(dane, centra_danych, promien):
    Teta = []
    for i in range(len(dane)):
        temp = []
        for j in range(len(centra_danych)):
            wartosc_aktywacji = np.exp(-np.square(np.linalg.norm(dane[i] - centra_danych[j])) / (promien))
            temp.append(wartosc_aktywacji)
        Teta.append(temp)
    Teta = np.array(Teta)
    return Teta

def Policz_wektor_wag(y, Teta):
    wektor_wag = np.dot((np.linalg.pinv(np.dot(np.transpose(Teta), Teta))), (np.dot(np.transpose(Teta), y)))
    return wektor_wag

def Policz_funkcje_aktywacji(wektor_wag, dane, centra_danych, promien):
    y = 0
    for i in range(len(wektor_wag)):
        y = y + (wektor_wag[i] * np.exp(-np.square(np.linalg.norm(dane - centra_danych[i])) / (promien)))
    return y

# Uczenie sieci RBF:
# 1. Ustalenie liczby neuronów w warstwie ukrytej i dobranie centrów funkcji radialnych (korzystając z algorytmu Kohenena),
# 2. Wyznaczenie szerokości funkcji radialnych każdego neuronu ukrytego,
# 3. Obliczenie wag powiązań między warstwamą ukrytą i wyjściową.

dane_gieldowe = pd.read_csv('WYNN_data.csv')
wskazniki = dane_gieldowe[['close', 'low', 'high', 'volume', 'open']]

wskaznik_pred = 'low'
horyzont_pred = 3

X_train, y = przetwarzanie_danych(wskazniki, wskaznik_pred, horyzont_pred)
X_std = Normalizacja_danych(X_train)

klasy = 120

centra_danych = JaneKohonen(X_train, klasy, alfa=0.8, miara_podob=2, mod_alfa=2, iteracje=50)
promien = Promien_funkcji_aktywacji(centra_danych)
Teta = Policz_teta(X_std, centra_danych, promien)
wektor_wag = Policz_wektor_wag(y, Teta)

wektor_pred = []
for i in range(len(X_std)):
    funkcja_aktywacji = Policz_funkcje_aktywacji(wektor_wag, X_std[i], centra_danych, promien)
    wektor_pred.append(funkcja_aktywacji)

t = range(0, len(wektor_pred))
plt.plot(t, wektor_pred, label = 'Predykcja')
t2 = range(0, len(y))
plt.plot(t2, y, label = 'Rzeczywistosc')
plt.xlabel("próbka")
plt.ylabel("wskaźnik" + wskaznik_pred)
plt.title("liczba klas: " + str(klasy) + ", promien: " + str(round(promien, 3)) + ", horyzont predykcji: " + str(horyzont_pred))
plt.legend()

# Funkcja strat i kryterium oceny jakości predykcji MSE

pred = np.dot(Teta, wektor_wag)
loss = 0
for i in range(len(pred)):
    loss = loss + (math.pow((y[i] - pred[i]), 2))/len(pred)

print("Błąd MSE dla danego problemu wynosi: ")
print(loss)
plt.show()

