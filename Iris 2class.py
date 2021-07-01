#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0.0
            for xi, target in zip(X,y):
                update = self.eta * (self.predict(xi) - target)
                self.w_[1:] -= update * xi
                self.w_[0] -= update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# In[3]:


class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


# In[4]:


iris = datasets.load_iris()
X = iris.data[50:150, 0:4]
y = iris.target[50:150] 
print(X)
print(y)
#Sprawdzamy wszystkie możliwe kombinacje dla 4 danych, następnie z Iris data w postaci [PL PW SL SW] i wyciągamy odpowiednie
#kolumny 

Plength_Pwidth  = [] #Petal Length/Petal Width    
Plength_Slength = [] #Petal Length/Sepal Length    
Plength_Swidth  = [] #Petal Length/Sepal Width    
Pwidth_Slength  = [] #Petal Width/Sepal Length    
Pwidth_Swidth   = [] #Petal Width/Sepal Width    
Slength_Swidth  = [] #Sepal Length/Sepal Width    

Plength_Pwidth  = X[:,0:2] 
Plength_Slength = X[:,::2] 
Plength_Swidth  = X[:,::3] 
Pwidth_Slength  = X[:,1:3] 
Pwidth_Swidth   = X[:,1:4] 
Pwidth_Swidth   = np.delete(Pwidth_Swidth,1,1)
Slength_Swidth  = X[:,2:4]


# In[5]:


yy = []
for x in y:
    if x == 1:
        x = -1
    else:
        x = 1
    yy.append(x)
y=yy


# In[6]:


X_train,X_test,y_train,y_test=train_test_split(Plength_Pwidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Płatka")
plt.ylabel("Szerokość Płatka")
plt.title("Perceptron")


# In[7]:


X_train,X_test,y_train,y_test=train_test_split(Plength_Slength,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Płatka")
plt.ylabel("Długość Kielicha")
plt.title("Perceptron")


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(Plength_Swidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Płatka")
plt.ylabel("Szerokość Kielicha")
plt.title("Perceptron")


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(Pwidth_Slength,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Szerokość Płatka")
plt.ylabel("Długość Kielicha")
plt.title("Perceptron")


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(Pwidth_Swidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Szerokość Płatka")
plt.ylabel("Szerokość Kielicha")
plt.title("Perceptron")


# In[11]:


X_train,X_test,y_train,y_test=train_test_split(Slength_Swidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Kielicha")
plt.ylabel("Szerokość Płatka")
plt.title("Perceptron")


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(Plength_Pwidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=AdalineGD(n_iter=100,eta=0.01,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Płatka")
plt.ylabel("Szerokość Płatka")
plt.title("Adalaine")


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(Plength_Slength,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=AdalineGD(n_iter=100,eta=0.01,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Płatka")
plt.ylabel("Długość Kielicha")
plt.title("Adaline")


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(Plength_Swidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=AdalineGD(n_iter=100,eta=0.01,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Płatka")
plt.ylabel("Szerokość Kielicha")
plt.title("Adaline")


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(Pwidth_Slength,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=AdalineGD(n_iter=100,eta=0.01,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Szerokość Płatka")
plt.ylabel("Długość Kielicha")
plt.title("Adaline")


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(Pwidth_Swidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=AdalineGD(n_iter=100,eta=0.01,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Szerokość Płatka")
plt.ylabel("Szerokość Kielicha")
plt.title("Adaline")


# In[17]:


X_train,X_test,y_train,y_test=train_test_split(Slength_Swidth,y,test_size=0.2,random_state=0)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

ppn=AdalineGD(n_iter=100,eta=0.01,random_state=0)
ppn.fit(X_train_std,y_train)

y_pred=ppn.predict(X_test_std)
print("Zle probki: %d" %(y_test != y_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y_test,y_pred))

X_combined_std=np.vstack((X_train_std,X_test_std))
y_combined=np.hstack((y_train,y_test))

plot_decision_regions(X=X_combined_std,y=y_combined,feature_index=[0,1],clf=ppn)
plt.xlabel("Długość Kielicha")
plt.ylabel("Szerokość Płatka")
plt.title("Adaline")

