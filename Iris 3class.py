#!/usr/bin/env python
# coding: utf-8

# In[65]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[66]:


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
    
    def predict2(self, X):
        return self.net_input(X)


# In[67]:


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

    def predict2(self, X):
        return self.net_input(X)


# In[68]:


def prepare_output(y, class_value):
    yy = []
    for x in y:
        if x == class_value:
            x = 1
        else:
            x = -1
        yy.append(x)
    y=yy
    return y


# In[69]:


def check(a):
    if a < 50: 
        return 0
    elif a >= 50 and a < 100:
        return 1 
    elif a >= 100:
        return 2


# In[70]:


iris = datasets.load_iris()
X1 = iris.data[0:150, 0:4]
yy1 = iris.target[0:150]
y1 = prepare_output(yy1,0)

X2 = iris.data[0:150, 0:4]
yy2 = iris.target[0:150]
y2 = prepare_output(yy2,1)

X3 = iris.data[0:150, 0:4]
yy3 = iris.target[0:150]
y3 = prepare_output(yy3,2)

print(y1)
print(y2)
print(y3)


# In[71]:


plt.figure(2, figsize=(8,6))
plt.clf()
plt.scatter(X3[:,0], X3[:,1], c=y3, cmap=plt.cm.Set1, edgecolor='k')


# In[72]:


X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=0)
X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=0.2,random_state=0)


# In[73]:


sc=StandardScaler()

sc.fit(X1_train)
X1_train_std=sc.transform(X1_train)
X1_test_std=sc.transform(X1_test)

sc.fit(X2_train)
X2_train_std=sc.transform(X2_train)
X2_test_std=sc.transform(X2_test)

sc.fit(X3_train)
X3_train_std=sc.transform(X3_train)
X3_test_std=sc.transform(X3_test)


# In[74]:


ppn1=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn2=Perceptron(n_iter=100,eta=0.1,random_state=0)
ppn3=Perceptron(n_iter=100,eta=0.1,random_state=0)

ppn1.fit(X1_train_std,y1_train)
ppn2.fit(X2_train_std,y2_train)
ppn3.fit(X3_train_std,y3_train)


# In[75]:


y1_pred=ppn1.predict(X1_test_std)
y2_pred=ppn2.predict(X2_test_std)
y3_pred=ppn3.predict(X3_test_std)


# In[76]:


sc = StandardScaler()
sc.fit(X1)
X1_std = sc.transform(X1)


# In[77]:


print("Błędy perceptron")
err_count = 0
for a in range (150):
    y1_pred0 = ppn1.predict2(X1_std[a,:])
    y2_pred0 = ppn2.predict2(X1_std[a,:])
    y3_pred0 = ppn3.predict2(X1_std[a,:])
    
    error=np.argmax([y1_pred0, y2_pred0, y3_pred0])
    if error!=check(a):   
        print("Błąd dla a= %d" %(a))
        err_count+=1
print("Zle sklasyfikowanych: %d" %(err_count))


# In[78]:


X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=0)
X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=0.2,random_state=0)


# In[79]:


sc=StandardScaler()

sc.fit(X1_train)
X1_train_std=sc.transform(X1_train)
X1_test_std=sc.transform(X1_test)

sc.fit(X2_train)
X2_train_std=sc.transform(X2_train)
X2_test_std=sc.transform(X2_test)

sc.fit(X3_train)
X3_train_std=sc.transform(X3_train)
X3_test_std=sc.transform(X3_test)


# In[80]:


ppn1=AdalineGD(n_iter=1000,eta=0.0001,random_state=0)
ppn2=AdalineGD(n_iter=1000,eta=0.0001,random_state=0)
ppn3=AdalineGD(n_iter=1000,eta=0.0001,random_state=0)

ppn1.fit(X1_train_std,y1_train)
ppn2.fit(X2_train_std,y2_train)
ppn3.fit(X3_train_std,y3_train)


# In[81]:


y1_pred=ppn1.predict(X1_test_std)
y2_pred=ppn2.predict(X2_test_std)
y3_pred=ppn3.predict(X3_test_std)

print("Zle probki: %d" %(y1_test != y1_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y1_test,y1_pred))
print("Zle probki: %d" %(y2_test != y2_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y2_test,y2_pred))
print("Zle probki: %d" %(y3_test != y3_pred).sum())
print ("Dokladnosc: %.2f" % accuracy_score(y3_test,y3_pred))


# In[82]:


sc = StandardScaler()
sc.fit(X1)
X1_std = sc.transform(X1)


# In[83]:


print("Błędy Adaline")
err_count = 0
for a in range (150):
    y1_pred0 = ppn1.predict2(X1_std[a,:])
    y2_pred0 = ppn2.predict2(X1_std[a,:])
    y3_pred0 = ppn3.predict2(X1_std[a,:])

    error=np.argmax([y1_pred0, y2_pred0, y3_pred0])
    if error!=check(a):   
        print("Błąd dla a= %d" %(a))
        err_count+=1
print("Zle sklasyfikowanych: %d" %(err_count))

