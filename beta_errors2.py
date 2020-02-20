#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:12:02 2019

@author: aidasaglinskas
"""
from scipy.optimize import minimize 
import numpy as np
from matplotlib import pyplot as plt
#%% Multiple Regression
def f(xs):
    
    b0 = 5
    b = [1,5,3,4]
    
    Y = b0 + b[0]*xs[0] + b[1]*xs[1] + b[2]*xs[2] + b[3]*xs[3] + np.random.randn()*300
    
    return Y


X = np.array([np.random.randint(20) for i in range(4)])
Y = np.array(f(X))

for i in range(14):
    xs = np.array([np.random.randint(20) for i in range(4)])
    y = np.array(f(xs))
    
    X = np.vstack((X,xs))
    Y = np.append(Y,y)
    
params = [4,1,2,3,4]
def objective(params):
    
    b0 = params[0]
    b = params[1:]

    y_hat = b0 + X[:,0]*b[0] + X[:,1]*b[1] + X[:,2]*b[2] + X[:,3]*b[3]
    
    err = Y-y_hat
    sum_sq = np.sum(err**2)
    
    return sum_sq


initGuess = [0,1,2,3,4]
sol = minimize(objective,initGuess,method='SLSQP') 
print(sol)
#%%

b = sol.x
from scipy.stats import t
Coefs = b[1:]
SE = np.std(X[:,:],axis=0) / np.sqrt(14)

N = X.shape[0]
p = X.shape[1]

df = N-p
alpha = .5
critical_t = abs(t.ppf((1-.95)/2,df))

CI_upperBound = Coefs+(SE*critical_t)
CI_lowerBound = Coefs-(SE*critical_t)

yerr = np.vstack((CI_lowerBound,CI_upperBound))
xs = np.arange(p)

plt.bar(xs,Coefs)
plt.errorbar(xs,Coefs,yerr=yerr,ecolor='red',fmt='none')














