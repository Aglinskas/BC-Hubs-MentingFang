#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:59:20 2019

@author: aidasaglinskas
"""

import numpy as np
from matplotlib import pyplot as plt


fn = '/Users/aidasaglinskas/Desktop/MentingFang/Hub_vec.npy'
hub = np.load(fn)

fn = '/Users/aidasaglinskas/Desktop/MentingFang/ROI_vec.npy'
roi = np.load(fn)

roi.shape
hub.shape

# 14 Subjects 
# 5 HUBS
# 4 ROIs
# 3 runs
# 6 triu 

#[0][1] # Face-Body
#[1][2] # Body-Object
#[2][3] # Object-Scene
#[1][3] # Body-Scene
#[0][3] # Face-Scene
#[0][2] # Face-Object


#def get_X_matrix(roi_ind,run_ind):
##    roi_ind = 0
##    run_ind = 0
#    X = np.zeros([14,6])
#    for subject in range(roi.shape[1]):
#        for dist in range(roi.shape[3]):
#            X[subject,dist] = roi[roi_ind,subject,run_ind,dist]
#    return X
#
#def get_Y_matrix(hub_ind,run_ind):
##    hub_ind = 0
##    run_ind = 0
#    Y = np.zeros([14,6])
#    for subject in range(hub.shape[1]):
#        for dist in range(roi.shape[3]):
#            Y[subject,dist] = hub[hub_ind,subject,run_ind,dist]
#    return Y


def get_X_matrix(roi_ind,run_ind,sub_ind):
#    roi_ind = 0
#    run_ind = 0
    X = np.zeros(6)
    for dist in range(roi.shape[3]):
            X[dist] = roi[roi_ind,sub_ind,run_ind,dist]
    return X

def get_Y_matrix(hub_ind,run_ind,sub_ind):
#    hub_ind = 0
#    run_ind = 0
    Y = np.zeros(6)
    for dist in range(roi.shape[3]):
        Y[dist] = hub[hub_ind,sub_ind,run_ind,dist]
    return Y






hub_ind = 1
sub_ind = 1
run_ind = 1




def get_data(ind_run,sub_ind,ind_hub):
    X1 = get_X_matrix(0,ind_run,sub_ind)
    X2 = get_X_matrix(1,ind_run,sub_ind)
    X3 = get_X_matrix(2,ind_run,sub_ind)
    X4 = get_X_matrix(3,ind_run,sub_ind)
    
    Y = get_Y_matrix(ind_hub,ind_run,sub_ind)
    return Y,X1,X2,X3,X4


Y,X1,X2,X3,X4 = get_data(ind_run=1,sub_ind=3,ind_hub=2)
b = [0,1,1,1,1]
   
def objective(b):
    Y_hat = b[0] + b[1]*X1 + b[2]*X2 + b[3]*X3 + b[4]*X4
    err = Y_hat-Y
    err = np.sum(err)
    return err

def contraint1(X1,X2,X3,X4,b,Y):
    
    err = b[0] + b[1]*X1 + b[2]*X2 + b[3]*X3 + b[4]*X4 - Y
    err = np.sum(err**2)
    
#%%
minmax = (0.1,20.0)
bnds = (minmax,minmax,minmax,minmax,minmax)

con1 = {'type' : 'ineq', 'fun' : contraint1}

sol = minimize(objective,b,method='SLSQP',bounds=bnds)
print(sol.fun)
print(sol.success)
print(sol.x)

b = sol.x
Y_hat = b[0] + b[1]*X1 + b[2]*X2 + b[3]*X3 + b[4]*X4
err = np.sum((Y-Y_hat)**2)

#%% Simple Linear Regression
from scipy.optimize import minimize  

def f(x):
    true_a = 5
    true_c = 2
    y = (true_a*x)+true_c
    return y

x = np.array(range(0,10))
y = np.array([f(i) for i in x])

#params = [5,2]
def objective(params):
    a_hat = params[0]
    c_hat = params[1]
    
    y_hat = x*a_hat+c_hat
    err = y_hat-y
    err = np.sum(err**2)
    return err
    
initGuess = [0,0] # Initialize parameters
sol = minimize(objective,initGuess,method='SLSQP') 
print(sol.fun)
print(sol.x)
#%% Multiple Regression

def f(xs):
    
    b0 = 4
    b = [1,2,3,4]
    
    Y = b0 + b[0]*xs[0] + b[1]*xs[1] + b[2]*xs[2] + b[3]*xs[3]  # + np.random.randn()
    
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
print(sol.success)
print(sol.fun)
print(sol.x)
#%%

def get_X_matrix(roi_ind,run_ind):
#    roi_ind = 0
#    run_ind = 0
    X = np.zeros([14,6])
    for subject in range(roi.shape[1]):
        for dist in range(roi.shape[3]):
            X[subject,dist] = roi[roi_ind,subject,run_ind,dist]
    return X

def get_Y_matrix(hub_ind,run_ind):
#    hub_ind = 0
#    run_ind = 0
    Y = np.zeros([14,6])
    for subject in range(hub.shape[1]):
        for dist in range(roi.shape[3]):
            Y[subject,dist] = hub[hub_ind,subject,run_ind,dist]
    return Y


run_ind = 2
X1,X2,X3,X4 = [get_X_matrix(0,run_ind),get_X_matrix(1,run_ind),get_X_matrix(2,run_ind),get_X_matrix(3,run_ind)]
hub_ind = 1
Y = np.array(get_Y_matrix(hub_ind,run_ind))
params = np.array([1,2,3,4,5])
def objective(params):
    
    b0 = params[0]
    b = params[1:] # from 1 to end
    #params[0] = 0
    
    Y_hat = params[0] + X1*params[1] + X2*params[2] + X3*params[3] + X4*params[4]
    err = Y-Y_hat
    sum_sq = np.sum(err**2)
    
    return sum_sq
    

#initGuess = [0,0,0,0,0]
initGuess = [10,10,10,10,10]
sol = minimize(objective,initGuess,method='SLSQP') 
print(sol.success)
print(sol.fun)
print(sol.x)

b_hat = sol.x / sum(sol.x)
plt.plot(sol.x)
#%% Across subs,runs,and Triu
fn = '/Users/aidasaglinskas/Desktop/MentingFang/Hub_vec.npy'
hub = np.load(fn)

fn = '/Users/aidasaglinskas/Desktop/MentingFang/ROI_vec.npy'
roi = np.load(fn)

hub_ind = 1


hub.shape
roi.shape

roi = roi-np.mean(roi,2,keepdims=True)
hub = hub-np.mean(hub,2,keepdims=True)

roi = roi-np.mean(roi,1,keepdims=True)
hub = hub-np.mean(hub,1,keepdims=True)

roi = roi-np.mean(roi,0,keepdims=True)
hub = hub-np.mean(hub,0,keepdims=True)

def objective(params):
    b0 = params[0]
    b = params[1:]
    #b0=0
    y_hat = b0 + roi[0,:,:,:]*b[0] + roi[1,:,:,:]*b[1] + roi[2,:,:,:]*b[2] + roi[3,:,:,:]*b[3]
    y = hub[hub_ind,:,:,:]
    err = np.sum(y_hat-y**2)
    
    return err

initGuess = [2,2,2,2,2]
sol = minimize(objective,initGuess,method='SLSQP') 
print(sol.success)
print(sol.fun)
print(sol.x)




















    
    
    
    


