#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:50:37 2019

@author: aidasaglinskas
"""

import numpy as np
from matplotlib import pyplot as plt

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


fn = '/Users/aidasaglinskas/Desktop/MentingFang/Hub_vec.npy'
hub = np.load(fn)

fn = '/Users/aidasaglinskas/Desktop/MentingFang/ROI_vec.npy'
roi = np.load(fn)

roi.shape
hub.shape

#ROIS: “FACE”, “BODY”, “OBJECT”, “SCENE”
#HUBS: ‘Thalamus’, ‘Middle Cingulate’, ‘Posterior Cingulate’, ‘Angular Gyrus’, ‘Cerebellum’
# FFA EBA LOC PPA
#roi[0][0][1][0]


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

def e_distance(v1,v2):
    #v1 = X[0,:]
    #v2 = Y[0,:]
    dist = np.sqrt(np.sum((v1-v2)**2));
    return dist




X1 = get_X_matrix(0,0)
X2 = get_X_matrix(1,0)
X3 = get_X_matrix(2,0)
X4 = get_X_matrix(3,0)

Y = get_Y_matrix(0,0)
B0,B1,B2,B3,B4 = 0,1,1,1,1



Y = get_Y_matrix(1,0)
best_e = np.inf
val_range = np.linspace(-10,10,20)
beta_hat = [B0,B1,B2,B3,B4];
i = 0
for B0 in val_range:
    for B1 in val_range:
        for B2 in val_range:
            for B3 in val_range:
                for B4 in val_range:
                    i+=1
                    if np.mod(i,10)==0:
                        print(i)

                    Y_hat = B0+X1*B1+X2*B2+X3*B3+X4*B4
                    e = sum(sum((Y-Y_hat)**2))
                    if e < best_e:
                        best_e = e
                        beta_hat = [B0,B1,B2,B3,B4]
print('done')

B0,B1,B2,B3,B4 = beta_hat
Y_hat = B0+X1*B1+X2*B2+X3*B3+X4*B4
e = sum(sum((Y-Y_hat)**2))






X

t = [e_distance(X[i,:],Y[i,:]) for i in range(X.shape[0])]
t = np.array(t)
t.shape
tt = np.zeros(14).transpose()
tt.shape

regress(t,tt)

def regress(X,Y):
    XX = np.matmul(X.transpose(),X)
    XX_inverse = np.linalg.inv(XX)
    XY = np.matmul(X.transpose(),Y)
    
    B_hat = np.matmul(XX_inverse,XY)
    B_hat.shape
    return B_hat















