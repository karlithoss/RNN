# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:14:18 2018

@author: FMA
"""
import numpy as np
import functions as fct
import matplotlib.pyplot as plt


x_d1 = fct.readMatFile('td4_d1.mat')

x_d1 = np.transpose(x_d1)

x_d2 = fct.readMatFile('td4_d2.mat') 

x_d2 = np.transpose(x_d2)

""" 1 Mise en place d'un perceptron simple """

x = np.zeros((4,3))

x[0,:] = [1,0,0]
x[1,:] = [1,0,1]
x[2,:] = [1,1,0]
x[3,:] = [1,1,1]

w = np.array(([-0.5], [1], [1]))

y = np.zeros((1,4))
i = 0
for xl in x:
    
    y[0,i] = fct.perceptron_simple(xl, w, 0)
    i = i + 1


x_ds, y_ds = fct.calcul_pente_separatrice(x,w)

fct.affichage_resultat(x,x_ds,y_ds,'Résultat perceptron avec OU')



x_d1p = np.concatenate((np.ones((len(x_d1),1)), x_d1), axis = 1)

yd_d1 = np.ones((len(x_d1),1))
yd_d1[0:25,0] = -1

w_d1 = np.array(([-0.5], [1], [1]))

w_res_d1, y_res_d1 = fct.apprentissage_simple(x_d1p,w_d1,yd_d1)

x_ds_d1, y_ds_d1 = fct.calcul_pente_separatrice(x_d1p,w_res_d1)

fct.affichage_resultat(x_d1p,x_ds_d1,y_ds_d1,'Résultat apprentissage simple d1')









