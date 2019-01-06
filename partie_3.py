# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:23:55 2018

@author: FMA
"""

import numpy as np
import functions as fct 
import matplotlib.pyplot as plt




""" 3 Perceptron multicouches """


x_mp = np.zeros((3,1))

x_mp[0,0] = 1
x_mp[1,0] = 1
x_mp[2,0] = 1

x_XOR = np.zeros((3,4))

x_XOR[0,:] = [1.0,1.0,1.0,1.0]
x_XOR[1,:] = [0.0,1.0,0.0,1.0]
x_XOR[2,:] = [0.0,0.0,1.0,1.0]

w1_mp = np.array([(+1.0, 2.0), (1.0, 0.5), (-1.0, 1.5)])

w2_mp = np.array([(-1.0), (0.5), (-1.0)])

y_mp = fct.multiperceptron(x_mp, w1_mp, w2_mp, 2)


y_XOR = np.array([(0.0), (1.0), (1.0), (0.0)])

fct.affiche_class(x_XOR[1:,:], y_XOR, 'Jeu de données XOR')

n_iter_XOR = 5001
seuil = 0.0000


w1_mp_new, w2_mp_new, y_res_XOR, Error_value = fct.multiperceptron_widrow(x_XOR, w1_mp, w2_mp, y_XOR, seuil, n_iter_XOR, 'XOR', 10)

fct.affichage_apprentissage_mc1(x_XOR,w1_mp_new,w2_mp_new,Error_value,'Résultat finale multiperceptron table XOR')
