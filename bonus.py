# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:52:26 2019

@author: FMA
"""

import numpy as np
import functions as fct
import matplotlib.pyplot as plt
import scipy.io 


donne = scipy.io.loadmat('donnee.mat')
data = donne.get('x')
oracle = donne.get('oracle')


#Créer le matrice de y desirées
y_des = np.zeros((np.shape(data)))

s1 = np.array(([1.0], [1.0], [0.0], [0.0], [0.0], [0.0]))
s2 = np.array(([0.0], [0.0], [1.0], [1.0], [0.0], [0.0]))
s3 = np.array(([0.0], [0.0], [0.0], [0.0], [1.0], [1.0]))

for i in range(len(oracle[0,:])):
    
    if (oracle[0,i] == 1):
        y_des[:,i] = s1[:,0]
    elif (oracle[0,i] == 2):
        y_des[:,i] = s2[:,0]
    elif (oracle[0,i] == 3):
        y_des[:,i] = s3[:,0]
        
"""multiperceptron 2 couches à 6 entrées et à 6 sorties"""
##########################
"""Réseau de neurones"""
""" N_HL1 -- N_OL1 """
""" N_HL2 -- N_OL2 """
""" N_HL3 -- N_OL3 """
""" N_HL4 -- N_OL4 """
""" N_HL5 -- N_OL5 """
""" N_HL6 -- N_OL6 """
#########################
#Poids synaptiques couches 1
w1 = np.matrix([[-1.0, 2.0, 1.0, 0.5, 1.0, 0.5], 
                [0.5, -2.0, -2.0, 0.5, 2.0, 2.0],
                [2.0, 1.0, 2.0, -1.0, -1.0, 1.5],
                [-1.0, 2.0, 0.5, -2.0, -0.5, -1.5],
                [2.0, 0.5, 1.0, 2.0, -1.5, 0.5],
                [-0.5, 0.5, 1.5, 1.0, 1.5, 0.5],
                [-2.0, -1.0, 2.0, 0.5, 2.0, 2.0]])
              
#Poids synaptiques couches 1    
w2 = np.matrix([[2.0, 2.0, -1.5, -2.0, -1.0, 1.5], 
                [1.5, 1.0, 1.0, 1.0, 0.5, -0.5],
                [-1.0, 2.0, 1.5, 1.0, 0.5, 0.5],
                [-0.5, -2.0, 1.5, 0.5, -0.5, -2.0],
                [-2.0, -1.5, 0.5, 2.0, 1.5, 1.5],
                [-1.0, -0.5, -0.5, -2.0, -2.0, -1.0],
                [0.5, 1.5, -1.5, -0.5, -1.5, -0.5]])

##Initialisation 
data_x = np.ones((7,np.shape(data)[1]))
data_x[1:,:] = data
Error_value = []
length_data = np.shape(data_x)[1]
alpha = 0.1
wp1 = np.copy(w1)
wp2 = np.copy(w2)
Err_OL = np.ones((6,length_data))
Err_OL_copie = np.copy(Err_OL)

Err_HL = np.zeros((6,1))
y_HL = np.zeros((6,1))
y_OL = np.zeros((6,1))
y_res = np.zeros((6,length_data))

n_iter = 5001
a = 1
seuil = 0.000001
while( ((sum(sum(Err_OL_copie))**2) >= seuil) and (a < n_iter) ):
    
        for i in range(length_data):
            
            #Couche d'entrée Hiden Layer
            for hl in range(6):
                y_HL[hl,0] = fct.perceptron_simple(data_x[:,i],wp1[:,hl],2)
            
            #Couche de sortie Output Layer
            y_HL1 = np.ones((7,1))
            y_HL1[1:,:] = y_HL
            for ol in range(6):
                y_OL[ol,0] = fct.perceptron_simple(y_HL1[:,0],wp2[:,ol],2)
                Err_OL[ol,i] = (y_OL[ol,0] - y_OL[ol,0]**2)*(y_des[ol,i] - y_OL[ol,0])
                Err_OL_copie[ol,i] = abs(Err_OL[ol,i])
                
                y_res[ol,i] = y_OL[ol,0]
                
            #Rétropropagation de l'erreur "couche d'entrée" pour le neurone ol
            
            for b in range(np.shape(wp1)[1]):
                sum_wErr_OL = 0
                for g in range(6):
                    sum_wErr_OL = sum_wErr_OL + wp2[b+1,g]*Err_OL[g,i]
                
                Err_HL[b,0] = (y_HL[b,0] - y_HL[b,0]**2)*sum_wErr_OL
                    
            #calcul des nouveaux poids pour les neurones de la couches d'entrée
            n = 0
            for wk1 in np.transpose(wp1):
                for j in range(len(wk1)):
                    wk1[j] = wk1[j] + alpha*Err_HL[n,0]*data_x[j,i]
                n = n + 1
            
            #calcul des nouveaux poids pour les neurones de la couches de sortie
            n = 0
            for wk2 in np.transpose(wp2):
                for j in range(len(wk2)):
                    wk2[j] = wk2[j] + alpha*Err_OL[n,i]*y_HL1[j,0]
                n = n + 1
                
        Error_value.append(sum(sum(Err_OL_copie)))
        print('itération = ', a)
        a = a + 1

#Affichege de l'évolution de la courbe d'erreur
plt.xlabel('Itération')
plt.ylabel('Error value')
plt.title('Évolution de l erreur en fonction des itération')
plt.plot(Error_value)
plt.show()

