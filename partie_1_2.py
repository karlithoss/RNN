# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:14:18 2018

@author: FMA
"""
import numpy as np
import functions as fct
import matplotlib.pyplot as plt

"""Remarque : l'execution des fonction d'apprentissage simple et widrow-hoff affiche les résultats obtenus pour chaque itération
              sur une figure qui s'affiche pour un laps de temps donné '1s' et se ferme automatiquement. Pour modifier ceet affichage, 
              il faut allez vers le fichier functions.py et configurer les fonction affichage apprentissage présentent au niveau des fonctions
              d'apprentissage respective"""


#Chargement des datas
x_d1 = fct.readMatFile('td4_d1.mat')

x_d1 = np.transpose(x_d1)

x_d2 = fct.readMatFile('td4_d2.mat') 

x_d2 = np.transpose(x_d2)

""" 1 Mise en place d'un perceptron simple """ 

#création du tabeleau "OU"
x = np.zeros((4,3))

x[0,:] = [1,0,0]
x[1,:] = [1,0,1]
x[2,:] = [1,1,0]
x[3,:] = [1,1,1]

#Poids synaptiques 
w = np.array(([-0.5], [1], [1]))

#vecteur y résultat "OU"
y = np.zeros((1,4))

#Calcul perceptron
i = 0
for xl in x:
    
    y[0,i] = fct.perceptron_simple(xl, w, 0)
    i = i + 1

#Calcul droite séparatrice 
x_ds, y_ds = fct.calcul_pente_separatrice(x,w)

#Affichage résultat
fct.affichage_resultat(x,x_ds,y_ds,'Résultat perceptron avec OU')

#Affiche data 1 class
d1 = 'd1'

yd1_show = np.ones((len(x_d1),1))
yd1_show[0:25,0] = 2
fct.affiche_class(np.transpose(x_d1), yd1_show, d1)

""" 2 Etude de l'apprentissage """

"""Apprentissage simple sur d1"""

n_iter_simple_d1 = 15

#Ajout des '1' pour le 1er poids synaptiques du neurone
x_d1sp_d1 = np.concatenate((np.ones((len(x_d1),1)), x_d1), axis = 1)


#y desiré
yd_d1s = np.ones((len(x_d1),1))
yd_d1s[0:25,0] = -1

#Poids synaptiques
w_d1s = np.array(([-0.5], [1], [1]))

#Apprentissage simple
w_res_d1s, y_res_d1s, Error_value_d1s = fct.apprentissage_simple(x_d1sp_d1,w_d1s,yd_d1s, d1, n_iter_simple_d1)

#Calcul droite séparatrice
x_ds_d1s, y_ds_d1s = fct.calcul_pente_separatrice(x_d1sp_d1,w_res_d1s)

#Affichage résultat
fct.affichage_resultat1(np.transpose(x_d1sp_d1),x_ds_d1s,y_ds_d1s, yd1_show, Error_value_d1s, 'Résultat apprentissage simple d1')


"""Apprentissage widrow sur d1"""

#Ajout des '1' pour le 1er poids synaptiques du neurone
x_d1wp_d1 = np.concatenate((np.ones((len(x_d1),1)), x_d1), axis = 1)



#Poids synaptiques
w_d1w = np.array(([-0.5], [1], [1]))

n_iter_widrow_d1 = 16

#Apprentissage widrow-hoff avec active = tanh(x)
active_w1_d1 = 1

#Y désié
yd_d1w1 = np.ones((len(x_d1),1))
yd_d1w1[0:25,0] = -1.0

w_res_d1w1, y_res_d1w1, r_widrow1, Error_value_d1w1 = fct.apprentissage_widrow(x_d1wp_d1,w_d1w,yd_d1w1, n_iter_widrow_d1, d1, active_w1_d1)

x_ds_d1w1, y_ds_d1w1 = fct.calcul_pente_separatrice(x_d1wp_d1,w_res_d1w1)

#Affichage résultat final
fct.affichage_resultat1(np.transpose(x_d1wp_d1),x_ds_d1w1,y_ds_d1w1, yd1_show, Error_value_d1w1, 'Résultat apprentissage widrow active=tanh(x) d1')


#Apprentissage widrow-hoff avec active = sigmoide(x)
yd_d1w2 = np.ones((len(x_d1),1))
yd_d1w2[0:25,0] = 0.0
active_w2_d1 = 2

w_res_d1w2, y_res_d1w2, r_widrow2, Error_value_d1w2 = fct.apprentissage_widrow(x_d1wp_d1,w_d1w,yd_d1w2, n_iter_widrow_d1, d1, active_w2_d1)

x_ds_d1w2, y_ds_d1w2 = fct.calcul_pente_separatrice(x_d1wp_d1,w_res_d1w2)

fct.affichage_resultat1(np.transpose(x_d1wp_d1),x_ds_d1w2,y_ds_d1w2, yd1_show, Error_value_d1w2, 'Résultat apprentissage widrow active=(1/1+e^-x) d1')



d2 = 'd2'

#Affiche data 1 class 
yd2_show = np.ones((len(x_d2),1))
yd2_show[0:25,0] = 2
fct.affiche_class(np.transpose(x_d2), yd2_show, d2)

"""Apprentissage simple sur d2"""

n_iter_simple_d2 = 151

x_d2sp = np.concatenate((np.ones((len(x_d2),1)), x_d2), axis = 1)

yd_d2s = np.ones((len(x_d2),1))
yd_d2s[0:25,0] = -1.0

w_d2s = np.array(([-0.5], [+1], [1]))

w_res_d2s, y_res_d2s, Error_value_d2s = fct.apprentissage_simple(x_d2sp,w_d2s,yd_d2s, d2, n_iter_simple_d2)

x_ds_d2s, y_ds_d2s = fct.calcul_pente_separatrice(x_d2sp,w_res_d2s)


fct.affichage_resultat1(np.transpose(x_d2sp),x_ds_d2s,y_ds_d2s, yd2_show, Error_value_d2s, 'Résultat apprentissage simple d2')

"""Apprentissage widrow sur d2"""

x_d2wp = np.concatenate((np.ones((len(x_d2),1)), x_d2), axis = 1) 

yd_d2w1 = np.ones((len(x_d2),1))
yd_d2w1[0:25,0] = -1

w_d2w = np.array(([+0.8], [-2], [1]))
n_iter_widrow_d2 = 151
active_w1_d2 = 1

w_res_d2w1, y_res_d2w1, r_widrow1, Error_value_d2w1 = fct.apprentissage_widrow(x_d2wp,w_d2w,yd_d2w1, n_iter_widrow_d2, d2, active_w1_d2)

x_ds_d2w1, y_ds_d2w1 = fct.calcul_pente_separatrice(x_d2wp,w_res_d2w1)


fct.affichage_resultat1(np.transpose(x_d2wp),x_ds_d2w1,y_ds_d2w1, yd2_show, Error_value_d2w1, 'Résultat apprentissage widrow active=tanh(x) d2')


yd_d2w2 = np.ones((len(x_d2),1))
yd_d2w2[0:25,0] = 0.0
active_w2_d2 = 2

n_iter_widrow_d2 = 151

w_res_d2w2, y_res_d2w2, r_widrow2, Error_value_d2w2 = fct.apprentissage_widrow(x_d2wp,w_d2w,yd_d2w2, n_iter_widrow_d2, d2, active_w2_d2)

x_ds_d2w2, y_ds_d2w2 = fct.calcul_pente_separatrice(x_d2wp,w_res_d2w2)

fct.affichage_resultat1(np.transpose(x_d2wp),x_ds_d2w2,y_ds_d2w2, yd2_show, Error_value_d2w2, 'Résultat apprentissage widrow active=(1/1+e^-x) d2')





