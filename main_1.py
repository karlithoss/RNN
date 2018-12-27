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

d1 = 'd1'

yd1_show = np.ones((len(x_d1),1))
yd1_show[0:25,0] = 2
fct.affiche_class(np.transpose(x_d1), yd1_show, d1)

""" 2 Etude de l'apprentissage """

"""Apprentissage simple sur d1"""

n_iter_simple_d1 = 15

x_d1sp_d1 = np.concatenate((np.ones((len(x_d1),1)), x_d1), axis = 1)

yd_d1s = np.ones((len(x_d1),1))
yd_d1s[0:25,0] = -1

w_d1s = np.array(([-0.5], [1], [1]))

w_res_d1s, y_res_d1s, Error_value_d1s = fct.apprentissage_simple(x_d1sp_d1,w_d1s,yd_d1s, d1, n_iter_simple_d1)

x_ds_d1s, y_ds_d1s = fct.calcul_pente_separatrice(x_d1sp_d1,w_res_d1s)

fct.affichage_resultat1(np.transpose(x_d1sp_d1),x_ds_d1s,y_ds_d1s, yd1_show, Error_value_d1s, 'Résultat apprentissage simple d1')


"""Apprentissage widrow sur d1"""

x_d1wp_d1 = np.concatenate((np.ones((len(x_d1),1)), x_d1), axis = 1)

yd_d1w = np.ones((len(x_d1),1))
yd_d1w[0:25,0] = -1

w_d1w = np.array(([-0.5], [1], [1]))
n_iter_widrow_d1 = 15

w_res_d1w, y_res_d1w, r_widrow, Error_value_d1w = fct.apprentissage_widrow(x_d1wp_d1,w_d1w,yd_d1w, n_iter_widrow_d1, d1)

x_ds_d1w, y_ds_d1w = fct.calcul_pente_separatrice(x_d1wp_d1,w_res_d1w)

fct.affichage_resultat1(np.transpose(x_d1wp_d1),x_ds_d1w,y_ds_d1w, yd1_show, Error_value_d1w, 'Résultat apprentissage widrow d1')


d2 = 'd2'

yd2_show = np.ones((len(x_d2),1))
yd2_show[0:25,0] = 2
fct.affiche_class(np.transpose(x_d2), yd2_show, d2)

"""Apprentissage simple sur d2"""

n_iter_simple_d2 = 25

x_d2sp = np.concatenate((np.ones((len(x_d2),1)), x_d2), axis = 1)

yd_d2s = np.ones((len(x_d2),1))
yd_d2s[0:25,0] = -1

w_d2s = np.array(([-0.5], [1], [1]))

w_res_d2s, y_res_d2s, Error_value_d2s = fct.apprentissage_simple(x_d2sp,w_d2s,yd_d2s, d2, n_iter_simple_d2)

x_ds_d2s, y_ds_d2s = fct.calcul_pente_separatrice(x_d2sp,w_res_d2s)

fct.affichage_resultat1(np.transpose(x_d2sp),x_ds_d2s,y_ds_d2s, yd2_show, Error_value_d2s, 'Résultat apprentissage simple d2')

"""Apprentissage widrow sur d1"""

x_d2wp = np.concatenate((np.ones((len(x_d2),1)), x_d2), axis = 1)

yd_d2w = np.ones((len(x_d2),1))
yd_d2w[0:25,0] = -1

w_d2w = np.array(([-0.5], [1], [1]))
n_iter_widrow_d2 = 15

w_res_d2w, y_res_d2w, r_widrow, Error_value_d2w = fct.apprentissage_widrow(x_d2wp,w_d2w,yd_d2w, n_iter_widrow_d2, d2)

x_ds_d2w, y_ds_d2w = fct.calcul_pente_separatrice(x_d2wp,w_res_d2w)

fct.affichage_resultat1(np.transpose(x_d2wp),x_ds_d2w,y_ds_d2w, yd2_show, Error_value_d2w, 'Résultat apprentissage widrow d2')

""" 3 Perceptron multicouches """


x_mp = np.zeros((3,1))

x_mp[0,0] = 1
x_mp[1,0] = 1
x_mp[2,0] = 1

x_XOR = np.zeros((3,4))

x_XOR[0,:] = [1,1,1,1]
x_XOR[1,:] = [0,1,0,1]
x_XOR[2,:] = [0,0,1,1]

w1_mp = np.array([(-0.5, 0.5), (2, 1), (-1, 0.5)])

w2_mp = np.array([(2), (-1), (1)])

y_mp = fct.multiperceptron(x_mp, w1_mp, w2_mp, 2)


y_XOR = np.array([(0), (1), (1), (0)])

n_iter_XOR = 20

#w1_mp_new, w2_mp2_new, y_res_XOR = fct.multiperceptron_widrow(x_XOR, w1_mp, w2_mp, y_XOR, n_iter_XOR, 'Resultat multiperceptron table XOR')

#length_x = np.shape(x_XOR)[1]
#alpha = 0.5
#wp1 = np.copy(w1_mp)
#wp2 = np.copy(w2_mp)
#Err_OL = np.ones((length_x,1))
#Err_OL_copie = np.copy(Err_OL)
#yOL = np.zeros((length_x,1))
#yHL = np.zeros((3,length_x))
#uHL = np.zeros((2,length_x))
#a = 1
#    
#while((sum(Err_OL_copie[:,0]) != 0) and (a < n_iter_XOR)):
#        
#    for i in range(length_x):
#            
#        yOL[i,0], B, C = fct.multiperceptron1(x_XOR[:,i], wp1, wp2, 2)
#        
#        Err_OL[i,0] = y_XOR[i] - yOL[i,0]
#        Err_OL_copie[i,0] = abs(Err_OL[i,0])
#        
#        for j in range(len(wp2)):
#            wp2[j] = wp2[j] + alpha*Err_OL[i,0]*B[j,0]
#        
#        b = 0
#        Err_HL = 0
#        for wk in np.transpose(wp1):
#            Err_HL = ((np.exp(-C[b,0]))/(1+np.exp(-C[b,0]))**2)*wp2[b+1]*Err_OL[i,0]
#                
#            for l in range(len(wp1[:,b])):
#                wk[l] = wk[l] + alpha*Err_HL*x_XOR[l,i]
#                
#            b = b + 1
#            
#            
#        a = a + 1







plt.figure()
plt.plot(Error_value_d1s)
plt.show()
