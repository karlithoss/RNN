# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:19:36 2018

@author: FMA
"""

import numpy as np
import scipy.io 
import h5py
import matplotlib.pyplot as plt

def readMatFile(fileName):
    
    file = scipy.io.loadmat(fileName)
    data = file.get('x')
    return data

def phi(u,active):
    
     if (active == 0):
         result = (np.sign(u))*(active == 0)
     elif (active == 1):
         result = (np.tanh(u))*(active == 1)
     elif (active == 2):
         result = (1/(1+np.exp(-u)))*(active == 2)
     
     return result

def perceptron_simple(x,w,active):
    
    u = 0
    for i in range(len(x)):
        
        u = u + x[i]*w[i]
    
    y = phi(u, active)
    
    return y

def apprentissage_simple(x,w,yd):
    
    alpha = 0.1
    wp = np.copy(w)
    r = np.ones((len(x),1))
    r_copie = np.copy(r)
    y = np.zeros((len(x),1))
    a = 1
    
    while((sum(r_copie[:,0]) != 0) and (a < 101)):
        
        for i in range(len(x)):
        
            y[i,0] = perceptron_simple(x[i,:],wp,0)
            r[i,0] = yd[i,0]-y[i,0]
            r_copie[i,0] = abs(r[i,0])
            for j in range(len(w)):
                wp[j,0] = wp[j,0] + alpha*r[i,0]*x[i,j] 
            
        
        titre = 'Résultat apprentissage simple itération =' + str(a)
        x_ds, y_ds = calcul_pente_separatrice(x,wp)
        affichage_apprentissage(x,x_ds,y_ds,titre)
        a = a + 1

    return wp, y


def calcul_pente_separatrice(x,w):
    
    b = -w[0]/w[2]
    a = -w[1]/w[2]
    
    x_droite = np.zeros((1,2))
    
    x_droite[0,0] = min(x[:,1])-1
    x_droite[0,1] = max(x[:,1])+1

    y_droite = np.zeros((1,2))
    
    for i in range(len(x_droite)+1):
        
        y_droite[0,i] = (a*x_droite[0,i] + b)
        
    return x_droite, y_droite

def affichage_apprentissage(x,x_droite,y_droite,titre):
    
    fig = plt.figure()
    plt.xlabel('Dimension 1 (x1)')
    plt.ylabel('Dimension 2 (x2)')
    plt.title(titre)
    plt.scatter(x[:,1], x[:,2])
#    plt.axis([-1.5,1.5,-1.5,1.5])
    plt.grid()
    plt.show()
    
    plt.hold(True)
    plt.plot(x_droite[0,:],y_droite[0,:], 'r')
    plt.axis([np.amin(x[:,1])-0.5, np.amax(x[:,1])+0.5, np.amin(x[:,2])-0.5, np.amax(x[:,2])+0.5])
    plt.show()
    plt.pause(2)
    plt.close(fig)
    
def affichage_resultat(x, x_droite, y_droite, titre):
    
    plt.figure()
    plt.xlabel('Dimension 1 (x1)')
    plt.ylabel('Dimension 2 (x2)')
    plt.title(titre)
    plt.scatter(x[:,1], x[:,2])
#    plt.axis([-1.5,1.5,-1.5,1.5])
    plt.grid()
    plt.show()
    
    plt.hold(True)
    plt.plot(x_droite[0,:],y_droite[0,:], 'g')
    plt.axis([np.amin(x[:,1])-0.5, np.amax(x[:,1])+0.5, np.amin(x[:,2])-0.5, np.amax(x[:,2])+0.5])
    plt.show()
    

    