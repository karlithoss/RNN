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
         result = (np.sign(u)) 
     elif (active == 1):
         result = (np.tanh(u))
     elif (active == 2):
         result = (1/(1+np.exp(-u)))
     
     return result

def perceptron_simple(x,w,active):
    
    u = 0
    for i in range(len(x)):
        
        u = u + x[i]*w[i]
    
    y = phi(u, active)
    
    return y

def perceptron_simple1(x,w,active):
    
    u = 0
    for i in range(len(x)):
        
        u = u + x[i]*w[i]
    
    y = phi(u, active)
    
    return y, u

def apprentissage_simple(x,w,yd, name_data, n_iter):
    
    alpha = 0.1
    wp = np.copy(w)
    r = np.ones((len(x),1))
    r_copie = np.copy(r)
    y = np.zeros((len(x),1))
    a = 1
    Error_value = []
    while((sum(r_copie[:,0]) != 0) and (a < n_iter)):
        
        for i in range(len(x)):
        
            y[i,0] = perceptron_simple(x[i,:],wp,0)
            r[i,0] = yd[i,0]-y[i,0]
            r_copie[i,0] = abs(r[i,0])
            for j in range(len(w)):
                wp[j,0] = wp[j,0] + alpha*r[i,0]*x[i,j] 
            
        
        titre = 'Résultat apprentissage simple ' + name_data + ' itération = ' + str(a) + '\n' + 'Error value = ' + str(sum(r_copie[:,0]))
        x_ds, y_ds = calcul_pente_separatrice(x,wp)
        affichage_apprentissage(x,x_ds,y_ds,titre)
        
        Error_value.append(sum(r_copie[:,0]))
        a = a + 1

    return wp, y, Error_value

def apprentissage_widrow(x,w,yd, n_iter, name_data):
    
    alpha = 0.1
    wp = np.copy(w)
    r = np.ones((len(x),1))
    r_copie = np.copy(r)
    y = np.zeros((len(x),1))
    a = 1
    Error_value = []
    while((sum(r_copie[:,0]) != 0) and (a < n_iter)):
        
        for i in range(len(x)):
        
            y[i,0] = perceptron_simple(x[i,:],wp,1)
            r[i,0] = yd[i,0]-y[i,0]
            r_copie[i,0] = abs(r[i,0])
            for j in range(len(w)):
                wp[j,0] = wp[j,0] + alpha*r[i,0]*x[i,j] 
            
        
        titre = 'Résultat apprentissage widrow ' + name_data + ' itération = ' + str(a) + '\n' + 'Error value = ' + str(sum(r_copie[:,0]))
        x_ds, y_ds = calcul_pente_separatrice(x,wp)
        affichage_apprentissage(x,x_ds,y_ds,titre)
        
        Error_value.append(sum(r_copie[:,0]))
        a = a + 1

    return wp, y, r_copie, Error_value


def multiperceptron(x, w1, w2, active):
    
    yHL = np.zeros((3,1))
    
    yHL[0,0] = 1
    yHL[1,0] = perceptron_simple(x, w1[:,0], active)
    yHL[2,0] = perceptron_simple(x, w1[:,1], active)
    
    yOL = perceptron_simple(yHL, w2, active)
    
    return yOL

def multiperceptron1(x, w1, w2, active):
    
    yHL = np.zeros((3,1))
    uHL = np.zeros((2,1))
    
    uOL = 0
    
    yHL[0,0] = 1
    yHL[1,0], uHL[0,0] = perceptron_simple1(x, w1[:,0], active)
    yHL[2,0], uHL[1,0] = perceptron_simple1(x, w1[:,1], active)
    
    yOL, uOL = perceptron_simple1(yHL, w2, active)
    
    return yOL, yHL, uHL

def multiperceptron_widrow(x, w1, w2, yd, n_iter, name_data):
    
    length_x = np.shape(x)[1]
    alpha = 0.5
    wp1 = np.copy(w1)
    wp2 = np.copy(w2)
    Err_OL = np.ones((length_x,1))
    Err_OL_copie = np.copy(Err_OL)
    yOL = np.zeros((length_x,1))
    a = 1
    
    while((sum(Err_OL_copie[:,0]) != 0) and (a < n_iter_XOR)):
            
        for i in range(length_x):
                
            yOL[i,0], yHL, uHL = fct.multiperceptron1(x_XOR[:,i], wp1, wp2, 2)
            
            Err_OL[i,0] = yOL[i,0] - yd[i]
            Err_OL_copie[i,0] = abs(Err_OL[i,0])
            
            b = 0
            Err_HL = 0
            for wk in np.transpose(wp1):
                Err_HL = ((np.exp(-uHL[b,0]))/(1+np.exp(-uHL[b,0])))*wp1[b+1,0]*Err_OL[i,0]
                    
                for l in range(len(wp1[:,b])):
                    wp1[l,b] = wp1[l,b] + alpha*Err_HL*x_XOR[l,i]
                
                b = b + 1
            
            for j in range(len(wp2)):
                wp2[j] = wp2[j] + alpha*Err_OL[i,0]*yHL[j,0]
            
            a = a + 1

        return wp1, wp2, yOL

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
    plt.pause(0.7)
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
    
def affichage_resultat1(x, x_droite, y_droite, clas, Error_value, titre):
    
    coul = ['mo','go','ro','yo','ko', 'bo','co']
    list_indice = []
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel('Dimension 1 (x1)')
    plt.ylabel('Dimension 2 (x2)')
    plt.title(titre + ' Classification')
    
    for i in range(int(max(clas))+1):
        for j in range(len(clas)):
        
            if (clas[j] == i):
                list_indice.append(j)
            
        list_indice = tuple(list_indice)
        plt.plot(x[1,list_indice],x[2,list_indice],coul[i])
        plt.show()
        plt.grid()
        plt.hold(True)
        list_indice = [] 

    
    plt.hold(True)
    plt.plot(x_droite[0,:],y_droite[0,:], 'g')
    plt.axis([np.amin(x[1,:])-0.5, np.amax(x[1,:])+0.5, np.amin(x[2,:])-0.5, np.amax(x[2,:])+0.5])
    plt.show()
    
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Iteration')
    plt.ylabel('Error_value')
    plt.title(titre + ' Evolution Error value')
    plt.plot(Error_value)
    
   
def affiche_class(x, clas, titre):
    
    coul = ['mo','go','ro','yo','ko', 'bo','co']

    list_indice = []
    
    plt.figure()
    plt.xlabel('Dimension 1 (x)')
    plt.ylabel('Dimension (2)')
    plt.title('Data classification '+titre)
    
    
    for i in range(int(max(clas))+1):
        for j in range(len(clas)):
        
            if (clas[j] == i):
                list_indice.append(j)
            
        list_indice = tuple(list_indice)
        #plt.scatter(x[0,list_indice],x[1,list_indice],coul[i])
        plt.plot(x[0,list_indice],x[1,list_indice],coul[i])
        plt.show()
        plt.hold(True)
        list_indice = [] 
    
    