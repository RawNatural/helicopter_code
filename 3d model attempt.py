#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 18:25:59 2024

@author: nathanrawiri
"""

import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt


"""Set initial parameters for u, w, q, θ_f"""
u = 40
w = 500
q = 1
theta_f = 5

"""Initialise general helicopter parameters"""
g = 9.81
R = 5.345
omega = 213/R 
S = math.pi*R**2
m = 1015 + 405 
W = m*g
n_blades = 3
blade_chord = 0.3
sigma = n_blades * blade_chord / (np.pi * R)
rho = 1.225

"""Guessing these values until figured them out"""
I_y = 6000 #How to calculate I_y?
Cla = 5.7
h = 1.5 #not sure what this value is?
C_D = 0.02


"""Make functions for each re-calculable variable"""
def lambda_c_(V, omega, R, a_c):
    return (V/(omega*R))*np.sin(a_c)

def meu_(V, omega, R, a_c):
    return (V/(omega*R))*np.cos(a_c)

def a_c_(w, u, theta_c):
    a__c = -np.arctan(w/u)+theta_c
    if u < 0:
        a__c += math.pi
    return a__c

def V_(w, u):
    return math.sqrt(w**2 + u**2);

def a_1_(lambda_i, lambda_c, meu, gamma, q, theta_0, omega):
    return (((-16*q/(gamma*omega))+ 8/3*meu*theta_0 - 2*meu*(lambda_c+lambda_i)) /(1-((meu**2)/2)))

def CTelem_(Cla, sigma, theta_0, lambda_i, lambda_c, meu):
    return Cla * sigma/4 * (2/3*theta_0*(1+3/2*meu**2)-(lambda_c+lambda_i))

def CTglau_(lambda_i, a_c, a_1, V, omega, R):
    return 2*lambda_i*math.sqrt((V/(omega*R)*np.cos(a_c-a_1))**2+(V/(omega*R)*np.sin(a_c-a_1)+lambda_i)**2)
def CTglau_(lambda_i, D, W, V, omega, R):
    return 2*lambda_i*math.sqrt((V/(omega*R)*np.cos(D/W))**2+(V/(omega*R)*np.sin(D/W)+lambda_i)**2)   


def T_(C_T, rho, omega, R):
    return C_T*rho*(omega*R)**2*math.pi*R**2

def D_(C_D, rho, V, S):
    return C_D*0.5*rho*V**2*S

"""Create derivative functions"""
def du_(g, theta_f, D, u, V, m, T, theta_c, a_1, w, q):
    return -g * np.sin(theta_f) - D/m*u/V + T/m*np.sin(theta_c-a_1) - q*w
    
def dw_(g, theta_f, D, u, V, m, T, theta_c, a_1, w, q):
    return g * np.cos(theta_f) - D/m*w/V - T/m*np.cos(theta_c-a_1) + q*u
    
def dq_(T, I_y, h, theta_c, a_1):
    return -T/I_y*h*np.sin(theta_c-a_1)

def dtheta_f_(q):
    return q

def F_(CTglau, CTelem):
    """Would like F to only be varied by lambda_i so add CT parameters in in func or elsewhere"""
    return CTelem - CTglau


'''
def glauert(vih, Vh, W, CDfus, Sfus, error):
    """
    :param vih:
    :param Vh:
    :param W:
    :param CDfus:
    :param Sfus:
    :param error:
    :return:
    """
    Vbar = Vh / vih
    D = 0.5 * rho * (Vh**2) * Sfus * CDfus
    alphad = np.arcsin(D/W)
    V1 = Vbar * np.cos(alphad)
    V2 = Vbar * np.sin(alphad)
    vibar0 = 1
    ite = 100
    for i in range(ite):
        print(i)
        vibar = np.sqrt(1/(V1 *2 + (V2 + vibar0)*2 ))
        if np.abs((vibar - vibar0) / vibar) <= error:
            return vibar
        vibar0 = vibar
    return vibar'''

""" Choose angles """
theta_c = 0; theta_0 = 0 #degrees

ws = []
ts = []

t = 0;




    
while t < 5:    
    V = V_(w, u);
    a_c = a_c_(w, u, theta_c)
    meu = meu_(V, omega, R, a_c)
    lambda_c = lambda_c_(V, omega, R, a_c)
    lambda_i = 1
    F = 0.2
    while not -0.1 < F < 0.1:
        print(f'F = {F}')
        if F > 0:
            lambda_i += 0.000001
        else:
            lambda_i -= 0.000001
        gamma = lambda_i / V
        a_1 = a_1_(lambda_i, lambda_c, meu, gamma, q, theta_0, omega)
        CTelem = CTelem_(Cla, sigma, theta_0, lambda_i, lambda_c, meu)
        CTglau = CTglau_(lambda_i, a_c, a_1, V, omega, R)
        F = F_(CTglau, CTelem)
    CT = CTelem
    print(f"Final CT = {CT}")
    print(f"Final lambda_i = {lambda_i}")
    
    D = D_(C_D, rho, V, S)
    T = T_(CT, rho, omega, R)
    q += dq_(T, I_y, h, theta_c, a_1)
    theta_f += dtheta_f_(q)
    u += du_(g, theta_f, D, u, V, m, T, theta_c, a_1, w, q)

    w += dw_(g, theta_f, D, u, V, m, T, theta_c, a_1, w, q)
    print(f"u = {u}")
    print(f"w = {w}")
    print(f"q = {q}")
    print(f"θ_f = {theta_f}")
    ws.append(w)
    ts.append(t)
    
    t += 1


plt.figure()
plt.xlabel('t')
plt.ylabel('w')
plt.scatter(ts, ws)
plt.scatter(ts, ws)
plt.show()




"""Shitty first attempt...
The assumptions i made:
    - Vh instead of u or w or V??? Don't even know what to do about this?? Ur dumb
    - lambda_i? How to get? What is Vh or Vbar or Vibar in matteus code?
    - I_y. Is I_y even needed?
    - How come you are trying to use theta_0 before you find theta_0? Like this is just doomed to fail
    - Where TF are those initial parameters coming from?
    - 



"""
















'''
"""While loop"""

while abs(du) > 0.01 and abs(du) > 0.01 and abs(dq) > 0.01 and abs(dtheta_f) > 0.01:
    """Calculate 1st 4 things"""
    V = V_(w, u);
    a_c = a_c_(w, u, theta_c)
    meu = meu_(V, omega, R, a_c)
    lambda_c = lambda_c_(V, omega, R, a_c)
    """Create while loop for getting Vind, a1, CT"""
    """Choose lambda_i"""
    lambda_i = 0.789
    F = 5
    while not -0.001 < F < 0.001:
        if F > 0:
            lambda_i += 0.001
        else:
            lambda_i -= 0.001
        gamma = lambda_i / V
        a_1 = a_1_(lambda_i, lambda_c, meu, gamma, q, theta_0, omega)
        CTelem = CTelem_(Cla, sigma, theta_0, lambda_i, lambda_c, meu)
        CTglau = CTglau_(lambda_i, a_c, a_1, V, omega, R)
        F = F_(CTglau, CTelem)
        #print(F)
    CT = CTelem
    print(f"Final lambda_i = {lambda_i}")
    
    
    
    """Continue"""
    T = T_(CT, rho, omega, R)
    D = D_(C_D, rho, V, S)
    """create time program in which it will vary the 4 parameters of motion over time"""

    du = du_(g, theta_f, D, u, V, m, T, theta_c, a_1, w, q)
    dw = dw_(g, theta_f, D, u, V, m, T, theta_c, a_1, w, q)
    dq = dq_(T, I_y, h, theta_c, a_1)
    dtheta_f = dtheta_f_(q)
    u += du
    w += dw
    q += dq
    theta_f += dtheta_f
    print(f"u = {u}")
    print(f"w = {w}")
    print(f"q = {q}")
    print(f"theta = {theta_f}")
    
print("any ended?")
'''

"""See what happens when you change V, and change collective and cyclic pitches"""

"""Try to get a relation between V and pitches"""