#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:08:15 2024

@author: nathanrawiri
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import symbols, solve
import sympy as sp



def D_(C_D, rho, V, S):
    return C_D*0.5*rho*V**2*S

def C_T_(T, rho, omega, R):
    return T / (rho*(omega*R)**2*math.pi*R**2)
    
def T_(W, D):
    return math.sqrt(W**2 + D**2)


"""initilise variables"""
rho = 1.225 # [kg / m^3]
g = 9.81 # [m/s^2]
m_emp = 1051 # [kg]
m_fuel = 405 #[kg]
m = m_emp + m_fuel # [kg]
W = m*g # [N]
CDfus = 0.02 
Sfus = 90 
R = 5.345 # [m]
omega = 213/R # rad/s
Cla = 5.7
n_blades = 3
blade_chord = 0.3 # [m]
sigma = n_blades * blade_chord / (math.pi * R)


def getTrimAngles(Vh):
    vi = symbols('vi')
    """initialise variables"""
    meu = Vh / (omega * R)
    D = D_(CDfus, rho, Vh, Sfus)
    T = T_(W, D)
    C_T = C_T_(T, rho, omega, R)
    
    """The below 3 lines is what causes the program to run so slowly. sp solve"""
    vi_eq = (2*vi*sp.sqrt((Vh/(omega*R)*sp.cos(D/W))**2+(Vh/(omega*R)*sp.sin(D/W)+vi)**2)) - C_T
    lambda_i = solve(vi_eq, vi) #"""This line is what takes a long time"""
    vibar = lambda_i[0]
    """Above is the slow part"""
    
    """Do matrix manipulation"""
    """Take matrix equation as A(2x2).B(1x2) = C(1x2). Rearrange for B = A-1 . C"""
    A = [[1+(3/2)*meu**2,-8/3*meu],
        [-meu, 2/3 + meu**2]]
    C = [[-2*meu**2*D/W-2*meu*vibar],
          [4/sigma*C_T/Cla+meu*D/W+vibar]]
    A_inv = np.linalg.inv(A)
    B = np.dot(A_inv, C)
    
    """Get collective and cyclic angles"""
    cyclic = B[0][0]
    collect = B[1][0]
    #print(f"cyclic = {cyclic *180/np.pi}")
    #print(f"collect = {collect *180/np.pi}")
    #print(f"Pitch = {theta_f}")
    return cyclic, collect

def main():
    cyclics = []
    collects = []
    
    Vhs = np.linspace(0.1, 80, 80)
    for Vh in Vhs:
        print(Vh)
        cyclic, collect = getTrimAngles(Vh)
        cyclics.append(cyclic * 180/np.pi)
        collects.append(collect *180/np.pi)
    
    plt.figure()
    plt.xlabel('V (m/s)')
    plt.ylabel('θ')
    plt.plot(Vhs, collects, label="collective (θ0)", color="b")
    plt.plot(Vhs, cyclics, label="cyclic (θc)", color="c")
    plt.legend()
    plt.show()

main()

