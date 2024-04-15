'''
Title: Performance of Helicopter
Description: This code aims to develop the equations of perfomrance of a helicopter
using the estimated parameters
Authors: Matteus and Nathan
'''
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp

# --- Helicopter Parameters
# lb2kg = 0.4539
Wemp = 1051 # [kg]
Wfuel = 405 #[kg]
W = Wemp + Wfuel # [kg]
blade_r = 5.345 # [m]
omega = 213 / blade_r # [rad/s]
Cdp = 0.0053

"""Tail rotor values"""
Rtr = 1.86/2
chord_tr = 0.185
sigma_tr = 0.127
n_blades_tr = 2
omega_tr = 199 / Rtr
ktr = 1.4
ltr = blade_r  + Rtr + 0.2 # approx the distance between centres of main rotor to tail rotor
Adisc_tr = np.pi * Rtr ** 2

def mu_(V, omega, R):
    return V / (omega*R)


def Ttr_(Phov, omega_tr, ltr):
    return Phov / (omega_tr * ltr)

"""
NOTES TO SELF:
    - 1. Assume that the angle of attack of the tail rotor disk is 0.
    - 1. Make a function for basically doing all of this again with tail rotor. Make assumptions where necessary
    - Remember on assignment it gives info for how to get induced tail power. But get all power types and plot total TR POWER:
    - try to make the graph work correctly for the max speed that we are using. it currently looks good for up to 50 m/s.
    
"""

def Ptail(ktr, Ttr, vitr, Rtr, sigma_tr, Cdp, omega_tr, mu_tr):
    return 1.1 * ktr * Ttr * vitr + (sigma_tr*Cdp)/8*rho*(omega_tr*Rtr)**3*np.pi*Rtr**2*(1+4.65*mu_tr**2)
    #return np.sqrt(Ttr / (2*rho*np.pi*Rtr**2))

# TODO:
#   Análise no xfoil ou xflr5 do aerofólio da helice em duas seções
#   E fazer a média do Cd
n_blades = 3
blade_chord = 0.3 # [m]
# --- Induced velocity
g = 9.81
T = W*g   # [N]
rho = 1.225 # [kg/m^3]
Adisc = np.pi * blade_r ** 2 # [m^2]
vh = np.linspace(0,60,200) # [m/s] # 80 m/s is the max speed.

vi_hov = (T/(2*rho*Adisc)) ** 0.5
vi_for = (-(vh ** 2 ) / 2 + (((vh ** 2) / 2) ** 2 + vi_hov**4) ** 0.5) ** 0.5

# --- Ideal power
Pi = W * vi_hov

# --- Hover power
# -- ACT
FM = 0.65
Pact_hover = Pi / FM

# -- BEMs
# - Induced power
k = 1.1
P_ind = k * T * vi_hov

# - Profile drag
# Rotor solidity
sigma = n_blades * blade_chord / (np.pi * blade_r)

# Profile drag power

P_p = (sigma * Cdp / 8) * rho * ((omega * blade_r) ** 3) * np.pi * blade_r **2

# Power hover
Pbem_hover = P_ind + P_p


# --- Total Power in Forward Flight

k_for = 2 
eq_flt_plt_area = 2 #m^2 #This is an estimate .
#ChatGPT Calcuates this to be 2.5 at 35m/s. We should make this accurate because it actually makes
#a really large effect on the Parasite power

# Advance ratio
meu = vh / (omega * blade_r)
# Profile Power (Mixure of Pp and Pd)
Ppro = P_p * (1 + 4.65 * meu ** 2)
#Ppro = Ppro / 2 # this is illegal, but it gives better results
# Induced Power
Pi = k * T * vi_for
# Parasite Fuselage Power
Ppar = eq_flt_plt_area * 0.5 * rho * vh**3
# Total Forward Power
P_for = Ppro + Pi + Ppar

# -- Tail Rotor
ktr = 1.3
#Pitr = 1.1 * ktr * Ttr * vitr #H ow to know Ttr and vitr???
Pmisc = 0 # Miscellaneous power. Not sure?

Ttr = Ttr_(P_for, omega, ltr)

vi_tr = (Ttr/(2*rho*Adisc_tr)) ** 0.5
mu_tr = mu_(vh, omega_tr, Rtr)

Ptr = Ptail(ktr, Ttr, vi_tr, Rtr, sigma_tr, Cdp, omega_tr, mu_tr)

percent = Ptr / P_for
#print(percent)

P_for += Ptr


min_y_index = np.argmin(P_for)
v_min_power = vh[min_y_index]
print(f"Speed for best endurance = {v_min_power}")

# Calculate tangent point on curve
# Calculate gradient between consecutive points
gradients = np.diff(P_for) / np.diff(vh)
tolerance = 300  # Adjust as needed
# Check if any slope corresponds to the line passing through the origin
for i, gradient in enumerate(gradients):
    # Check if the absolute value of the slope is within the tolerance level
    c = P_for[i] - ( gradient * vh[i])
    if -tolerance < c < tolerance:
        #print(c)
        print("Speed for best range:", (vh[i])) #change this code to just take the min abs(c)

def plot_vi():
    f1 = plt.figure()
    plt.xlabel('Forward velocity [m/s]')
    plt.ylabel('Induced velocity [m/s]')
    # plt.title("")
    plt.yticks(np.arange(0,45+0.1, 5))
    plt.ylim(0,10)
    plt.grid(which="major", linewidth=1)
    plt.grid(which="minor", linewidth=0.2)
    plt.plot(vh, vi_for, label="Ex")
    # plt.legend(shadow=True, loc="upper center", fontsize=15)
    plt.show()
plot_vi()

def plot_P():
    f2 = plt.figure()
    plt.xlabel('Forward velocity [m/s]')
    plt.ylabel('Power [kW]')
    plt.plot(vh, P_for/1000, label='Total') #/1000 to get into kW
    plt.plot(vh, Ppro/1000, label='Profile') #looks like profile is too high. maybe our vi_hov is too high.
    plt.plot(vh, Pi/1000, label='Induced')
    plt.plot(vh, Ppar/1000, label='Parasite')
    plt.plot(vh, Ptr/1000, label='Tail Rotor Power')
    plt.legend()
    plt.show()
plot_P()