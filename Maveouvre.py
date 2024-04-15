#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:32:07 2024

@author: nathanrawiri
"""

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
#import Trim



# PERSONAL HELI DATA BELOW RESET ALL


# INITIAL DATA HELICOPTER
g = 9.81                        # (kg/m^2) Gravitational acceleration
cla = 5.7                       # (rad^-1) Profile NACA 0012 Lift Coefficient slope                    
lok = 6                         # Lock number \\\\\Assumed value
cds = 0.02                      # Drag coefficient fo the fuselage
mass = 1456                     # Helicopter mass
rho = 1.225                     # (kg/m^3) Air density
vtip = 213                      # (m/sec) Blade tip speed
diam = 10.69                    # (m) Rotor diameter
R = diam / 2                    # (m) Rotor radius
n_blades = 3; c = 0.3           # n blades and chord
volh = n_blades*c/(np.pi*R)     # blade solidity
iy = 6000                       # kg m^2) Helicopter moment of inertia in the pitch axis \\\\\Assumed value
mast = 1                        # %(m) Vertical distance between rotor CG and rotor hub \\\\\Assumed value
omega = vtip / (diam / 2)       # Rotor tip speed
area = np.pi / 4 * diam ** 2    # Rotor area
tau = .1                        # (sec) time constant in dynamics inflow!!!
collect = [8.35*np.pi/180]         # (rad)  Collective pitch angle
longit = [3.5*np.pi/180]          # (rad) Longitudinal cyclic pitch angle







# INITIAL VALUES SIMULATION
t0 = 0                                      #(sec) Setting initial time 
pitch0 = 0#-5 * np.pi / 180                    #(rad) Setting initial helicopter pitch angle
u0 = 50 #* np.cos(pitch0)                     #(m/sec) Setting initial helicopter airspeed component along body x-axis 
w0 = 0#V0 #* np.sin(pitch0)                     #(m/sec) Setting initial helicopter airspeed component along body z-axis
q0 = 0                                      #(rad/sec) Setting initial helicopter pitch rate 
x0 = 0                                      #(m) Setting initial helicopter longitudinal position 
labi0 = np.sqrt(mass*g/(area*2*rho))/vtip   # Initialization non-dimensional inflow
del_col0 = 0
h0 = 0

#-------

t = [t0]
u = [u0]
w = [w0]
q = [q0]
pitch = [pitch0]
x = [x0]
labi = [labi0]
z = [0]

del_col = [del_col0]
du = [0]


#get trim angle at a given velocity
#cyclic, collective = Trim.getTrimAngles(u0)
#print(cyclic, collective)
#collect = [collective]; longit = [cyclic];

# INTEGRATION
aantal = 801
teind = 80
stap = (teind - t0) / aantal

#-------
# SETTING V
V0 = 50
V_knot0 = 90
V_knot1 = 70
V_knot2 = 90
V_knot3 = 110
V0 = V_knot0 * 0.514444 #m/s

V = [V0]

u_des = V_knot0 * 0.514444

for i in range(aantal):
    #np.append(V, )
    ti = t[-1]
    '''if 0.5 <= ti < 1:
        longit = np.append(longit, 1 * np.pi / 180)
    else:
        longit = np.append(longit, 0 * np.pi / 180)'''
    longit = np.append(longit, longit[0])

    #================================================
    # 4. Starting the controller
    #================================================
    
    if 40 > ti >= 20:
        if u_des > V_knot1 * 0.514444:
            u_des -= 0.5* 0.514444
            #print(u_des / 0.51444)
        #u_des = V_knot1 * 0.514444
    elif 70 > ti >= 40:
        u_des = V_knot2 * 0.514444
    elif ti >= 70:
        u_des = V_knot3 * 0.514444
        
    """if ti >= 50: #change up to 90 knots
        if u_des < V_knot2:
            u_des += 0.1*V_knot2 * 0.514444
    if ti >= 70: #change up to 110 knots
        if u_des < V_knot3:
            u_des += 0.1*V_knot3 * 0.514444"""
    
    
    #pitch_des = 0.02 * (u[-1] - u0)
    pitch_des = -0.09*(u_des-u[-1]) - 0.01*w[-1] #- 0.1*du[-1] #+ 0.02 * u[-1]
    #pitch_des = -5*np.pi/180
    #if ti >= 15:
    longitgrd = 0.2 * (pitch[-1]-pitch_des) * 180 / np.pi + 0.5 * (q[-1] * 180 / np.pi)  # PD in deg
    #longitgrd = 0.2 * q[-1] * 180 / np.pi  # PD in deg
    #longitgrd = -0.05*w[-1]
    longit[-1] = longitgrd * np.pi / 180  # in rad


    # NO LAW FOR COLLECTIVE
    c = u[-1] * np.sin(pitch[-1]) - w[-1] * np.cos(pitch[-1])
    h = -z[-1]
    collect = np.append(collect, collect[0])

    
    # SET LAW FOR COLLECTIVE
    h_des = 0
    #c_des = 0
    K1 = 0.5; K2 = 0.5; K3 = 0.5
    c_des = K3 * (h_des - h)
    collectgrd = K1 * (c_des - c) + K2 * (del_col[-1])
    #c_des = 0.01*(h-des_h) + 0.002 * w[-1]
    #collectgrd = -0.7 * c + 2* w[-1]+ ((q[-1])) #- 0.02*w[-1] #0.2 * (u[-1]-u0) + 0.02*c#+ 0.2 * w[-1]
    #collectgrd = 0.2 * (c-c_des) * 180 / np.pi + 0.2 * q[-1] * 180 / np.pi 
    collect[-1] = collect[0] + (collectgrd) * np.pi / 180
    #collect[-1] = -0.05*(h-h_des) * np.pi/180 #+ 0.02 * w[-1] #+ 0.01 * (u0-u[-1])
    #print(collect[-1]*180/np.pi)
    
    
    
    #================================================
    # 5. See calculation of parameters above
    #================================================
    qdiml = q[-1] / omega
    vdiml = np.sqrt(u[-1]**2 + w[-1]**2) / vtip
    if u[-1] == 0:
        phi = np.pi / 2 if w[-1] > 0 else -np.pi / 2
    else:
        phi = np.arctan(w[-1] / u[-1])
    if u[-1] < 0:
        phi += np.pi
    alfc = longit[-1] - phi

    mu = vdiml * np.cos(alfc)
    labc = vdiml * np.sin(alfc)

    # a1 Flapping calculi
    teller = -16 / lok * qdiml + 8 / 3 * mu * collect[-1] - 2 * mu * (labc + labi[-1])
    a1 = teller / (1 - .5 * mu**2)

    # the thrust coefficient
    ctelem = cla * volh / 4 * (2 / 3 * collect[-1] * (1 + 1.5 * mu**2) - (labc + labi[-1]))
    # Thrust coefficient from Glauert
    alfd = alfc - a1
    ctglau = 2 * labi[-1] * np.sqrt((vdiml * np.cos(alfd))**2 + (vdiml * np.sin(alfd) + labi[-1])**2)

    #================================================
    # 6. See equations of motion below
    #================================================
    labidot = ctelem
    thrust = labidot * rho * vtip**2 * area
    helling = longit[-1] - a1
    vv = vdiml * vtip  # it is 1/sqrt(u^2+w^2)

    """Equations of motion"""
    udot = -g * np.sin(pitch[-1]) - cds / mass * .5 * rho * u[-1] * vv + thrust / mass * np.sin(helling) - q[-1] * w[-1]
    wdot = g * np.cos(pitch[-1]) - cds / mass * .5 * rho * w[-1] * vv - thrust / mass * np.cos(helling) + q[-1] * u[-1]
    qdot = -thrust * mast / iy * np.sin(helling)
    pitchdot = q[-1]

    xdot = u[-1] * np.cos(pitch[-1]) + w[-1] * np.sin(pitch[-1])
    zdot = -c
    labidot = (ctelem - ctglau) / tau
    
    del_coldot = c_des - c
    dudot = u_des - u[-1]

    u.append(u[-1] + stap * udot)
    w.append(w[-1] + stap * wdot)
    q.append(q[-1] + stap * qdot)
    pitch.append(pitch[-1] + stap * pitchdot)
    x.append(x[-1] + stap * xdot)
    labi.append(labi[-1] + stap * labidot)
    z.append(z[-1] + stap * zdot)
    t.append(ti + stap)
    
    #adding more:
    del_col.append(del_col[-1] + stap * del_coldot)
    du.append(du[-1] + stap * dudot)
    


plt.figure()
plt.title('Vertical Velocity over Time')
plt.ylabel('w (m/s)')
plt.xlabel('t (s)')
plt.plot(t, w, label="w", color="b") 
plt.legend()
plt.grid()
plt.show()



plt.figure()
plt.title('Displacement in X-Direction over Time')
plt.ylabel('x (m)')
plt.xlabel('t (s)')
plt.plot(t, x, label="x", color="b") 
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Rate of Pitch Change over Time')
plt.ylabel('q (deg/s)')
plt.xlabel('t (s)')
plt.plot(t, np.array(q)*180/np.pi, label="q", color="b") 
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Non-Dimensional Inflow over Time')
plt.ylabel('labi')
plt.xlabel('t (s)')
plt.plot(t, labi, label="labi", color="b") 
plt.legend()
plt.grid()
plt.show()



plt.figure()
plt.title('Velocity and Pitch over Time')
#plt.ylabel('u & pitch')
plt.xlabel('t (s)')
plt.plot(t, u, label="u (m/s)", color="b") 
plt.plot(t, np.array(pitch) * 180 / np.pi, label="pitch (deg)", color="c") 
plt.legend()
#plt.xlim(xmin=0, xmax=40)  # Adjust the limits for the x axis
#plt.ylim(ymin=-20, ymax=70)  # Adjust the limits for the y axis
plt.grid()
plt.show()

plt.figure()
plt.ylabel('long cyclic')
plt.xlabel('t')
plt.plot(t, np.array(longit) * 180 / np.pi, label="longitudinal cyclic pitch", color="b")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Collective over Time')
plt.ylabel(' theta (deg)')
plt.xlabel('t (s)')
plt.plot(t, collect * 180 / np.pi, label="theta (deg)", color="b") 
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title('Altitude over Time')
plt.ylabel(' h (m)')
plt.xlabel('t (s)')
plt.plot(t, np.array(z), label="h (m)", color="b") 
plt.legend()
plt.grid()
plt.show()

'''plt.figure()
plt.title('Displacement in X-Direction  and Pitch over Time')
plt.xlabel('t (s)')
plt.plot(t, x, label="x (m)", color="b") 
plt.plot(t, np.array(pitch) * 180 / np.pi, label="pitch (deg)", color="c") 
plt.legend()
#plt.xlim(xmin=0, xmax=40)  # Adjust the limits for the x axis
#plt.ylim(ymin=-20, ymax=70)  # Adjust the limits for the y axis
plt.grid()
plt.show()'''
'''
plt.figure()
plt.title('Velocity over Time')
plt.ylabel(' V (m/s)')
plt.xlabel('t (s)')
plt.plot(t, np.array(V), label="h (m)", color="b") 
plt.legend()
plt.grid()
plt.show()'''

plt.figure()
plt.title('The 4 mains')
plt.xlabel('t (s)')
plt.plot(t, np.array(u) / 0.514444, label="u (knot)", color="k") 
plt.plot(t, w, label="w (m/s)", color="r") 
plt.plot(t, np.array(q)*180/np.pi, label="q", color="m") 
plt.plot(t, np.array(pitch) * 180 / np.pi, label="pitch (deg)", color="c") 
plt.legend()
#plt.xlim(xmin=0, xmax=40)  # Adjust the limits for the x axis
#plt.ylim(ymin=-20, ymax=70)  # Adjust the limits for the y axis
plt.grid()
plt.show()


