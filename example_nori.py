#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:18:07 2020

@author: stefano.radice

simulation of a force on a spherical particel for a mie scattering with an evanescent wave
"""

import SEW_experiment
from math import pi
import numpy as np
from numpy import sqrt,linspace,zeros
import matplotlib.pyplot as plt
import time
import cmath

ini = time.time();

lun = 100
raggio = linspace(1e-8,3*5e-7,lun);#linspace(0.1,1,lun)#
#raggio = 0.05*(1.06*1e-6)/(2*pi)
#print(raggio)
theta_obj = pi/2.2

lamb = 1.06*1e-6;
f_x = zeros(lun);
f_y = zeros(lun);
f_z = zeros(lun);
ka  = zeros(lun);
power = zeros(lun)

eps0 = 8.854187813*1e-12
mu0 = 4*pi*1e-7

convers = 1e-5#sqrt(4*pi*eps0)
n_part = cmath.sqrt(-12.2+3j)

Ep = 0#convers*
Es = 1

dati = open('dati.txt', 'w')

for i in range(lun):
    
    #print(i)
    #print('----------------------------------------------------')
    #print('----------------------------------------------------')
    #P_0 = eps0*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)#(1/(4*pi))*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)
    #P_0 = eps0*(10e-8)**2*(abs(10)**2)
    P_0= (1/(4*pi))*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)
    #print(abs(Ep)**2)
    #convers = 1#/3*1e-4    
    
    sim = SEW_experiment.SEW_experiment(1.75**2, 1, 1, 1, Ep, Es, raggio[i], lamb, (51*(pi/180)), 2*raggio[i], 1.5);
    #sim = SEW_experiment.SEW_experiment(1.5**2, 1, 1, 1, 0, 10, 10e-8, 632.8*1e-9, (42*(pi/180)), 2*10e-8, 1j);
    #sim = SEW_experiment.SEW_experiment(1.75**2, 1, 1, 1, Ep, Es, raggio, lamb, (51*(pi/180)), 2*raggio, 1.5);
    
    F_x, F_y, F_z = sim.IntegrateOnSphere();
    p, perr = sim.scatteredPower(theta_obj)
    
    power[i] = p*sqrt(eps0/mu0)*(1.2*1e5)**2

    f_x[i] = F_x[0]/P_0;
    f_y[i] = F_y[0]/P_0;
    f_z[i] = F_z[0]/P_0; 

    #fz_app = sim.Fz_dipole()

    ka[i] = raggio[i] * ((2*pi) / lamb);
    
    dati.write(str(ka[i]))
    dati.write(' ')
    dati.write(str(f_x[i]))
    dati.write(' ')
    dati.write(str(f_y[i]))
    dati.write(' ')
    dati.write(str(f_z[i]))
    dati.write(' ')
    
    #P_0=(1/(4*pi))*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)
    #P_0 = (1/(4*pi))*(10e-8)**2*(abs(10)**2)
    #dati.write(str(fz_app/P_0))      
    dati.write('\n')
    progress = int((i/lun)*100)
    print("\r [{0}{1}] {2}%".format("*"*int(round(progress/10))," "*int(10-round(progress/10)),progress), end="")

    
dati.close()

fin = time.time();

t = fin - ini;

print ("tempo esecuzione(s):",t)

plt.plot(ka,f_x,'r',ka,f_y,'b',ka,f_z,'g');

plt.figure()
plt.plot(ka,power)

plt.show();

x = 0
y = 0
theta = linspace(0,pi,50)
z_plot = []
ez_plot = []
'''
for k in range (50):
    x,y,z = SEW_experiment.fromPolToCart(theta[k], 0, 1e-6)
    x_,y_,z_ = sim.complexAngleRotation(x,y,z, np.conj(sim.gamma))
    t,p,r = SEW_experiment.fromCartToPol(x_,y_,z_)
    et,ep,er = sim.IncidentField(t,p,r)
    ex,ey,ez = SEW_experiment.fromPolToCartField(et,ep,er, t,p)
    ex_,ey_,ez_ = sim.complexAngleRotation(x,y,z, -np.conj(sim.gamma))
    z_plot.append(z)
    ez_plot.append(abs(ez_)**2)
    
plt.plot(z_plot,ez_plot)'''