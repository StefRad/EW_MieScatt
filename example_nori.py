#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:18:07 2020

@author: stefano.radice

simulation of a force on a spherical particel for a mie scattering with an evanescent wave
"""

import SEW_experiment
from math import pi
from numpy import sqrt,linspace,zeros
import matplotlib.pyplot as plt
import time

ini = time.time();

lun = 25
raggio = linspace(2e-9,1.2*1e-6,lun);
lamb = 800*1e-9;
f_x = zeros(lun);
f_y = zeros(lun);
f_z = zeros(lun);
ka  = zeros(lun);

eps0 = 8.854187813*1e-12
convers = sqrt(4*pi*eps0)

Ep = convers*1.2*1e5
Es = 0

dati = open('dati.txt', 'w')

for i in range(lun):
    
    print(i)
    
    P_0 = eps0*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)#(1/(4*pi))*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)
    #P_0 = eps0*(10e-8)**2*(abs(10)**2)
    #P_0= (1/(4*pi))*(raggio[i])**2*(abs(Ep)**2+abs(Es)**2)
    #print(abs(Ep)**2)
    #convers = 1#/3*1e-4    
    
    sim = SEW_experiment.SEW_experiment(1.75**2, 1, 1, 1, Ep, Es, raggio[i], lamb, (51*(pi/180)), 2*raggio[i], 1.5);
    #sim = SEW_experiment.SEW_experiment(1.5**2, 1, 1, 1, 0, 10, 10e-8, 632.8*1e-9, (42*(pi/180)), 2*10e-8, 1j);
    
    F_x, F_y, F_z = sim.IntegrateOnSphere();

    f_x[i] = F_x[0]/P_0;
    f_y[i] = F_y[0]/P_0;
    f_z[i] = F_z[0]/P_0; 
    
    fz_app = sim.Fz_dipole()

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
    dati.write(str(fz_app/P_0))      
    dati.write('\n')
    
dati.close()

fin = time.time();

t = fin - ini;

print ("tempo esecuzione(s):",t)

plt.plot(ka,f_x,'r',ka,f_y,'b',ka,f_z,'g');

plt.show();