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

lun = 100
raggio = linspace(1e-12,1e-6,lun);
lamb = 600*1e-9;
f_x = zeros(lun);
f_y = zeros(lun);
f_z = zeros(lun);
ka  = zeros(lun);

for i in range(lun):
    
    print(i)
    
    sim = SEW_experiment.SEW_experiment(sqrt(1.75), 1, sqrt(1.5), 1, 1, 1, raggio[i], lamb, (51*(pi/180)), 2*raggio[i], 1);
    
    F_x, F_y, F_z = sim.IntegrateOnSphere();

    f_x[i] = F_x[0];
    f_y[i] = F_y[0];
    f_z[i] = F_z[0]; 

    ka[i] = raggio[i] * ((2*pi) / lamb);

fin = time.time();

t = fin - ini;

print ("tempo esecuzione(s):",t)

plt.plot(ka,f_x,'r',ka,f_y,'b',ka,f_z,'g');

plt.show();