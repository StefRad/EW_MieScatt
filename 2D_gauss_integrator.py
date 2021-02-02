# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:50:15 2021

@author: Davide Deca
                                       _
+                       +               | 
   +                 +     +            |
     +             +         +          |
      +           +           +         | ampiezza
       +         +             +        |
         +     +                 +      |
            +                       +  _|
"""
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

def gauss_integrator(ampiezza, sigma):
    
    def gaussiana(x,y):
        
        return 1/(2*np.pi*sigma**2)*np.exp(-((x+ampiezza/2)**2+y**2)/(2*sigma**2))

    vol_sx = integrate.nquad(gaussiana,[[-3e+7*ampiezza,0],[-3e+7*ampiezza,3e+7*ampiezza]])
    vol_dx = integrate.nquad(gaussiana,[[0,3e+7*ampiezza],[-3e+7*ampiezza,3e+7*ampiezza]])
    
    x_extr = 3e+7*ampiezza
    npunti = 101 #deve essere dispari!!!
    lato = 2*x_extr/npunti
        
    surface = np.empty((npunti,npunti))
        
    for i in range(npunti):
        for j in range(npunti):
                
            surface[i,j] = gaussiana(-x_extr+(0.5+j)*lato, -x_extr+(0.5+i)*lato)
            if j == (npunti-1)/2:
                surface[i,j] = 0.2
                #print(-x_extr+(0.5+j)*lato)
            
    plt.imshow(surface, cmap='Blues_r')        
    
    return vol_sx, vol_dx