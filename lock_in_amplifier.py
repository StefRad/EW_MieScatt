# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 08:24:27 2020

@author: Davide Deca
"""
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
#from cmath import sin, cos
from math import sin, cos,pi, atan2

class lock_in_amplifier:
    
    def __init__(self, power, amplitude, phase, npunti, periodo_segnale):
        
        self.npunti = npunti
        self.amplitude = amplitude
        self.phase = phase
        self.periodo_segnale = periodo_segnale
    
        noise = np.random.normal(np.sqrt(power), size=npunti)
        signal = []
        self.signal_noise = []
        
        for i in range(npunti):
            
            signal.append(np.sqrt(2)*amplitude*sin(2*pi/periodo_segnale*i+phase))
            self.signal_noise.append(np.sqrt(2)*amplitude*sin(2*pi/periodo_segnale*i+phase)+noise[i])
        
        plt.plot(self.signal_noise)
        plt.show()
        
    def mixing(self):
        
        self.signal_noise_X = []
        self.signal_noise_Y = []        
        
        for i in range(self.npunti):
        
            self.signal_noise_X.append(np.sqrt(2)*self.signal_noise[i]*cos(2*pi*i/self.periodo_segnale))
            self.signal_noise_Y.append(-np.sqrt(2)*self.signal_noise[i]*sin(2*pi*i/self.periodo_segnale))        

        self.Xf = ft.fft(self.signal_noise_X)
        self.Yf = ft.fft(self.signal_noise_Y)
        
        plt.plot(self.signal_noise_X)
        plt.show()
        
        plt.plot(self.Xf)
        plt.show()
        
    def filtering(self, tau, order):
        
        #x = 1/(2*pi)*ft.fft(self.Xf)
        
        #plt.plot(x)
        #plt.show()
        
        X = np.mean(self.signal_noise_X)
        Y = np.mean(self.signal_noise_Y) 
        
        R = np.sqrt(X**2+Y**2)
        phi = atan2(Y,X)
        
        return R, phi