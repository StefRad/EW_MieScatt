from cmath import sin, cos, asin, acos, sinh, cosh, exp
import numpy as np
from math import pi,factorial
from scipy.special import clpmn as Legendre
from scipy.special import spherical_jn as Bessel_jn
from scipy.special import spherical_yn as Bessel_yn
from scipy.special import riccati_jn,jv,iv
from scipy import integrate
import scipy

class MieScattForce:

    def __init__(self, n1, n2, n3, theta1, E0p, E0s, a, dist, lamb):
        
        # system 1:fiber, 2:medium, 3:particle
        self.n1 = n1;
        self.n2 = n2;
        self.n3 = n3;
        
        # incident angle 1-2
        self.theta1 = theta1;
        
        #incident field in 1
        self.E0p = E0p;
        self.E0s = E0s;
        
        self.H0p = np.sqrt((8.8542*1e-12)/(4*pi*1e-7))*E0p;
        self.H0s = np.sqrt((8.8542*1e-12)/(4*pi*1e-7))*E0s;
        
        #particle radius
        self.a = a;
        self.b = a+1e-6;############################
        
        # distance from surface
        self.h = dist;
        
        # laser wavelenght
        self.lamb = lamb;
        
        self.n21 = n2 / n1;
        
        self.alpha = (n2*2*pi*a) / (lamb);
        self.bet   = (n2*2*pi*self.b) / (lamb);
        self.beta  = ((n1*2*pi)/(lamb))*np.sqrt(sin(theta1)**2 - self.n21**2)
        self.gamma = (n1*2*pi/lamb)*sin(theta1);
        
        # transmission coefficient
        self.Tp = (2*self.n21*cos(theta1)) / (self.n21**2 * cos(theta1) + 1j*np.sqrt(sin(theta1)**2 - self.n21**2));
        self.Ts = (2*cos(theta1)) / (cos(theta1) + 1j*np.sqrt(sin(theta1)**2 + self.n21**2));
        
        
        
    def alpha1(self,l,m):
        fatt1 = np.sqrt( ((2*l+1)*factorial(l-m)) / (4*pi*factorial(l+m)) )
        psi,psi_der = riccati_jn(l, self.bet)
        fatt2 = (self.b/self.a)**2 / (l*(l+1)*psi[l])#############
        return fatt1*fatt2
    
    def Q1(self,l,m):
        fatt = 2*pi*((-1)**(m-1))
        if (l+m)%2 == 0:
            def integrand(theta):
                u = self.beta*self.b*sin(theta)
                res = sin(theta)**2 * cos(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m-1),u)+Imz(abs(m+1),u))
                return res
        else:
            def integrand(theta):
                u = self.beta*self.b*sin(theta)
                res = sin(theta)**2 * (1j) * sin(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m-1),u)+Imz(abs(m+1),u))
                return res
            
        def real_part(theta):
            return np.real(integrand(theta))
        def imag_part(theta):
            return np.imag(integrand(theta))
        
        I_real = integrate.quad(real_part,0,pi/2)
        I_imag = integrate.quad(imag_part,0,pi/2)
        
        return fatt*(I_real[0] + 1j*I_imag[0])
    
    def Q2(self,l,m):
        fatt = 4*pi*((-1)**(m))
        if (l+m)%2 == 0:
            def integrand(theta):
                u = self.beta*self.b*sin(theta)
                res = sin(theta) * cos(theta) * (1j) * sin(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                return res
        else:
            def integrand(theta):
                u = self.beta*self.b*sin(theta)
                res = sin(theta) * cos(theta) * cos(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                return res
        def real_part(theta):
            return np.real(integrand(theta))
        def imag_part(theta):
            return np.imag(integrand(theta))
        
        I_real = integrate.quad(real_part,0,pi/2)
        I_imag = integrate.quad(imag_part,0,pi/2)
        
        return fatt*(I_real[0] + 1j*I_imag[0])
    
    def Q3(self,l,m):
        fatt = 4*pi*1j*((-1)**(m))*(m / (self.beta*self.b)) ######################
        if (l+m)%2 == 0:
            def integrand(theta):
                u = self.beta*self.b*sin(theta)
                res = sin(theta) * cos(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                return res
        else:
            def integrand(theta):
                u = self.beta*self.b*sin(theta)
                res = sin(theta) * (1j) * sin(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                return res
        
        def real_part(theta):
            return np.real(integrand(theta))
        def imag_part(theta):
            return np.imag(integrand(theta))
        
        I_real = integrate.quad(real_part,0,pi/2)
        I_imag = integrate.quad(imag_part,0,pi/2)
        
        return fatt*(I_real[0] + 1j*I_imag[0])
        
        
    def Mie_AB(self,l):
        
        Ap = []
        Bp = []
        As = []
        Bs = []
        
        for i in range(l):
            Ap1 = []
            Bp1 = []
            As1 = []
            Bs1 = []
            k = i+1
            for j in range(2*k+1):
                m = j-k;
                Ap1.append((self.alpha1(k,m)/self.n21) * self.Tp * exp(-self.beta*self.h) * ((sin(self.theta1)*self.Q1(k,m))-((1j)*np.sqrt(sin(self.theta1)**2 - self.n21**2)*self.Q2(k,m))))
                Bp1.append(self.n2*self.alpha1(k,m) * self.Tp * exp(-self.beta*self.h) * self.Q3(k,m))
                As1.append((self.Ts / (self.n2*self.Tp)) * self.n2*self.alpha1(k,m) * self.Tp * exp(-self.beta*self.h) * self.Q3(k,m))
                Bs1.append(-((self.n2*self.Ts)/self.Tp) * (self.alpha1(k,m)/self.n21) * self.Tp * exp(-self.beta*self.h) * (sin(self.theta1)*self.Q1(k,m)-(1j)*np.sqrt(sin(self.theta1)**2 - self.n21**2)*self.Q2(k,m)))
            Ap.append(Ap1)
            Bp.append(Bp1)
            As.append(As1)
            Bs.append(Bs1)
        
        return Ap,As,Bp,Bs
    
    
def Imz(m,z):
    arg = 1j*z
    res = ((1j)**(-m)) * jv(m,arg)
    # res = iv(m,z)
    print(res)
    return res

