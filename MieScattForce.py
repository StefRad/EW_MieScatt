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
        self.b = a-0.999999;############################
        
        # distance from surface
        self.h = dist;
        
        # laser wavelenght
        self.lamb = lamb;
        
        self.n21 = n2 / n1;
        self.n32 = n3 / n2;
        
        self.alpha = (n2*2*pi*a) / (lamb);
        self.bet   = (n2*2*pi*self.b) / (lamb);
        self.beta  = np.real(((n1*2*pi)/(lamb))*np.sqrt(sin(theta1)**2 - self.n21**2))
        self.gamma = (n1*2*pi/lamb)*sin(theta1);
        
        # transmission coefficient
        self.Tp = (2*self.n21*cos(theta1)) / (self.n21**2 * cos(theta1) + 1j*np.sqrt(sin(theta1)**2 - self.n21**2));
        self.Ts = (2*cos(theta1)) / (cos(theta1) + 1j*np.sqrt(sin(theta1)**2 - self.n21**2));
        
        print('b = ',self.b)
        print('beta = ',self.beta)
        
    def alpha1(self,l,m):
        fatt1 = np.sqrt( ((2*l+1)*factorial(l-m)) / (4*pi*factorial(l+m)) )
        psi,psi_der = riccati_jn(l, self.bet)
        fatt2 = (self.b/self.a)**2 / (l*(l+1)*psi[l])#############
        return fatt1*fatt2
    
    def Q1(self,l,m):
        fatt = 2*pi*((-1)**(m-1))
        if (l+abs(m))%2 == 0:
            if m > 0:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta)**2 * cos(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m-1),u)+Imz(abs(m+1),u))
                    return res
            else:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta)**2 * cos(self.gamma*self.b*cos(theta)) *(-1)**abs(m)* Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m-1),u)+Imz(abs(m+1),u))
                    return res    
        else:
            if m > 0:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta)**2 * (1j) * sin(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m-1),u)+Imz(abs(m+1),u))
                    return res
            else:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta)**2 * (1j) * sin(self.gamma*self.b*cos(theta)) *(-1)**abs(m)* Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m-1),u)+Imz(abs(m+1),u))
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
        if (l+abs(m))%2 == 0:
            if m > 0:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * cos(theta) * (1j) * sin(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
            else:    
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * cos(theta) * (1j) * sin(self.gamma*self.b*cos(theta)) * (-1)**abs(m)*Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
        else:
            if m > 0:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * cos(theta) * cos(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
            else:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * cos(theta) * cos(self.gamma*self.b*cos(theta)) * (-1)**abs(m)*Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
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
            if m > 0:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * cos(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
            else:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * cos(self.gamma*self.b*cos(theta)) * (-1)**abs(m)* Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
        else:
            if m > 0:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * (1j) * sin(self.gamma*self.b*cos(theta)) * Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
            else:
                def integrand(theta):
                    u = self.beta*self.b*sin(theta)
                    res = sin(theta) * (1j) * sin(self.gamma*self.b*cos(theta)) * (-1)**abs(m)* Legendre(m,l,cos(theta))[0][m,l]*(Imz(abs(m),u))
                    return res
        def real_part(theta):
            return np.real(integrand(theta))
        def imag_part(theta):
            return np.imag(integrand(theta))
        
        I_real = integrate.quad(real_part,0,pi/2)
        I_imag = integrate.quad(imag_part,0,pi/2)
        
        return fatt*(I_real[0] + 1j*I_imag[0])
        
        
    def Mie_AB(self,l):
        
        self.l = l
        
        self.Ap = []
        self.Bp = []
        self.As = []
        self.Bs = []
        
        self.a_p = []
        self.b_p = []
        self.a_s = []
        self.b_s = []
        
        for i in range(l):
            
            Ap1 = []
            Bp1 = []
            As1 = []
            Bs1 = []
            
            a_p1 = []
            b_p1 = []
            a_s1 = []
            b_s1 = []
            
            k = i+1
            
            for j in range(2*k+1):
                
                Hankel = Bessel_jn(k, self.alpha, derivative = False) +1j*Bessel_yn(k, self.alpha, derivative = False)
                xi = self.alpha*Hankel

                Hankel_der = Bessel_jn(k, self.alpha, derivative = True) +1j*Bessel_yn(k, self.alpha, derivative = True)
                xi_der = self.alpha*Hankel_der + Hankel
                
                psi,psi_der     = riccati_jn(k, self.alpha)
                psi_n,psi_der_n = riccati_jn(k, self.n32*self.alpha)
                
                m = j-k;
                
                Alm_p = (self.alpha1(k,m)/self.n21) * self.Tp * exp(-self.beta*self.h) * ((sin(self.theta1)*self.Q1(k,m))-((1j)*np.sqrt(sin(self.theta1)**2 - self.n21**2)*self.Q2(k,m)))
                Blm_p = self.n2*self.alpha1(k,m) * self.Tp * exp(-self.beta*self.h) * self.Q3(k,m)
                Alm_s = (self.Ts / (self.n2*self.Tp)) * Blm_p
                Blm_s = -((self.n2*self.Ts)/self.Tp) * Alm_p
                alm_p = ((psi_der_n[k]*psi[k] - self.n32*psi_n[k]*psi_der[k]) / (self.n32*psi_n[k]*xi_der - psi_der_n[k]*xi)) * Alm_p
                blm_p = ((self.n32*psi_der_n[k]*psi[k] - psi_n[k]*psi_der[k]) / (psi_n[k]*xi_der - self.n32*psi_der_n[k]*xi)) * Blm_p
                alm_s = ((psi_der_n[k]*psi[k] - self.n32*psi_n[k]*psi_der[k]) / (self.n32*psi_n[k]*xi_der - psi_der_n[k]*xi)) * Alm_s
                blm_s = ((self.n32*psi_der_n[k]*psi[k] - psi_n[k]*psi_der[k]) / (self.n32*psi_n[k]*xi_der - psi_der_n[k]*xi)) * Blm_s
                
                Ap1.append(Alm_p)
                Bp1.append(Blm_p)
                As1.append(Alm_s)
                Bs1.append(Blm_s)
                
                a_p1.append(alm_p)
                b_p1.append(blm_p)
                a_s1.append(alm_s)
                b_s1.append(blm_s)
                
                
                
            self.Ap.append(Ap1)
            self.Bp.append(Bp1)
            self.As.append(As1)
            self.Bs.append(Bs1)
            
            self.a_p.append(a_p1)
            self.b_p.append(b_p1)
            self.a_s.append(a_s1)
            self.b_s.append(b_s1)
            
        
        return self.Ap,self.As,self.Bp,self.Bs
    
    def force(self):
        
        f_xy = 0;
        f_z  = 0;
        
        fatt = (1j/2) * (self.alpha**2 / 2);
        
        n = self.n2**2
        
        a = self.a_s
        b = self.b_s
        A = self.As
        B = self.Bs
        
        for i in range(self.l-1):
            
            k = i+1
            
            for j in range(2*k+1):
                
                m = j-k;
                
                pf1 = np.sqrt(((k+m+2)*(k+m+1))/((2*k + 1)*(2*k + 3))) * k * (k+2);
                pf2 = np.sqrt(((k-m+2)*(k-m+1))/((2*k + 1)*(2*k + 3))) * k * (k+2);
                pf3 = np.sqrt((k+m+1)*(k-m)) * self.n2;
                
                add1 = pf1*(2*n*a[i][j]*np.conj(a[i+1][j+2]) + n*a[i][j]*np.conj(A[i+1][j+2]) + n*A[i][j]*np.conj(a[i+1][j+2]) + 2*b[i][k]*np.conj(b[i+1][j+2]) + b[i][k]*np.conj(B[i][k]) + B[i][j]*np.conj(b[i+1][j+2]) )
                add2 = pf2*(2*n*a[i+1][j]*np.conj(a[i][j]) + n*a[i+1][j]*np.conj(A[i][j]) + n*A[i+1][j]*np.conj(a[i][j]) + 2*b[i+1][j]*np.conj(b[i][j]) + b[i+1][j]*np.conj(B[i][j]) + B[i+1][j]*np.conj(b[i][k]))
                if j < i-1 :
                    add3 = pf3*(-2*a[i][j]*np.conj(b[i][j+1]) + 2*b[i][j]*np.conj(a[i][j+1]) - a[i][j]*np.conj(B[i][j+1]) + b[i][j]*np.conj(A[i][j]) + B[i][j]*np.conj(a[i][j+1]) + A[i][j]*np.conj(b[i][j+1]) )
                else:
                    add3 = 0
                
                
                f_xy = f_xy + add1 + add2 - add3;
                
        f_xy = fatt*f_xy;
        
        f_x = np.real(f_xy)
        f_y = np.imag(f_xy)
        
        return f_x,f_y,f_z
    
    
def Imz(m,z):
    arg = 1j*z
    res = ((1j)**(-m)) * jv(m,arg)
    # res = iv(m,z)
    #print(res)
    return res

def integr(theta):
    return np.exp(1.4*np.cos(theta))*np.cos(2*theta)
