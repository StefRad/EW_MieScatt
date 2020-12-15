from cmath import sin, cos, asin, acos, sinh, cosh, exp
import numpy as np
from math import pi
import PyMieScatt as ms
from scipy.special import clpmn as Legendre
from scipy.special import spherical_jn as Bessel_jn
from scipy.special import spherical_yn as Bessel_yn
from scipy import integrate


class SEW_experiment:
    
    def __init__(self, eps1, mu1, eps2, mu2, E0p, E0s, dist, lambd, theta, diam_part, indice_rifr):
        
        self.eps1 = eps1
        self.mu1 = mu1
        self.eps2 = eps2
        self.mu2 = mu2
        
        self.n1 = np.sqrt(eps1*mu1)
        self.n2 = np.sqrt(eps2*mu2)
        self.dist = dist
        self.gamma = asin(self.n1/self.n2*sin(theta))
        alpha = -np.imag(self.gamma)
        self.k = 2*pi/lambd
        exp_ = np.exp(-self.k*dist*sinh(alpha))
        
        self.Es = E0s*2*(mu2/mu1)*cos(theta)/((mu2/mu1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha))*exp_
        self.Ep = E0p*2*(self.n2/self.n1)*cos(theta)/((eps2/eps1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha))*exp_
        
        self.indice_rifr = indice_rifr
        self.size_param = self.k*diam_part/2
        self.raggio = diam_part / 2;
        
        an, bn = ms.Mie_ab(self.indice_rifr, self.size_param)
        self.an = an
        self.bn = bn
        
    def ScatteredField(self, theta, phi, r):
        
        n_coeff = np.size(self.an)
        n = 0
        
        lp, lp_der = Legendre(0, n_coeff, cos(theta), type = 3)
        
        #prefactors = []
        
        #lp, lp_der = Legendre(1, n_coeff+1, type = 3)
        
        E_theta = 0
        E_phi = 0
        E_r = 0
        
        H_theta = 0
        H_phi = 0
        H_r = 0
        
        for n in range(n_coeff):
            m = n+1
            
            
            
            
            prefactor = (1j**m)*(2*m+1)/(m*(m+1))
            
            Hankel = Bessel_jn(n, self.k*r, derivative = False) +1j*Bessel_yn(n, self.k*r, derivative = False)
            xi = self.k*r*Hankel
        
            Hankel_der = Bessel_jn(n, self.k*r, derivative = True) +1j*Bessel_yn(n, self.k*r, derivative = True)
            xi_der = self.k*r*Hankel_der + Hankel
            
            pi = lp[0,n]/sin(theta)
            tau = sin(theta)*lp_der[0,n]
            
            # print ("iterazione:")
            # print(n)
            
            # print("tau:")
            # print(tau)
            
            # print("pi:")
            # print(pi)
            
            Ep_ = cos(phi)*self.Ep+sin(phi)*self.Es
            Es_ = -sin(phi)*self.Ep+cos(phi)*self.Es
            
            #Hs_ = np.sqrt(self.eps2/self.mu2)*(-cos(phi)*self.Ep+sin(phi)*self.Es)
            #Hp_ = np.sqrt(self.eps2/self.mu2)*(sin(phi)*self.Ep+cos(phi)*self.Es)  
            
            E_theta = E_theta + Ep_*(1/r)*prefactor*(1j*self.an[n]*xi_der*tau-self.bn[n]*xi*pi)
            E_phi = E_phi + Es_*(1/r)*prefactor*(self.bn[n]*xi*tau-1j*self.an[n]*xi_der*pi)
            E_r = E_r + Ep_*sin(theta)*(1/r**2)*prefactor*(m*(m+1))*1j*self.an[n]*xi*pi
            
            H_theta = H_theta - Es_*(1/r)*np.sqrt(self.eps2/self.mu2)*prefactor*(1j*self.bn[n]*xi_der*tau-self.an[n]*xi*pi)
            H_phi = H_phi + Ep_*(1/r)*np.sqrt(self.eps2/self.mu2)*prefactor*(1j*self.bn[n]*xi_der*pi-self.an[n]*xi*tau)
            H_r = H_r - Es_*sin(theta)*(1/r**2)*np.sqrt(self.eps2/self.mu2)*prefactor*(m*(m+1))*1j*self.bn[n]*xi*pi            
            
        return E_theta, E_phi, E_r, H_theta, H_phi, H_r
        
    def complexAngleRotation(self, Ex, Ey, Ez, gamma):
        
        Ex_ = Ex*cos(gamma)-Ez*sin(gamma)
        Ey_ = Ey
        Ez_ = Ez*sin(gamma)+Ez*cos(gamma)
        
        return Ex_, Ey_, Ez_
    
    def IncidentField(self, theta, phi, r):
        
        
        
        x,y,z = fromPolToCart(theta, phi, r)
        
        Ex = self.Ep*exp(1j*self.k*z)
        Ey = self.Es*exp(1j*self.k*z)
        Ez = 0
        
        Hx = -np.sqrt(self.eps2/self.mu2)*self.Es*exp(1j*self.k*z)
        Hy = np.sqrt(self.eps2/self.mu2)*self.Ep*exp(1j*self.k*z)
        Hz = 0
        
        E_theta, E_phi, E_r = fromCartToPolField(Ex, Ey, Ez, x ,y, z)
        H_theta, H_phi, H_r = fromCartToPolField(Hx, Hy, Hz, x ,y, z)
        
        return E_theta, E_phi, E_r, H_theta, H_phi, H_r
    
    def MaxwellTensorDotR(self, theta, phi, r):
        
        
        
        x,y,z = fromPolToCart(theta, phi, r)
        x_,y_,z_ = self.complexAngleRotation(x,y,z, self.gamma)
        
        theta_, phi_, r_ = fromCartToPol(x_,y_,z_)

        E_theta_i, E_phi_i, E_r_i, H_theta_i, H_phi_i, H_r_i = self.IncidentField(theta_, phi_, r_)
        E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_)
        
        E_theta = E_theta_i+E_theta_s
        E_phi = E_phi_i+E_phi_s
        E_r = E_r_i+E_r_s
        
        H_theta = H_theta_i+H_theta_s
        H_phi = H_phi_i+H_phi_s
        H_r = H_r_i+H_r_s
        

        Ex,Ey,Ez = fromPolToCartField(E_theta, E_phi, E_r, theta, phi) #controlla
        Ex_, Ey_, Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-self.gamma) 
        
        Hx,Hy,Hz = fromPolToCartField(H_theta, H_phi, H_r, theta, phi) #controlla
        Hx_, Hy_, Hz_ = self.complexAngleRotation(Hx,Hy,Hz,-self.gamma) 

        H2 = abs(Hx_)**2+abs(Hy_)**2+abs(Hz_)**2        
        E2 = abs(Ex_)**2+abs(Ey_)**2+abs(Ez_)**2  
        
        g = 0.5;
        
        delta = 0.5*(self.eps2*E2+self.mu2*H2)
        
        T_xx = g * np.real(self.eps2*(np.conj(Ex_))*Ex_ + self.mu2*(np.conj(Hx_))*Hx_ - delta);
        T_yy = g * np.real(self.eps2*(np.conj(Ey_))*Ey_ + self.mu2*(np.conj(Hy_))*Hy_ - delta);
        T_zz = g * np.real(self.eps2*(np.conj(Ez_))*Ez_ + self.mu2*(np.conj(Hz_))*Hz_ - delta);
        T_xy = g * np.real(self.eps2*(np.conj(Ex_))*Ey_ + self.mu2*(np.conj(Hx_))*Hy_);
        T_yx = g * np.real(self.eps2*(np.conj(Ey_))*Ex_ + self.mu2*(np.conj(Hy_))*Hx_);
        T_xz = g * np.real(self.eps2*(np.conj(Ex_))*Ez_ + self.mu2*(np.conj(Hx_))*Hz_);
        T_zx = g * np.real(self.eps2*(np.conj(Ez_))*Ex_ + self.mu2*(np.conj(Hz_))*Hx_);
        T_zy = g * np.real(self.eps2*(np.conj(Ez_))*Ey_ + self.mu2*(np.conj(Hz_))*Hy_);
        T_yz = g * np.real(self.eps2*(np.conj(Ey_))*Ez_ + self.mu2*(np.conj(Hy_))*Hz_);
        
        # E_theta, E_phi, E_r = fromCartToPolField(Ex_, Ey_, Ez_, x, y, z) #controlla
        # H_theta, H_phi, H_r = fromCartToPolField(Hx_, Hy_, Hz_, x, y, z) #controlla
        
        # T_dot_r_theta = (1/(8*pi))*np.real(self.eps2*(np.conj(E_theta))*E_r+ self.mu2*(np.conj(H_theta))*H_r)
        # T_dot_r_phi =(1/(8*pi))*np.real(self.eps2*(np.conj(E_phi))*E_r+ self.mu2*(np.conj(H_phi))*H_r)
        # T_dot_r_r = (1/(8*pi))*np.real(self.eps2*np.conj(E_r)*E_r+self.mu2*(np.conj(H_r)*H_r)-1/2*(self.eps2*E2+self.mu2*H2))
    
        return T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz

    def IntegrateOnSphere(self):
        raggio = self.raggio;
        def dFx(theta,phi):
            T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(theta,phi,raggio);
            dF = T_xx*(sin(theta)*cos(phi)) + T_xy*(sin(theta)*sin(phi)) + T_xz*cos(theta);
            return np.real( dF * sin(theta) * raggio**2 )
        
        def dFy(theta,phi):
            T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(theta,phi,raggio);
            dF = T_yx*(sin(theta)*cos(phi)) + T_yy*(sin(theta)*sin(phi)) + T_yz*cos(theta);
            return np.real( dF * sin(theta) * raggio**2 )
        
        def dFz(theta,phi):
            T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(theta,phi,raggio);
            dF = T_zx*(sin(theta)*cos(phi)) + T_zy*(sin(theta)*sin(phi)) + T_zz*cos(theta);
            return np.real( dF * sin(theta) * raggio**2 )
        
        F_x = integrate.nquad(dFx, [[0,pi],[0,2*pi]]);
        
        F_y = integrate.nquad(dFy, [[0,pi],[0,2*pi]]);
        
        F_z = integrate.nquad(dFz, [[0,pi],[0,2*pi]]);
        
        return F_x,F_y,F_z

                                 
                                       
def fromPolToCartField(E_theta,E_phi,E_r, theta, phi):
        
    Ex = cos(theta)*cos(phi)*E_theta - sin(phi)*E_phi + sin(theta)*cos(phi)*E_r
    Ey = cos(theta)*sin(phi)*E_theta + cos(phi)*E_phi + sin(theta)*sin(phi)*E_r
    Ez = -sin(theta)*E_theta + cos(theta)*E_r
        
    return Ex, Ey, Ez
    
def fromCartToPolField(Ex, Ey, Ez, x, y, z):
    
    if x!=0 and y!=0:
        phi = acos(x/np.sqrt(x**2+y**2))
        theta = asin(np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2))
    else:
        phi = 1e-10;
        theta = 1e-10;
    
    E_theta = Ey*sin(phi)*cos(theta)+Ex*cos(phi)*cos(theta)-Ez*sin(theta)
    E_phi = Ex*sin(phi)+Ey*cos(phi)
    E_r = Ex*cos(phi)*sin(theta)+Ey*sin(phi)*sin(theta)+Ez*cos(theta)
        
    return E_theta, E_phi, E_r    
        
def fromCartToPol(x,y,z):
    
    r = np.sqrt(x**2+y**2+z**2)
    if x!=0 and y!=0:
        phi = acos(x/np.sqrt(x**2+y**2))
        theta = asin(np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2))
    else:
        phi = 1e-10;
        theta = 1e-10;
    
    return theta, phi, r

def fromPolToCart(theta, phi, r):
    
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    
    return x, y, z