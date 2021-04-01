'''Class that performs Mie calculations with an evanescent wave instead of an 
infinite plane wave. You need to install the module PyMieScatt.py to work with 
this library. The evanescent wave is produced using total internal reflection 
of an electromagnetic plane wave at the interface between a first medium with 
refractive index n1 and a second medium with refractive index n2 < n1.
For further details on Mie scattering calculations, see the documentation of 
the library PyMieScatt.py. We implement the formalism discussed in the article
Bekshaev, A. Y., Bliokh, K. Y., & Nori, F. (2013). 
"Mie scattering and optical forces from evanescent fields: A complex-angle 
approach." Optics Express, 21(6), 7082. https://doi.org/10.1364/oe.21.007082
This program uses CGS units.'''

from cmath import sin, cos, asin, acos, sinh, cosh, exp
import numpy as np
from math import pi
import PyMieScatt as ms
from scipy.special import clpmn as Legendre
from scipy.special import spherical_jn as Bessel_jn
from scipy.special import spherical_yn as Bessel_yn
from scipy import integrate
from matplotlib import pyplot as plt

eps0 = 8.854187813*1e-12
mu0 = 4*pi*1e-7

class SEW_experiment:
    
    '''
    This initializer takes arguments as follows:
        
        experiment = SEW_experiment.SEW_experiment(eps1, mu1, eps2, mu2, E0p, E0s, dist, lambd, theta, diam_part, indice_rifr)
        
        eps1: relative permittivity of the first medium.
        mu1: relative permeability of the first medium.
        eps2: relative permittivity of the second medium (with the current implementation you must set = 1).
        mu2: relative permeability of the second medium (with the current implementation you must set = 1).
        E0p: complex amplitude of the p component of the electric field.
        E0s: complex amplitude of the s component of the electric field.
        dist: distance of the CENTRE of the particle from the (flat) interface between the two media.
        lambd: wavelenght of the light in the first medium.
        theta: angle of incidence of the light with respect to the interface (must be greater than the
               critical angle, use radiants).
        diam_part: diameter of the particle.
        indice_rifr: complex refractive index of the particle.
        
    Remember that we use CGS units, so use cm instead of m!
    '''
    def __init__(self, eps1, mu1, eps2, mu2, E0p, E0s, dist, lambd, theta, diam_part, indice_rifr):
        
        self.eps1 = eps1
        self.mu1 = mu1
        self.eps2 = eps2
        self.mu2 = mu2
        self.E0p = E0p        
        self.E0s = E0s
        
        self.n1 = np.sqrt(eps1*mu1)
        self.n2 = np.sqrt(eps2*mu2)
        self.dist = dist
        
        self.gamma = asin(self.n1/self.n2*sin(theta))
        alpha = np.imag(self.gamma)
        self.alpha = alpha

        self.k = 2*pi/lambd

        exp_ = np.exp(-self.k*self.dist*sinh(alpha))
       
        self.Es = E0s*2*(mu2/mu1)*cos(theta)/((mu2/mu1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha))*exp_
        self.Ep = E0p*2*(self.n2/self.n1)*cos(theta)/((eps2/eps1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha))*exp_
        
        self.indice_rifr = indice_rifr
        self.size_param = self.k*diam_part/2

        self.raggio = diam_part/2
        
        an, bn = ms.Mie_ab(self.indice_rifr, self.size_param)
        self.an = an
        self.bn = bn


    '''
    The following function compute and returns the scattered e.m. field in a 
    given point outside the scatterer particle.
    
        E_theta, E_phi, E_r, H_theta, H_phi, H_r = experiment.ScatteredField(self, theta, phi, r,a)
        
        E_theta: polar component of the scattered electric field.
        E_phi: azimuthal component of the scattered electric field.
        E_r: radial component of the scattered electric field.
        H_theta: polar component of the scattered magnetic field.
        H_phi: azimuthal component of the scattered magnetic field.
        H_r: radial component of the scattered magnetic field.
        theta: polar coordinate of the point (use radiants).
        phi: azimuthal coordinate of the point (use radiants).
        r: radial coordinate of the point.
        a: used just for debugging. You may choose True or False.
    '''
    def ScatteredField(self, theta, phi, r,a):
        
        n_coeff = np.size(self.an)
        n = 0

        lp, lp_der = Legendre(1, n_coeff, cos(theta), type = 2)
    
        
        #lp, lp_der = Legendre(1, n_coeff+1, type = 3)
        
        E_theta = 0
        E_phi = 0
        E_r = 0
        
        H_theta = 0
        H_phi = 0
        H_r = 0
        test = 0
        for n in range(n_coeff):
            m = n+1
            
            
            
            
            prefactor = (1j**m)*(2*m+1)/(m*(m+1))
            
            Hankel = Bessel_jn(m, self.k*r, derivative = False) +1j*Bessel_yn(m, self.k*r, derivative = False)
            xi = self.k*r*Hankel
        
            Hankel_der = Bessel_jn(m, self.k*r, derivative = True) +1j*Bessel_yn(m, self.k*r, derivative = True)
            xi_der = self.k*r*Hankel_der + Hankel
            
            pi_ = -lp[1,m]/sin(theta)#-lp[1,m]/sin(theta)
            tau = sin(theta)*lp_der[1,m]#sin(theta)*lp_der[1,m]#
            
            Ep_ = cos(phi)*self.Es+sin(phi)*self.Ep
            Es_ = -sin(phi)*self.Es+cos(phi)*self.Ep
            
            #Hs_ = np.sqrt(self.eps2/self.mu2)*(-cos(phi)*self.Ep+sin(phi)*self.Es)
            #Hp_ = np.sqrt(self.eps2/self.mu2)*(sin(phi)*self.Ep+cos(phi)*self.Es)  
            
            E_theta = E_theta + Ep_*(1/(self.k*r))*prefactor*(1j*self.an[n]*xi_der*tau-self.bn[n]*xi*pi_)
            E_phi = E_phi - Es_*(1/(self.k*r))*prefactor*(self.bn[n]*xi*tau-1j*self.an[n]*xi_der*pi_)
            E_r = E_r + Ep_*sin(theta)*(1/(self.k*r)**2)*prefactor*(m*(m+1))*1j*self.an[n]*xi*pi_
            
            H_theta = H_theta - Es_*(1/(self.k*r))*np.sqrt(self.eps2/self.mu2)*prefactor*(1j*self.bn[n]*xi_der*tau-self.an[n]*xi*pi_)
            H_phi = H_phi + Ep_*(1/(self.k*r))*np.sqrt(self.eps2/self.mu2)*prefactor*(1j*self.bn[n]*xi_der*pi_-self.an[n]*xi*tau)
            H_r = H_r - Es_*sin(theta)*(1/(self.k*r)**2)*np.sqrt(self.eps2/self.mu2)*prefactor*(m*(m+1))*1j*self.bn[n]*xi*pi_            

        H_theta = H_theta#*np.sqrt(eps0/mu0)  
        H_phi = H_phi#*np.sqrt(eps0/mu0)  
        H_r = H_r#*np.sqrt(eps0/mu0)  
            
        return E_theta, E_phi, E_r, H_theta, H_phi, H_r
        
    '''
    The following function rotate a vector around the x axis. You can set a
    complex angle of rotation.
    
        Ex_, Ey_, Ez_ = experiment.complexAngleRotation(self, Ex, Ey, Ez, gamma)
    
        Ex_, Ey_, Ez_: cartesian components of the rotated vector.
        Ex, Ey, Ez: cartesian components of the initial vector.
        gamma: rotation angle (use radiants).
    '''
    def complexAngleRotation(self, Ex, Ey, Ez, gamma):

        Ey_ = Ey*cos(gamma)-Ez*sin(gamma)
        Ex_ = Ex
        Ez_ = Ey*sin(gamma)+Ez*cos(gamma)
        
        return Ex_, Ey_, Ez_
    
    '''
    The following function compute the and returns the e.m. field incident on 
    the particle (i.e., in the classical Mie problem, a plane wave) in a given
    point.
    
        E_theta, E_phi, E_r, H_theta, H_phi, H_r = experiment.IncidentField(self, theta, phi, r)

        E_theta: polar component of the scattered electric field.
        E_phi: azimuthal component of the scattered electric field.
        E_r: radial component of the scattered electric field.
        H_theta: polar component of the scattered magnetic field.
        H_phi: azimuthal component of the scattered magnetic field.
        H_r: radial component of the scattered magnetic field.
        theta: polar coordinate of the point (use radiants).
        phi: azimuthal coordinate of the point (use radiants).
        r: radial coordinate of the point.
    '''    
    
    def IncidentField(self, theta, phi, r):
        
        
        
        x,y,z = fromPolToCart(theta, phi, r)

        Ex = self.Es*exp(1j*self.k*z)
        Ey = self.Ep*exp(1j*self.k*z)
        Ez = 0
        
        Hx = -np.sqrt(self.eps2/self.mu2)*self.Ep*exp(1j*self.k*z)#*np.sqrt(eps0/mu0)
        Hy = np.sqrt(self.eps2/self.mu2)*self.Es*exp(1j*self.k*z)#*np.sqrt(eps0/mu0)
        Hz = 0
        
        E_theta, E_phi, E_r = fromCartToPolField(Ex, Ey, Ez, x ,y, z)
        H_theta, H_phi, H_r = fromCartToPolField(Hx, Hy, Hz, x ,y, z)

        Ex_,Ey_,Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma))

        
        return E_theta, E_phi, E_r, H_theta, H_phi, H_r
    
    '''    
    The following function compute the and returns the components of the 
    Maxwell stress tensor in a given point. To build the Maxwell stress tensor
    is used the TOTAL field E_tot = E_scattered + E_incident, so, we made use
    of the functions ScatteredField and IncidentField. All these field are
    considered to be harmonics.
    
        T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = experiment.MaxwellTensorDotR(self, theta, phi, r, a)
    
        T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz: cartesian components of 
                                                      the Maxwell stress tensor
        theta: polar coordinate of the point (use radiants).
        phi: azimuthal coordinate of the point (use radiants).
        r: radial coordinate of the point.
    '''
    
    def MaxwellTensorDotR(self, theta, phi, r, a):
        
        
        
        x,y,z = fromPolToCart(theta, phi, r)

        x_,y_,z_ = self.complexAngleRotation(x,y,z, np.conj(self.gamma))
       
        theta_, phi_, r_ = fromCartToPol(x_,y_,z_)

        E_theta_i, E_phi_i, E_r_i, H_theta_i, H_phi_i, H_r_i = self.IncidentField(theta_, phi_, r_)
        if a == True:
            E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_,True)
        else:
            E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_,False)
        
        E_theta = E_theta_s+E_theta_i
        E_phi = E_phi_s+E_phi_i
        E_r = E_r_s+E_r_i
        
        H_theta = H_theta_s+H_theta_i
        H_phi = H_phi_s+H_phi_i
        H_r = H_r_s+H_r_i
        
        
        Ex,Ey,Ez = fromPolToCartField(E_theta, E_phi, E_r, theta_, phi_) #controlla
        Ex_, Ey_, Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma)) 
        
        Hx,Hy,Hz = fromPolToCartField(H_theta, H_phi, H_r, theta_, phi_) #controlla
        Hx_, Hy_, Hz_ = self.complexAngleRotation(Hx,Hy,Hz,-np.conj(self.gamma)) 

        
        H2 = abs(Hx_)**2+abs(Hy_)**2+abs(Hz_)**2        
        E2 = abs(Ex_)**2+abs(Ey_)**2+abs(Ez_)**2  
        
        g = 1/(8*pi)#0.5;
        
        delta = 0.5*(self.eps2*E2+self.mu2*H2)
        
        T_xx = -g * np.real(-self.eps2*(np.conj(Ex_))*Ex_ - self.mu2*(np.conj(Hx_))*Hx_ + delta);
        T_yy = -g * np.real(-self.eps2*(np.conj(Ey_))*Ey_ - self.mu2*(np.conj(Hy_))*Hy_ + delta);
        T_zz = -g * np.real(-self.eps2*(np.conj(Ez_))*Ez_ - self.mu2*(np.conj(Hz_))*Hz_ + delta);
        T_xy = -g * np.real(-self.eps2*(np.conj(Ex_))*Ey_ - self.mu2*(np.conj(Hx_))*Hy_);
        T_yx = -g * np.real(-self.eps2*(np.conj(Ey_))*Ex_ - self.mu2*(np.conj(Hy_))*Hx_);
        T_xz = -g * np.real(-self.eps2*(np.conj(Ex_))*Ez_ - self.mu2*(np.conj(Hx_))*Hz_);
        T_zx = -g * np.real(-self.eps2*(np.conj(Ez_))*Ex_ - self.mu2*(np.conj(Hz_))*Hx_);
        T_zy = -g * np.real(-self.eps2*(np.conj(Ez_))*Ey_ - self.mu2*(np.conj(Hz_))*Hy_);
        T_yz = -g * np.real(-self.eps2*(np.conj(Ey_))*Ez_ - self.mu2*(np.conj(Hy_))*Hz_);
        
        
        return T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz
    '''
    The following function performs the integration of the scalar product
    between the versor normal to an ideal sphere with its centre coincident
    with the centre of the particle, on the surface of the sphere itself.
    Note that for simplicity we MUST use cartesian coordinate to perform the
    integration.
    
        Fx, Fy, Fz = experiment.IntegrateOnSphere(self)
        
        Fx, Fy, Fz: tuples, containing the three cartesian component (first 
                    component of the tuple) of the optical force acting on the
                    particle, and their confidence intervals (second).
    '''
    def IntegrateOnSphere(self):
        raggio = self.raggio#+10e-8;
        def dFx(theta,phi):
            T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(theta,phi,raggio, False);
            dF = T_xx*(sin(theta)*cos(phi)) + T_xy*(sin(theta)*sin(phi)) + T_xz*cos(theta);
            return np.real( dF * sin(theta) * raggio**2 )
        
        def dFy(theta,phi):
            T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(theta,phi,raggio, False);
            dF = T_yx*(sin(theta)*cos(phi)) + T_yy*(sin(theta)*sin(phi)) + T_yz*cos(theta);
            return np.real( dF * sin(theta) * raggio**2 )
        
        def dFz(theta,phi):
            T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(theta,phi,raggio, False);
            dF = T_zx*(sin(theta)*cos(phi)) + T_zy*(sin(theta)*sin(phi)) + T_zz*cos(theta);
            return np.real( dF * sin(theta) * raggio**2 )
        
        F_x = integrate.nquad(dFx, [[0,pi],[0,2*pi]]);    
        
        F_y = integrate.nquad(dFy, [[0,pi],[0,2*pi]]);
        
        F_z = integrate.nquad(dFz, [[0,pi],[0,2*pi]]);

        return F_x,F_y,F_z

    def Fz_dipole(self):
        exp_ = np.exp(-2*self.k*self.dist*sinh(self.alpha))
        Fz = -0.5*4*pi*eps0*(abs(self.Ep)**2+abs(self.Es)**2*(2*cosh(self.alpha)**2-1))*self.k*self.raggio**3*sinh(self.alpha)*np.real((self.indice_rifr**2-self.eps2)/(self.indice_rifr**2+2*self.eps2))*exp_
        return Fz
                 
    def ManualIntegrateOnSphere(self):
        raggio = self.raggio +1e-8;
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
        
        npunti = 100
        lato_theta = pi/npunti
        lato_phi = 2*pi/npunti
        
        int_x = 0
        int_y = 0
        int_z = 0
        
        for k in range(npunti):
            for n in range(npunti):
                int_x = int_x + dFx((0.5+k)*lato_theta, (0.5+n)*lato_phi)*lato_phi*lato_theta
                int_y = int_y + dFy((0.5+k)*lato_theta, (0.5+n)*lato_phi)*lato_phi*lato_theta
                int_z = int_z + dFz((0.5+k)*lato_theta, (0.5+n)*lato_phi)*lato_phi*lato_theta
                     
        return int_x, int_y, int_z
    '''
    The following function computes the scattered power in a cone with axis 
    perpendicular to the interface between the two media, and given aperture.
    
        P, Perr = experiment.scatteredPower(self, _theta_)
        
        _theta_: aperture of the cone.
        P: scattered power.
        P: error on scattered power.
    '''
    def scatteredPower(self, _theta_):
        raggio = self.raggio;
        def rPoynting(theta,phi):
            
            x,y,z = fromPolToCart(theta, phi, raggio)

            x_,y_,z_ = self.complexAngleRotation(x,y,z, np.conj(self.gamma))
            theta_, phi_, r_ = fromCartToPol(x_,y_,z_)

            et, ep, er, ht, hp, hr = self.ScatteredField(theta_,phi_,r_,False)
            
            Ex,Ey,Ez = fromPolToCartField(et,ep,er, theta_, phi_) #controlla
            Ex_, Ey_, Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma)) 
        
            Hx,Hy,Hz = fromPolToCartField(ht,hp,hr, theta_, phi_) #controlla
            Hx_, Hy_, Hz_ = self.complexAngleRotation(Hx,Hy,Hz,-np.conj(self.gamma))             
            
            Er, Et, Ep = fromCartToPolField(Ex_, Ey_, Ez_, x, y, z)
            Hr, Ht, Hp = fromCartToPolField(Hx_, Hy_, Hz_, x, y, z)
            
            integrand = Et*np.conj(Hp)-Ep*np.conj(Ht)
            
            return np.real(integrand) * np.real(sin(theta)) * (100*raggio)**2

        W = integrate.nquad(rPoynting, [[pi-_theta_,pi],[0,2*pi]]);

        return -0.5*W[0], -0.5*W[1] 
    
    '''
    The following function computes the intensity of the scattered light in a 
    given point outside the particle.
    
        I = experiment.scatteredIntensity_point(self, x,y,z)
    
        I: scattered intensity.
        x,y,z: cartesian coordinates of the point.
    '''
    def scatteredIntensity_point(self, x,y,z):

        x_,y_,z_ = self.complexAngleRotation(x,y,z, np.conj(self.gamma))
        theta_, phi_, r_ = fromCartToPol(x_,y_,z_)
        
        et, ep, er, ht, hp, hr = self.ScatteredField(theta_,phi_,r_,False)
        
        Ex,Ey,Ez = fromPolToCartField(et,ep,er, theta_, phi_) #controlla
        Ex_, Ey_, Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma)) 
    
        Hx,Hy,Hz = fromPolToCartField(ht,hp,hr, theta_, phi_) #controlla
        Hx_, Hy_, Hz_ = self.complexAngleRotation(Hx,Hy,Hz,-np.conj(self.gamma))             
        
        Sz = Ex_*np.conj(Hy_)-Ey_*np.conj(Hx_)

        return -0.5*np.real(Sz)
    
    '''
    The following function plots the intensity scattered on a flat square 
    surface located undaer the particle at the interface between the two media.
    You can choose the number of points to sample. The side of the square is 
    fixed as depicted below:
        
        glass = experiment.light_on_glass(self,npunti, _theta_)
              
             + +
          +       +
         +    *    +
          + * |<*-+----- _theta_
          *  +|+  *                 n1
    ----*-----|-----*-----------------
      *       |       *             n2
              |
        |___________|     
        square's side
        
        
        glass: matrix containing the sampled intensity.     
        npunti: number of samplings along the side of the square.
        
    Warning: with the current implementation this function works only if the
    particle is in contact with the interfaces and is setselt.dist = self.raggio
    '''
    
    def light_on_glass(self,npunti, _theta_):
        
        x_extr = self.dist*np.tan(_theta_)
        lato = 2*x_extr/npunti
        
        glass = np.empty((npunti,npunti))
        
        for i in range(npunti):
            for j in range(npunti):
                
                glass[i,j] = self.scatteredIntensity_point(-x_extr+(0.5+i)*lato, -x_extr+(0.5+j)*lato, self.dist)
                
        plt.imshow(glass, cmap='Blues_r')        
        return glass
         
def fromPolToCartField(E_theta,E_phi,E_r, theta, phi):
        
    Ex = cos(theta)*cos(phi)*E_theta - sin(phi)*E_phi + sin(theta)*cos(phi)*E_r
    Ey = cos(theta)*sin(phi)*E_theta + cos(phi)*E_phi + sin(theta)*sin(phi)*E_r
    Ez = -sin(theta)*E_theta + cos(theta)*E_r
        
    return Ex, Ey, Ez
    
def fromCartToPolField(Ex, Ey, Ez, x, y, z):
    
    asin_phi = asin(y/np.sqrt(x**2+y**2))
    acos_phi = acos(x/np.sqrt(x**2+y**2))
    theta = acos(z/np.sqrt(x**2+y**2+z**2))
    
    if np.real(acos_phi+asin_phi) < 3.14159266 and np.real(acos_phi+asin_phi) > 3.14159265:
        phi = acos_phi
    elif np.real(acos_phi-asin_phi) < 3.14159266 and np.real(acos_phi-asin_phi) > 3.14159265:
        phi = pi - asin_phi
    elif np.real(acos_phi) < -np.real(asin_phi) +1e-5 and np.real(acos_phi) > -np.real(asin_phi) -1e-5:
        phi = 2*pi + asin_phi
    else:
        phi = acos_phi
    
    E_theta = Ey*sin(phi)*cos(theta)+Ex*cos(phi)*cos(theta)-Ez*sin(theta)
    E_phi = -Ex*sin(phi)+Ey*cos(phi)
    E_r = Ex*cos(phi)*sin(theta)+Ey*sin(phi)*sin(theta)+Ez*cos(theta)
        
    return E_theta, E_phi, E_r    
        
def fromCartToPol(x,y,z):
    
    r = np.sqrt(x**2+y**2+z**2)
    asin_phi = asin(y/np.sqrt(x**2+y**2))
    acos_phi = acos(x/np.sqrt(x**2+y**2))
    theta = acos(z/np.sqrt(x**2+y**2+z**2))
    
    if np.real(acos_phi+asin_phi) < 3.14159266 and np.real(acos_phi+asin_phi) > 3.14159265:
        #print('ciao')
        phi = acos_phi
    elif np.real(acos_phi-asin_phi) < 3.14159266 and np.real(acos_phi-asin_phi) > 3.14159265:
        #print('ciao')
        phi = pi - asin_phi
    elif np.real(acos_phi) < -np.real(asin_phi) +1e-5 and np.real(acos_phi) > -np.real(asin_phi) -1e-5:
        phi = 2*pi + asin_phi
    else:
        #print('ciao', asin_phi)
        phi = acos_phi
    '''
    if x!=0 and y!=0:
        if y < 0:    
            phi = 2*pi-acos(x/np.sqrt(x**2+y**2))
            theta = asin(np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2))
        else:
            phi = acos(x/np.sqrt(x**2+y**2))
            theta = asin(np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2))
    else:
        phi = 1e-10;
        theta = 1e-10;'''
    
    return theta, phi, r

def fromPolToCart(theta, phi, r):
    
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    
    return x, y, z

'''
The following function computes the effective permittivity of the particle,
given the refractive index of the bulk material, the radius of the particle 
and the wavelength of the light. For further details, see Lucía B Scaffardi, 
and Jorge O Tocho "Size dependence of refractive index of gold nanoparticles",
Nanotecnology, 2006 https://core.ac.uk/reader/296378380

    epsilon = refractive_index_vs_radius(ref_ind,raggio,lambd) 
    
    epsilon: effective permittivity of the particle.
    ref_ind: refractive index of the bulk material.
    raggio: radius of the particle.
    lambd: wavelenght of the incident light.
'''

def refractive_index_vs_radius(ref_ind,raggio,lambd):
    
    c = 1/np.sqrt(mu0*eps0)
    h_tagliato = 6.582119569*1e-16 #eV*s
    omega = c*2*pi/lambd #in aria!
    omega_p = 13e+15 #Hz, Granqvist
    gamma_free = 1.1e+14 + 0.8*14.1e+14/raggio #(5)
    gamma_b = 2.4e+14
    Q_size = 2.3*1e+24*(1-exp(raggio/(3.5e-10))) #Hz
    k_b = 8.617333262 #eV*K
    omega_g = 2.1/h_tagliato
    T = 300
    
    eps_free = 1-omega_p**2/(omega**2+1j*gamma_free*omega)

    def integrand_real(x):
        
        def Fermi(y,T_):
            return 1/(1+np.exp((h_tagliato*omega-2.5)/(k_b*T_)))
        
        return np.real(np.sqrt(x-omega_g)/x*(1-Fermi(x,T))*(x**2-omega**2+gamma_b**2+2j*omega*gamma_b)/((x**2-omega**2+gamma_b**2+gamma_b**2)**2+4*omega**2*gamma_b**2))

    def integrand_imag(x):
        
        def Fermi(y,T_):
            return 1/(1+np.exp((h_tagliato*omega-2.5)/(k_b*T_)))
        
        return np.imag(np.sqrt(x-omega_g)/x*(1-Fermi(x,T))*(x**2-omega**2+gamma_b**2+2j*omega*gamma_b)/((x**2-omega**2+gamma_b**2+gamma_b**2)**2+4*omega**2*gamma_b**2))
        
        
    eps_bound_real = integrate.quad(integrand_real, omega_g, np.inf)
    eps_bound_imag = integrate.quad(integrand_imag, omega_g, np.inf)
    eps_bound = (eps_bound_real[0]+1j*eps_bound_imag[0])*Q_size
    
    return eps_free + eps_bound
    

def f(x):
    return abs((0.46236138708596713-1.1851104600554436j))*exp(-2*pi*x/800e-9)