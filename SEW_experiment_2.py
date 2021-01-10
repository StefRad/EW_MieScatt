from cmath import sin, cos, asin, acos, sinh, cosh, exp
import numpy as np
from math import pi
from scipy.special import clpmn as Legendre
from scipy.special import spherical_jn as Bessel_jn
from scipy.special import spherical_yn as Bessel_yn
from scipy import integrate

class SEW_experiment_2:

    def __init__(self, eps1, mu1, eps2, mu2, E0p, E0s, dist, lambd, theta, diam_part, indice_rifr, phase=0):

        self.eps1 = eps1
        self.mu1 = mu1
        self.eps2 = eps2
        self.mu2 = mu2

        self.n1 = np.sqrt(eps1*mu1)
        self.n2 = np.sqrt(eps2*mu2)
        self.dist = dist
        self.gamma = asin(self.n1/self.n2*sin(theta))
        alpha = np.imag(self.gamma)
        #print(sinh(alpha))
        self.k = 2*pi/lambd
        exp_ = np.exp(-self.k*dist*sinh(alpha))

        self.Es = np.sqrt(4*pi*8.8542*1e-12)*((E0s*2*(mu2/mu1)*cos(theta))/((mu2/mu1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha)))*exp_
        self.Ep = np.sqrt(4*pi*8.8542*1e-12)*((E0p*2*(self.n2/self.n1)*cos(theta))/((eps2/eps1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha)))*exp_
        self.ph = phase

        self.indice_rifr = indice_rifr
        self.size_param = self.k*diam_part/2
        self.raggio = diam_part / 2;

    def ScatteredField(self, theta, phi, r):

        n_coeff = int(2+self.size_param+4*(self.size_param)**(1/3))
        n = 0

        lp, lp_der = Legendre(1, n_coeff, cos(theta), type = 3)

        E_theta = 0
        E_phi = 0
        E_r = 0

        H_theta = 0
        H_phi = 0
        H_r = 0

        m_eps = np.sqrt((self.indice_rifr**2)/self.eps2);
        m_mu  = 1 ;# TODO: in general case and SI
        m_n   = self.indice_rifr / np.sqrt(self.eps2);

        for n in range(n_coeff):
            m = n+1




            prefactor = (1j**m)*(2*m+1)/(m*(m+1))

            Hankel = Bessel_jn(n, self.k*r, derivative = False) +1j*Bessel_yn(n, self.k*r, derivative = False)
            xi = self.k*r*Hankel

            Hankel_der = Bessel_jn(n, self.k*r, derivative = True) +1j*Bessel_yn(n, self.k*r, derivative = True)
            xi_der = self.k*r*Hankel_der + Hankel

            psi_chi  = self.size_param * Bessel_jn(n, self.size_param, derivative = False)
            psi_mchi = m_n * self.size_param * Bessel_jn(n, m_n * self.size_param, derivative = False)

            psi_chi_der  = self.size_param * Bessel_jn(n, self.size_param, derivative = True) + Bessel_jn(n, self.size_param, derivative = False)
            psi_mchi_der = m_n * self.size_param * Bessel_jn(n, m_n * self.size_param, derivative = True) + Bessel_jn(n, m_n * self.size_param, derivative = False)

            a = ((m_eps*psi_mchi*psi_chi_der) - (m_mu*psi_chi*psi_mchi_der)) / ((m_eps*psi_mchi*xi_der)-(m_mu*xi*psi_mchi_der));
            b = ((m_mu*psi_mchi*psi_chi_der) - (m_eps*psi_chi*psi_mchi_der)) / ((m_mu*psi_mchi*xi_der)-(m_eps*xi*psi_mchi_der));

            piu = lp[1,m]/sin(theta)
            tau = -sin(theta)*lp_der[1,m]

            # print ("iterazione:")
            # print(n)

            # print("tau:")
            # print(tau)

            # print("pi:")
            # print(pi)

            Ep_ = cos(phi)*self.Ep+sin(phi)*self.Es
            Es_ = -sin(phi)*self.Ep+cos(phi)*self.Es

            E_theta = E_theta + Ep_*(1/(self.k*r))*prefactor*(1j*a*xi_der*tau-b*xi*piu)
            E_phi = E_phi + Es_*(1/(self.k*r))*prefactor*(b*xi*tau-1j*a*xi_der*piu)
            E_r = E_r + Ep_*sin(theta)*(1/(self.k*r)**2)*prefactor*(m*(m+1))*1j*a*xi*piu

            H_theta = H_theta - Es_*(1/(self.k*r))*np.sqrt((self.eps2)/(self.mu2))*prefactor*(1j*b*xi_der*tau-a*xi*piu)
            H_phi = H_phi + Ep_*(1/(self.k*r))*np.sqrt((self.eps2)/(self.mu2))*prefactor*(1j*b*xi_der*piu-a*xi*tau)
            H_r = H_r - Es_*sin(theta)*(1/(self.k*r)**2)*np.sqrt((self.eps2)/(self.mu2))*prefactor*(m*(m+1))*1j*b*xi*piu

            return E_theta, E_phi, E_r, H_theta, H_phi, H_r

    def complexAngleRotation(self, Ex, Ey, Ez, gamma):

        Ex_ = Ex*cos(gamma)-Ez*sin(gamma)
        Ey_ = Ey
        Ez_ = Ex*sin(gamma)+Ez*cos(gamma)

        return Ex_, Ey_, Ez_

    def IncidentField(self, theta, phi, r):



        x,y,z = fromPolToCart(theta, phi, r)

        Ex = self.Ep*exp(1j*self.k*z)
        Ey = self.Es*exp(1j*self.k*z+self.ph)
        Ez = 0

        Hx = -np.sqrt(1.2*1e-6)*np.sqrt((self.eps2)/(self.mu2))*self.Es*exp(1j*self.k*z)
        Hy = np.sqrt(1.2*1e-6)*np.sqrt((self.eps2)/(self.mu2))*self.Ep*exp(1j*self.k*z)
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

        delta = 0.5 * ( (self.eps2*E2) + (self.mu2*H2) )

        T_xx = g * np.real(self.eps2*(np.conj(Ex_))*Ex_ + self.mu2*(np.conj(Hx_))*Hx_ - delta);
        T_yy = g * np.real(self.eps2*(np.conj(Ey_))*Ey_ + self.mu2*(np.conj(Hy_))*Hy_ - delta);
        T_zz = g * np.real(self.eps2*(np.conj(Ez_))*Ez_ + self.mu2*(np.conj(Hz_))*Hz_ - delta);
        T_xy = g * np.real(self.eps2*(np.conj(Ex_))*Ey_ + self.mu2*(np.conj(Hx_))*Hy_);
        T_yx = g * np.real(self.eps2*(np.conj(Ey_))*Ex_ + self.mu2*(np.conj(Hy_))*Hx_);
        T_xz = g * np.real(self.eps2*(np.conj(Ex_))*Ez_ + self.mu2*(np.conj(Hx_))*Hz_);
        T_zx = g * np.real(self.eps2*(np.conj(Ez_))*Ex_ + self.mu2*(np.conj(Hz_))*Hx_);
        T_zy = g * np.real(self.eps2*(np.conj(Ez_))*Ey_ + self.mu2*(np.conj(Hz_))*Hy_);
        T_yz = g * np.real(self.eps2*(np.conj(Ey_))*Ez_ + self.mu2*(np.conj(Hy_))*Hz_);


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
        phi = 1e-15;
        theta = 1e-15;

    E_theta = Ey*sin(phi)*cos(theta)+Ex*cos(phi)*cos(theta)-Ez*sin(theta)
    E_phi = -Ex*sin(phi)+Ey*cos(phi)
    E_r = Ex*cos(phi)*sin(theta)+Ey*sin(phi)*sin(theta)+Ez*cos(theta)

    return E_theta, E_phi, E_r

def fromCartToPol(x,y,z):

    r = np.sqrt(x**2+y**2+z**2)
    if x!=0 and y!=0:
        phi = acos(x/np.sqrt(x**2+y**2))
        theta = asin(np.sqrt(x**2+y**2)/np.sqrt(x**2+y**2+z**2))
    else:
        phi = 1e-15;
        theta = 1e-15;

    return theta, phi, r

def fromPolToCart(theta, phi, r):

    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)

    return x, y, z
