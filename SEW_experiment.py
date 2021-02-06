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
        #print(alpha)
        self.k = 2*pi/lambd
        #self.dist=1.5/self.k#########
        exp_ = np.exp(-self.k*self.dist*sinh(alpha))
        #print(sinh#)
        #print(self.dist*self.k)
        #print(exp_)
        #print(2*(self.n2/self.n1)*cos(theta))
        #print(((eps2/eps1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha)))
        self.Es = E0s*2*(mu2/mu1)*cos(theta)/((mu2/mu1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha))*exp_
        self.Ep = E0p*2*(self.n2/self.n1)*cos(theta)/((eps2/eps1)*cos(theta)+1j*(self.n2/self.n1)*sinh(alpha))*exp_
        #print(' Fresnel = ',2*(self.n1/self.n2)*cos(theta)/((mu1/mu2)*cos(theta)+1j*(self.n1/self.n2)*sinh(alpha)))
        self.indice_rifr = indice_rifr
        self.size_param = self.k*diam_part/2
        #self.size_param = 1.5#########
        self.raggio = diam_part/2
        
        an, bn = ms.Mie_ab(self.indice_rifr, self.size_param)
        self.an = an
        self.bn = bn
        #print(self.an) 
        '''
        print(self.size_param)

        n_coeff = np.size(self.an)
        n = 0   
        S1=0
        S2=0

        theta = 0.8        
        lp, lp_der = Legendre(1, n_coeff, cos(theta), type = 2)        
        
        for n in range(n_coeff):
            
            m = n+1
            prefactor = (2*m+1)/(m*(m+1))   
            pi_ = -lp[1,m]/sin(theta)
            tau = sin(theta)*lp_der[1,m]
            
            S1 = S1+prefactor*(self.an[n]*pi_+self.bn[n]*tau)
            S2 = S2+prefactor*(self.an[n]*tau+self.bn[n]*pi_)
            
            print(pi_)
            
        print('S1 = ', S1)
        print('S2 = ', S2)
        #print(cos(theta))
        
        x = self.size_param
        nmax = np.round(2+x+4*np.power(x,1/3))
        #an, bn = ms.AutoMie_ab(self.indice_rifr,x)
        pin, taun = ms.MiePiTau(np.real(cos(theta)),nmax)
        n = np.arange(1,int(nmax)+1)
        n2 = (2*n+1)/(n*(n+1))
        s1 = np.sum(n2[0:len(an)]*(pin[0:len(an)]*an+taun[0:len(bn)]*bn))
        s2 = np.sum(n2[0:len(an)]*(an*taun[0:len(an)]+bn*pin[0:len(bn)]))        
        
        print(pin)
        #s1,s2 = ms.MieS1S2(indice_rifr,self.size_param, np.real(cos(theta)))

        #print(np.size(self.an))
        #print(len(bn))

        print('s1 = ', s1)
        print('s2 = ', s2)
        print()        
        '''
    def ScatteredField(self, theta, phi, r,a):
        
        n_coeff = np.size(self.an)
        n = 0
        #print(self.an)
        lp, lp_der = Legendre(1, n_coeff, cos(theta), type = 2)
        
        #prefactors = []
        
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
            
            # print ("iterazione:")
            # print(n)
            
            # print("tau:")
            # print(tau)
            
            # print("pi:")
            # print(pi)
            
            Ep_ = cos(phi)*self.Es+sin(phi)*self.Ep
            Es_ = -sin(phi)*self.Es+cos(phi)*self.Ep
            
            #Hs_ = np.sqrt(self.eps2/self.mu2)*(-cos(phi)*self.Ep+sin(phi)*self.Es)
            #Hp_ = np.sqrt(self.eps2/self.mu2)*(sin(phi)*self.Ep+cos(phi)*self.Es)  
            
            E_theta = E_theta + Ep_*(1/(self.k*r))*prefactor*(1j*self.an[n]*xi_der*tau-self.bn[n]*xi*pi_)
            E_phi = E_phi - Es_*(1/(self.k*r))*prefactor*(self.bn[n]*xi*tau-1j*self.an[n]*xi_der*pi_)
            E_r = E_r + Ep_*sin(theta)*(1/(self.k*r)**2)*prefactor*(m*(m+1))*1j*self.an[n]*xi*pi_
            test = test+prefactor*(m*(m+1))*1j*self.an[n]*xi*pi_
            
            H_theta = H_theta - Es_*(1/(self.k*r))*np.sqrt(self.eps2/self.mu2)*prefactor*(1j*self.bn[n]*xi_der*tau-self.an[n]*xi*pi_)
            H_phi = H_phi + Ep_*(1/(self.k*r))*np.sqrt(self.eps2/self.mu2)*prefactor*(1j*self.bn[n]*xi_der*pi_-self.an[n]*xi*tau)
            H_r = H_r - Es_*sin(theta)*(1/(self.k*r)**2)*np.sqrt(self.eps2/self.mu2)*prefactor*(m*(m+1))*1j*self.bn[n]*xi*pi_            
            '''if a ==True:
                #print(xi)
                print('ciao = ',self.an[n]*xi*pi_)'''
        H_theta = H_theta#*np.sqrt(eps0/mu0)  
        H_phi = H_phi#*np.sqrt(eps0/mu0)  
        H_r = H_r#*np.sqrt(eps0/mu0)  
        '''  
        if E_theta > 0.5:         
            print('scattered:', str(theta), str(phi))
            print(E_theta)
            print(E_phi)
            print(E_r)
            print()
            print('jn = ', Bessel_jn(1, self.k*r, derivative = False))
            print('tau = ', tau)
            print('pi = ', pi_)
            print('r = ', r)
            print('m = ',m)
            print()     '''
        '''if a ==True: 
            print('!!!!!!')
            print(E_theta)
            print(E_phi)        
            print(E_r)
            print(sin(theta))'''
            
        return E_theta, E_phi, E_r, H_theta, H_phi, H_r
        
    def complexAngleRotation(self, Ex, Ey, Ez, gamma):
        '''
        Ex_ = Ex*cos(gamma)-Ez*sin(gamma)
        Ey_ = Ey
        Ez_ = Ex*sin(gamma)+Ez*cos(gamma)
        '''
        Ey_ = Ey*cos(gamma)-Ez*sin(gamma)
        Ex_ = Ex
        Ez_ = Ey*sin(gamma)+Ez*cos(gamma)
        
        #print('cos gamma = ', cos(gamma))
        #print('sinh = ', 1j*sinh(self.alpha))
        #print('ez = ', Ez)
        return Ex_, Ey_, Ez_
    
    def IncidentField(self, theta, phi, r):
        
        
        
        x,y,z = fromPolToCart(theta, phi, r)
        #x = 1e-4
        #y = 1e-4
        #z = 2e-4
        Ex = self.Es*exp(1j*self.k*z)
        Ey = self.Ep*exp(1j*self.k*z)
        Ez = 0
        
        Hx = -np.sqrt(self.eps2/self.mu2)*self.Ep*exp(1j*self.k*z)#*np.sqrt(eps0/mu0)
        Hy = np.sqrt(self.eps2/self.mu2)*self.Es*exp(1j*self.k*z)#*np.sqrt(eps0/mu0)
        Hz = 0
        
        E_theta, E_phi, E_r = fromCartToPolField(Ex, Ey, Ez, x ,y, z)
        H_theta, H_phi, H_r = fromCartToPolField(Hx, Hy, Hz, x ,y, z)

        Ex_,Ey_,Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma))
        #print('incident:')
        #print('Ey inc = ', Ey_)
        #print('Ez inc = ', Ez_)
        #print(self.Ep)
        #print(' complex rot:')

        
        #print(E_theta)
        #print(E_phi)
        #print(E_r)
        #print()
        
        return E_theta, E_phi, E_r, H_theta, H_phi, H_r#Ex, Ey, Ez, Hx, Hy, Hz#
    
    def MaxwellTensorDotR(self, theta, phi, r, a):
        
        
        
        x,y,z = fromPolToCart(theta, phi, r)
        '''x = 1e-5
        y = 3e-5
        z = 2e-5'''
        x_,y_,z_ = self.complexAngleRotation(x,y,z, np.conj(self.gamma))
        #print('x =',x_)
        #print('y =',y_)
        #print('z =',z_)        
        theta_, phi_, r_ = fromCartToPol(x_,y_,z_)
        #print(sin(theta_))
        #print(sin(phi_))
        E_theta_i, E_phi_i, E_r_i, H_theta_i, H_phi_i, H_r_i = self.IncidentField(theta_, phi_, r_)
        if a == True:
            E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_,True)
        else:
            E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_,False)
        
        #E_theta_s =1
        #E_phi_s = 2
        #E_r_s = 3
        #Ex1,Ey1,Ez1 = fromPolToCartField(E_theta_s, E_phi_s, E_r_s, theta_, phi_) #controlla
        #Ex_1, Ey_1, Ez_1 = self.complexAngleRotation(Ex1,Ey1,Ez1,-np.conj(self.gamma))        

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
        '''print('Exs =',Ex_)'''
        #print('Eys =',Ey_)
        #print('Ezs =',Ez_)        
        #Ex1_ = self.Ep*exp(1j*self.k*x*cosh(self.alpha)-self.k*z*sinh(self.alpha))*1j*sinh(self.alpha)
        #Ez1_ = self.Ep*exp(1j*self.k*x*cosh(self.alpha)-self.k*z*sinh(self.alpha))*(-cosh(self.alpha))
        #print('Ex1 = ', Ez1_)
        #print('Ex = ', Ez_1)
        #print()
        #print(x*cosh(self.alpha)+1j*z*sinh(self.alpha))
        #print(z_)
        #print(np.conj(self.gamma))
        #print()
        
        
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
        
        # E_theta, E_phi, E_r = fromCartToPolField(Ex_, Ey_, Ez_, x, y, z) #controlla
        # H_theta, H_phi, H_r = fromCartToPolField(Hx_, Hy_, Hz_, x, y, z) #controlla
        
        # T_dot_r_theta = (1/(8*pi))*np.real(self.eps2*(np.conj(E_theta))*E_r+ self.mu2*(np.conj(H_theta))*H_r)
        # T_dot_r_phi =(1/(8*pi))*np.real(self.eps2*(np.conj(E_phi))*E_r+ self.mu2*(np.conj(H_phi))*H_r)
        # T_dot_r_r = (1/(8*pi))*np.real(self.eps2*np.conj(E_r)*E_r+self.mu2*(np.conj(H_r)*H_r)-1/2*(self.eps2*E2+self.mu2*H2))
        '''
        Ex_ = np.real(Ex_)
        Ey_ = np.real(Ey_)
        Ez_ = np.real(Ez_)
        
        Bx_ = mu0*np.real(Hx_)
        By_ = mu0*np.real(Hy_)
        Bz_ = mu0*np.real(Hz_)
        
        B2 = 0.5*(abs(Bx_)**2+abs(By_)**2+abs(Bz_)**2)        
        E2 = 0.5*(abs(Ex_)**2+abs(Ey_)**2+abs(Ez_)**2)
        
        T_xx = 0.5*eps0*(Ex_*Ex_-E2)+(1/mu0)*(Bx_*Bx_-B2)
        T_yy = 0.5*eps0*(Ey_*Ey_-E2)+(1/mu0)*(By_*By_-B2)
        T_zz = 0.5*eps0*(Ez_*Ez_-E2)+(1/mu0)*(Bz_*Bz_-B2)
        T_xy = 0.5*eps0*(Ex_*Ey_)+(1/mu0)*(Bx_*By_)
        T_yx = 0.5*eps0*(Ey_*Ex_)+(1/mu0)*(By_*Bx_)
        T_xz = 0.5*eps0*(Ex_*Ez_)+(1/mu0)*(Bx_*Bz_)
        T_zx = 0.5*eps0*(Ez_*Ex_)+(1/mu0)*(Bz_*Bx_)
        T_zy = 0.5*eps0*(Ez_*Ey_)+(1/mu0)*(Bz_*By_)
        T_yz = 0.5*eps0*(Ey_*Ez_)+(1/mu0)*(By_*Bz_)'''
        '''
        print(' E_theta_i = ',E_theta_i)
        print(' E_phi_i = ',E_phi_i)
        print(' E_r_i = ',E_r_i)
        print(' E_theta_s = ',E_theta_s)
        print(' E_phi_s = ',E_phi_s)
        print(' E_r_s = ',E_r_s)
        print()-
        '''
        '''print('T_xx = ',T_xx)
        print('T_yx = ',T_yx)
        print('T_zx = ',T_zx)
        print('T_xy = ',T_xy)
        print('T_yy = ',T_yy)
        print('T_zy = ',T_zy)
        print('T_xz = ',T_xz)
        print('T_yz = ',T_yz)
        print('T_zz = ',T_zz)
        print()'''
        
        #print(' Hxs = ',Hx_)
        #print(' Ey = ',Ey_)
        #print(' Ez = ',Ez_)
        #print(' Exs = ',Ex1)
        #print(' Eys = ',Ey1)
        #print(' Ezs = ',Ez1)
        
        return T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz

    def IntegrateOnSphere(self):
        raggio = self.raggio#+1e-9;
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
        x = 1e-5
        y = 3e-5
        z = 2e-5
        t,p,r = fromCartToPol(x, y, z)
        t = 2
        p=0.8
        r = self.raggio
        #print(sin(p))
        T_xx,T_xy,T_xz,T_yx,T_yy,T_yz,T_zx,T_zy,T_zz = self.MaxwellTensorDotR(t,p,r,True);
        dF = T_zx*(sin(t)*cos(p)) + T_zy*(sin(t)*sin(p)) + T_zz*cos(t);
        '''print('dF = ',dF*sin(t)*8*pi)
        print('T_xx = ',T_xx*8*pi)
        print('T_yx = ',T_yx*8*pi)
        print('T_zx = ',T_zx*8*pi)
        print('T_xy = ',T_xy*8*pi)
        print('T_yy = ',T_yy*8*pi)
        print('T_zy = ',T_zy*8*pi)
        print('T_xz = ',T_xz*8*pi)
        print('T_yz = ',T_yz*8*pi)     
        print('T_zz = ',T_zz*8*pi)'''     
        
        F_y = integrate.nquad(dFy, [[0,pi],[0,2*pi]]);
        
        F_z = integrate.nquad(dFz, [[0,pi],[0,2*pi]]);
        #et,ep,er,ht,hp,hr = self.ScatteredField(t,p,r,True)
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
            '''if dF > 0.1:
                #print(dF)
                
                x,y,z = fromPolToCart(theta, phi, raggio)
                x_,y_,z_ = self.complexAngleRotation(x,y,z, np.conj(self.gamma))
                
                theta_, phi_, r_ = fromCartToPol(x_,y_,z_)

                E_theta_i, E_phi_i, E_r_i, H_theta_i, H_phi_i, H_r_i = self.IncidentField(theta_, phi_, r_)
                E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_)
                
                E_theta = E_theta_i+E_theta_s
                E_phi = E_phi_i+E_phi_s
                E_r = E_r_i+E_r_s
                
                H_theta = H_theta_i+H_theta_s
                H_phi = H_phi_i+H_phi_s
                H_r = H_r_i+H_r_s
                
                
                Ex,Ey,Ez = fromPolToCartField(E_theta, E_phi, E_r, theta_, phi_) #controlla
                Ex_, Ey_, Ez_ = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma)) 
                
                Hx,Hy,Hz = fromPolToCartField(H_theta, H_phi, H_r, theta_, phi_) #controlla
                Hx_, Hy_, Hz_ = self.complexAngleRotation(Hx,Hy,Hz,-np.conj(self.gamma)) 
                
                a = 1
                b = 2
                c = 3
                
                a1,b1,c1 = self.complexAngleRotation(a,b,c,-np.conj(self.gamma))
                a2,b2,c2 = self.complexAngleRotation(a1,b1,c1,np.conj(self.gamma))
                
                print('a = ',a2)
                print('b = ',b2)
                print('c = ',c2)
                
                H2 = abs(Hx_)**2+abs(Hy_)**2+abs(Hz_)**2        
                E2 = abs(Ex_)**2+abs(Ey_)**2+abs(Ez_)**2  
                
                g = 0.5;#1/(8*pi)#
                
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
                
                Ex_ = np.real(Ex_)
                Ey_ = np.real(Ey_)
                Ez_ = np.real(Ez_)
                
                Bx_ = mu0*np.real(Hx_)
                By_ = mu0*np.real(Hy_)
                Bz_ = mu0*np.real(Hz_)                

                print('theta = ', theta)
                print('phi = ', phi)
                E_theta_s, E_phi_s, E_r_s, H_theta_s, H_phi_s, H_r_s = self.ScatteredField(theta_, phi_, r_)
                Ex,Ey,Ez = fromPolToCartField(E_theta_s, E_phi_s, E_r_s, theta_, phi_) #controlla
                Ex1, Ey1, Ez1 = self.complexAngleRotation(Ex,Ey,Ez,-np.conj(self.gamma)) 
                print(' Ex = ',Ex_)
                print(' Ey = ',Ey_)
                print(' Ez = ',Ez_)
                print(' Exs = ',Ex1)
                print(' Eys = ',Ey1)
                print(' Ezs = ',Ez1)'''        
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