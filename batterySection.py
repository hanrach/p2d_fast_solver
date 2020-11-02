from dataclasses import dataclass
import coeffs
import jax
import jax.numpy as np
#import governing_eqns2 as gov
import coeffs as coeffs
from settings import *
#import test_jacres
from jax import jit, vmap, grad
from scipy.sparse.linalg import spsolve
from settings import Iapp, Tref, R, F


class Electrode:
    trans = 0.364;
    F = 96485;
    R = 8.314472;
    gamma = 2*(1-trans)*R/F;
    Tref = 200;
    
    
    def __init__(self, constants, gridparam, cavg, cmax):
        self.rho = constants.rho;
        self.Cp = constants.Cp;
        self.sigma = constants.sigma
        self.epsf = constants.epsf
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.a = constants.a
        self.Rp = constants.Rp
        self.k = constants.k
        self.l = constants.l
        self.Ds = constants.Ds;
        self.sigma = constants.sigma
        self.Ek = constants.Ek
        self.ED = constants.ED
        self.N = gridparam.N; self.M = gridparam.M
        N = self.N; M = self.M;
        
        self.hx = self.l/M; self.hy = self.Rp/N
        self.cmax = cmax
        self.cavg = cavg

        self.cmat = cavg*np.ones([N+2,M]);
        self.cmat_old = cavg*np.ones([N+2,M]);
        self.cs = np.zeros([M,1])+cavg;
        self.uvec = np.zeros([M+2,1])+1000;
#        self.uvec = np.reshape(np.linspace(950,1000,M+2), [M+2,1])
#        self.uvec_old = np.reshape(np.linspace(940,990,M+2), [M+2,1])
        self.uvec_old = np.zeros([M+2,1])+1000;
        self.phievec = np.zeros([M+2,1]) 
        
        self.Tvec = np.zeros([M+2,1]) + Tref;
        self.phisvec = np.zeros([M+2,1]) + self.open_circuit_poten(self.cavg, self.cavg,Tref,self.cmax);
        self.Told = np.zeros([M+2,1]) + Tref;
#        self.jvec = np.zeros([M,1]) + Iapp;
        
#        self.etavec= np.zeros([M,1])
#        self.jvec = np.zeros([M,1])
#        self.etavec = vmap(self.over_poten)(self.phisvec[1:M+1],self.phievec[1:M+1], self.Tvec[1:M+1],  self.cmat[self.N,:], self.cmat[self.N+1,:],self.cmax*np.ones([M,1]))
        self.etavec = np.zeros(self.M) + self.open_circuit_poten(self.cavg, self.cavg,Tref,self.cmax);
        self.jvec = vmap(self.ionic_flux)(self.uvec[1:M+1], self.Tvec[1:M+1], self.etavec[0:M], self.cs[0:M], self.cmax*np.ones([M,1]))
        
    def ionic_flux(self,u,T,eta,cs,cmax):
        keff = self.k*np.exp( (-self.Ek/R)*((1/T) - (1/Tref)))
        ans = 2*keff*np.sqrt(u*(cmax - cs)*cs)*np.sinh((0.5*R/(F*T))*(eta) )
        return ans.reshape()
    
    def open_circ_poten_ref(self,cM, cMp, cmax):
        cs = (cM + cMp)/2
        theta = cs/cmax;
        if (self.cavg ==25751):
            ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
            (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
        else:
            ans = 0.7222 + 0.1387*theta + 0.029*theta**(0.5) - 0.0172/theta + 0.0019/(theta**1.5) + 0.2808*np.exp(0.9 - 15*theta) - 0.7984*np.exp(0.4465*theta - 0.4108)
        return ans
    
    
    def open_circuit_poten(self,cM, cMp,T,cmax):
     
        Uref = self.open_circ_poten_ref(cM,cMp,cmax)
        ans = Uref + (T - Tref)*self.entropy_change(cM,cMp,cmax)
        return ans
    

    def entropy_change(self,cM, cMp,cmax):
        cs = (cM+cMp)/2
        theta = cs/cmax
        if (self.cavg == 25751 ):
            ans = -0.001*( (0.199521039 - 0.92837822*theta + 1.364550689*theta**2 - 0.6115448939999998*theta**3)/\
            (1 - 5.661479886999997*theta + 11.47636191*theta**2 - 9.82431213599998*theta**3 + \
             3.046755063*theta**4))
        else:
            ans = 0.001*(0.005269056 + 3.299265709*theta - 91.79325798*theta**2 + \
                         1004.911008*theta**3 - 5812.278127*theta**4 + \
                         19329.7549*theta**5 - 37147.8947*theta**6 + 38379.18127*theta**7 - \
                         16515.05308*theta**8)/(1 - 48.09287227*theta + 1017.234804*theta**2 - 10481.80419*theta**3 +\
                                             59431.3*theta**4 - 195881.6488*theta**5 + 374577.3152*theta**6 -\
                                             385821.1607*theta**7 + 165705.8597*theta**8)
        return ans
   
    def over_poten(self, phis,phie, T,  cM, cMp, cmax):
        ans = phis - phie - self.open_circuit_poten(cM, cMp,T,cmax);
        return ans.reshape()
    
    
        
class Separator:
    
    def __init__(self, constants, gridparam):
        self.rho = constants.rho;
        self.Cp = constants.Cp;
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.l = constants.l
        
        self.N = gridparam.N; self.M = gridparam.M
        N = self.N; M = self.M;
        self.hx = self.l/M; self.hy = self.l/N
#        self.hx = 1/M; self.hy = 1/N
        
        self.uvec = np.zeros([M+2,1])+1000;
        self.uvec_old = np.zeros([M+2,1])+1000;
        self.phievec = np.zeros([M+2,1]) + 0;
        self.Tvec = np.zeros([M+2,1]) + Tref;
        self.Told = np.zeros([M+2,1]) + Tref;

class CurrentCollector:
    # heat exchange coefficient
    h = 1
    def __init__(self, constants, gridparam):
        
        self.lam = constants.lam
        self.rho = constants.rho
        self.Cp = constants.Cp
        self.sigeff = constants.sigma
        self.l = constants.l
        self.N = gridparam.N; self.M = gridparam.M
        N = self.N; M = self.M;
        self.hx = self.l/M; self.hy = self.l/N
#        self.hx = 1/M; self.hy = 1/N
        # constants

        

        self.Tvec = np.zeros([M+2,1]) + Tref;
        self.Told = np.zeros([M+2,1]) + Tref;
        

@dataclass
class electrode_constants:
    eps: float; brugg:float;
    a: float; Rp: float;
    lam: float; epsf: float;
    rho: float; Cp: float;
    k:float
    Ds: float; l:float
    sigma: float; Ek: float;
    ED: float;
    
    
@dataclass
class separator_constants:    
    rho: float; Cp:float;
    eps: float; lam: float; 
    brugg: float; 
    l:float; 

@dataclass
class current_collector_constants:    
    lam: float; rho:float;
    Cp: float; sigma: float; 
    l:float
    
@dataclass
class grid_param_pack:
    M: int; N:int;
    
def p_electrode_constants():
    # porosity
    eps = 0.385; 
    
    # Bruggeman's coefficient
    brugg = 4;
    
    # Particle surface area to volume
    a= 885000;
    
    # Particle radius
    Rp= 2*1e-6;
    
    # Thermal conductivity
    lam = 2.1; 
    
    # Filler fraction
    epsf = 0.025;
    
    # Density
    rho = 2500; 
    
    # Specific heat
    Cp = 700;
    
    # Reaction rate
    k = 2.334*1e-11
    
    # Solid-phase diffusivity
    Ds = 1e-14; 
    
    # Thickness
    l = 8.0*1e-5;
#    l = 1;
    
    # Solid-phase conductivity
    sigma = 100;
    
    Ek = 5000
    
    ED = 5000
    
    return electrode_constants(eps,brugg,a, Rp,lam, epsf, \
                               rho,Cp, k, Ds, l,sigma,Ek, ED)
    
def n_electrode_constants():
    # porosity
    eps = 0.485; 
    
    # Bruggeman's coefficient
    brugg = 4;
    
    # Particle surface area to volume
    a= 723600;
    
    # Particle radius
    Rp= 2*1e-6;
    
    # Thermal conductivity
    lam = 1.7; 
    
    # Filler fraction
    epsf = 0.0326;
    
    # Density
    rho = 2500; 
    
    # Specific heat
    Cp = 700;
    
    # Reaction rate
    k = 5.031*1e-11
    
    # Solid-phase diffusivity
    Ds = 3.9*1e-14; 
    
    # Thickness
    l = 8.8*1e-5;
#    l = 8*1e-5;
#    l = 1;
    
    # Solid-phase conductivity
    sigma = 100;
    
    Ek = 5000;
    ED = 5000;
    return electrode_constants(eps,brugg,a, Rp,lam, epsf, \
                               rho,Cp, k, Ds, l,sigma, Ek, ED)

def sep_constants():

    
    rho = 1100;
    Cp = 700;
    eps = 0.724
    lam = 0.16;
    brugg = 4;
    l = 2.5*1e-5;
#    l = 8*1e-5;
#    l = 1;
    return separator_constants(rho, Cp, eps, lam, brugg, l)

def a_cc_constants():
    lam = 237; rho = 2700;
    Cp = 897;
    sigma = 3.55*1e7
    l = 1.0*1e-5
#    l = 8*1e-5;
    
    return current_collector_constants(lam, rho, Cp, sigma,l)

def z_cc_constants():
    lam = 401; rho = 8940;
    Cp = 385;
    sigma = 5.96*1e7
    l = 1.0*1e-5
#    l = 8*1e-5;
    
    return current_collector_constants(lam, rho, Cp, sigma,l)
    
def p_electrode_grid_param():
    M = 10; N = 5;

    return grid_param_pack(M,N)

def n_electrode_grid_param():
    M = 10; N = 5;

    return grid_param_pack(M,N)

def sep_grid_param():
    M = 10; N = 10;

    return grid_param_pack(M,N)

def cc_grid_param():
    M = 10; N = 10;

    return grid_param_pack(M,N)

pe = Electrode(p_electrode_constants(),p_electrode_grid_param(), 25751, 51554)
ne = Electrode(n_electrode_constants(),n_electrode_grid_param(), 26128, 30555)
sep = Separator(sep_constants(), sep_grid_param())
acc = CurrentCollector(a_cc_constants(),cc_grid_param())
zcc = CurrentCollector(z_cc_constants(),cc_grid_param())
