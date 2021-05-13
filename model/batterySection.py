from dataclasses import dataclass
import coeffs
import jax
import jax.numpy as np
#import governing_eqns2 as gov
import coeffs as coeffs
from model.settings import *
#import test_jacres
from jax import jit, vmap, grad
from scipy.sparse.linalg import spsolve
from settings import Iapp, Tref, R, F
        

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
    
