#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:05:04 2020

@author: hanrach
"""


from dataclasses import dataclass

global trans
global F
global R
global gamma
global Tref
global Iapp

Iapp = -30;
trans = 0.364;
F = 96485;
R = 8.314472;
gamma = 2*(1-trans)*R/F;
Tref = 298.15;
delta_t = 10;
h = 1

@dataclass
class sep_grid_param_pack:
    M: int;
    
@dataclass
class separator_constants:    
    rho: float; Cp:float;
    eps: float; lam: float; 
    brugg: float; 
    l:float; 
    
def sep_constants():
    rho = 1100;
    Cp = 700;
    eps = 0.724
    lam = 0.16;
    brugg = 4;
    l = 2.5*1e-5;
#    l = 8*1e-5;
    return separator_constants(rho, Cp, eps, lam, brugg, l)

def sep_grid_param(M):
    
    return sep_grid_param_pack(M)


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
    l = 8*1e-5;
    
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
    
    # Solid-phase conductivity
    sigma = 100;
    
    Ek = 5000;
    
    ED = 5000;

    return electrode_constants(eps,brugg,a, Rp,lam, epsf, \
                               rho,Cp, k, Ds, l,sigma, Ek, ED)


@dataclass
class elec_grid_param_pack:
    M: int; N:int
    
def p_electrode_grid_param(M, N):
#    M = 10; N = 5;
    return elec_grid_param_pack(M, N)

def n_electrode_grid_param(M, N):
#    M = 10; N = 5;
    return elec_grid_param_pack(M, N)

@dataclass
class current_collector_constants:    
    lam: float; rho:float;
    Cp: float; sigma: float; 
    l:float
    
@dataclass
class cc_grid_param_pack:
    M: int
    
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

def cc_grid_param(M):
    
    return cc_grid_param_pack(M)