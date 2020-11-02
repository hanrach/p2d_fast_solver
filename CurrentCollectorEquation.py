#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:20:04 2020

@author: hanrach
"""
from dataclasses import dataclass
import coeffs
import jax.numpy as np
from settings import F, R, gamma, trans, Tref, delta_t, Iapp, h
from jax.config import config
config.update("jax_enable_x64", True)

class CurrentCollectorEquation:
    
    def __init__(self, constants, gridparam):
        self.lam = constants.lam
        self.rho = constants.rho
        self.Cp = constants.Cp
        self.sigeff = constants.sigma
        self.l = constants.l
        self.M = gridparam.M
        M = self.M;
        self.hx = self.l/M;
#        self.hx = 1/M; self.hy = 1/N;
    
    def temperature(self,Tn,Tc,Tp, Told):
        hx = self.hx
#        ans = (Tc - Told) -  ( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
#        + Iapp**2/self.sigeff )*(delta_t/(self.rho*self.Cp))
        ans = (Tc - Told) -  ( self.lam*( Tn - 2*Tc + Tp)/hx**2 + Iapp**2/self.sigeff )*(delta_t/(self.rho*self.Cp))
        return ans.reshape()
    
    """ boundary condition """
    def bc_temp_a(self,T0,T1):
        bc = -self.lam*(T1-T0)/self.hx - h*(Tref - (T1+T0)/2)
        return bc.reshape()
    
    def bc_temp_z(self,T0,T1):
        bc = -self.lam*(T1-T0)/self.hx - h*((T0+T1)/2 - Tref)
        return bc.reshape()
    
@dataclass
class current_collector_constants:    
    lam: float; rho:float;
    Cp: float; sigma: float; 
    l:float
    
@dataclass
class grid_param_pack:
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
    
    return grid_param_pack(M)

#accq = CurrentCollectorEquation(a_cc_constants(),cc_grid_param())
#zccq = CurrentCollectorEquation(z_cc_constants(),cc_grid_param())