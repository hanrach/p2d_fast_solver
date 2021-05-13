#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 15:20:04 2020

@author: hanrach
"""

from model.settings import Tref, h
from jax.config import config
config.update("jax_enable_x64", True)

class CurrentCollectorEquation:
    
    def __init__(self, constants, gridparam, delta_t, Iapp):
        self.lam = constants.lam
        self.rho = constants.rho
        self.Cp = constants.Cp
        self.sigeff = constants.sigma
        self.l = constants.l
        self.M = gridparam.M
        M = self.M;
        self.hx = self.l/M;
        self.delta_t=delta_t
#        self.hx = 1/M; self.hy = 1/N;
        self.Iapp = Iapp
    
    def temperature(self,Tn,Tc,Tp, Told):
        hx = self.hx
#        ans = (Tc - Told) -  ( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
#        + Iapp**2/self.sigeff )*(delta_t/(self.rho*self.Cp))
        ans = (Tc - Told) -  ( self.lam*( Tn - 2*Tc + Tp)/hx**2 + self.Iapp**2/self.sigeff )*(self.delta_t/(self.rho*self.Cp))
        return ans.reshape()
    
    """ boundary condition """
    def bc_temp_a(self,T0,T1):
        bc = -self.lam*(T1-T0)/self.hx - h*(Tref - (T1+T0)/2)
        return bc.reshape()
    
    def bc_temp_z(self,T0,T1):
        bc = -self.lam*(T1-T0)/self.hx - h*((T0+T1)/2 - Tref)
        return bc.reshape()
    


#accq = CurrentCollectorEquation(a_cc_constants(),cc_grid_param())
#zccq = CurrentCollectorEquation(z_cc_constants(),cc_grid_param())