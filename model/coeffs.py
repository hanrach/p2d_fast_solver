#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:11:07 2020

@author: hanrach
"""
from settings import R, Tref
from jax.config import config
import jax.numpy as np
config.update("jax_enable_x64", True)

def electrolyteDiffCoeff(eps,brugg,u,T):
    D_eff = (eps**brugg)*1e-4*(10**(-4.43 - 54/(T - 229 - 5*1e-3*u) - 0.22*1e-3*u))
    return D_eff

def solidConductCoeff(sigma,eps,epsf):
    sig_eff = sigma*(1-eps-epsf)
    return sig_eff

def electrolyteConductCoeff(eps,brugg,u,T):
    kap_eff = (eps**brugg)*1e-4*u*(-10.5 + 0.668*1e-3*u + 0.494*1e-6*u**2 + \
                                 (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2)*T + \
                                 (-6.96*1e-5 + 2.8*1e-8*u)*T**2)**2
    return kap_eff

def electrolyteConductCoeff_delT(eps,brugg,u,T):
    kap_eff = 2*(eps**brugg)*1e-4*u*(-10.5 + 0.668*1e-3*u + 0.494*1e-6*u**2 + \
                                 (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2)*T + \
                                 (-6.96*1e-5 + 2.8*1e-8*u)*T**2)*(  (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2) + 2*(-6.96*1e-5 + 2.8*1e-8*u)*T) 
    return kap_eff

def solidDiffCoeff(Ds,EqD,T):
    Ds_eff = Ds*np.exp( -(EqD/R)*( (1/T) - (1/Tref) ) )
    return Ds_eff


