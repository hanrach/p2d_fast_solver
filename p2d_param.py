#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:07:37 2020

@author: hanrach
"""
from ElectrodeEquation import ElectrodeEquation, p_electrode_constants,p_electrode_grid_param, n_electrode_constants,n_electrode_grid_param
from SeparatorEquation import SeparatorEquation, sep_constants,sep_grid_param
from CurrentCollectorEquation import CurrentCollectorEquation, a_cc_constants, z_cc_constants, cc_grid_param

def get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz):
    peq = ElectrodeEquation(p_electrode_constants(),p_electrode_grid_param(Mp, Np), "positive", 25751,51554)
    neq = ElectrodeEquation(n_electrode_constants(),n_electrode_grid_param(Mn, Nn), "negative",26128,30555)
    sepq = SeparatorEquation(sep_constants(), sep_grid_param(Ms))
    accq = CurrentCollectorEquation(a_cc_constants(), cc_grid_param(Ma))
    zccq = CurrentCollectorEquation(z_cc_constants(), cc_grid_param(Mz))
    
    
    Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
    Np = peq.N; Nn = neq.N
    
    Ntot_pe = (Np+2)*(Mp) + 4*(Mp + 2) + 2*(Mp)
    Ntot_ne = (Nn+2)*(Mn) + 4*(Mn + 2) + 2*(Mn)
    
    Ntot_sep =  3*(Ms + 2)
    Ntot_acc =Ma+ 2
    Ntot_zcc = Mz+ 2
    Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
    return peq, neq, sepq, accq, zccq
