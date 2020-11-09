#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:57:11 2020

@author: hanrach
"""

import jax.numpy as np
from jax import vmap, grad
import jax
from jax.config import config
import timeit
#from batterySection import pe, sep, acc, zcc, ne
from scipy.sparse import coo_matrix, bmat, block_diag, hstack, vstack
config.update("jax_enable_x64", True)
import matplotlib.pylab as plt
from jacres_c import jacres_c
from jacres_u import jacres_u
from jacres_T import jacres_T
from jacres_phie import jacres_phie
from jacres_phis import jacres_phis
from jacres_j import jacres_j
from jacres_eta import jacres_eta
from batterySection import Electrode, Separator, CurrentCollector
from ElectrodeEquation import ElectrodeEquation, p_electrode_constants,p_electrode_grid_param, n_electrode_constants,n_electrode_grid_param
from SeparatorEquation import SeparatorEquation, sep_constants,sep_grid_param
from CurrentCollectorEquation import CurrentCollectorEquation, a_cc_constants, z_cc_constants, cc_grid_param

pe = Electrode(p_electrode_constants(),p_electrode_grid_param(), 25751, 51554)
ne = Electrode(n_electrode_constants(),n_electrode_grid_param(), 26128, 30555)
sep = Separator(sep_constants(), sep_grid_param())
acc = CurrentCollector(a_cc_constants(),cc_grid_param())
zcc = CurrentCollector(z_cc_constants(),cc_grid_param())

peq = ElectrodeEquation(p_electrode_constants(),p_electrode_grid_param(), "positive")
neq = ElectrodeEquation(n_electrode_constants(),n_electrode_grid_param(), "negative")



def jacres_tot(acc, pe, sep, ne, zcc):
    res_c, J_c = jacres_c(acc,pe,sep,ne,zcc);
    res_c = np.reshape(res_c, len(res_c))
    res_u, J_u = jacres_u(acc,pe,sep,ne,zcc)
    res_T, J_T = jacres_T(acc,pe,sep,ne,zcc)
    res_phie, J_phie = jacres_phie(acc,pe,sep,ne,zcc)
    res_phis, J_phis = jacres_phis(acc,pe,sep,ne,zcc)
    res_j, J_j = jacres_j(acc,pe,sep,ne,zcc)
    res_eta, J_eta = jacres_eta(acc,pe,sep,ne,zcc)
    
    res = np.hstack((res_c, res_u, res_T, res_phie, res_phis,
                     res_j, res_eta))
    J = vstack([
            J_c, 
            J_u, J_T, J_phie,
            J_phis, J_j, J_eta
            ])
    return res, J

res_c, J_c = jacres_c(acc,pe,sep,ne,zcc);
res_c = np.reshape(res_c, len(res_c))
res_u, J_u = jacres_u(acc,pe,sep,ne,zcc)
res_T, J_T = jacres_T(acc,pe,sep,ne,zcc)
res_phie, J_phie = jacres_phie(acc,pe,sep,ne,zcc)
res_phis, J_phis = jacres_phis(acc,pe,sep,ne,zcc)
res_j, J_j = jacres_j(acc,pe,sep,ne,zcc)
res_eta, J_eta = jacres_eta(acc,pe,sep,ne,zcc)

res = np.hstack((res_c, res_u, res_T, res_phie, res_phis,
                 res_j, res_eta))
J_manual = vstack([
        J_c, 
        J_u, J_T, J_phie,
        J_phis, J_j, J_eta
        ])