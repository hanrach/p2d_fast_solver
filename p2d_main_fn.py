#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:49:44 2020

@author: hanrach
"""
import jax
import jax.numpy as np
from jax import jacfwd
from jax.config import config
config.update('jax_enable_x64', True)
from numpy.linalg import norm
from jax.scipy.linalg import solve
import matplotlib.pylab as plt
from settings import delta_t,Tref
from p2d_param import get_battery_sections
import timeit
#from res_fn_order2 import fn
from unpack import unpack, unpack_fast
from scipy.sparse import csr_matrix, csc_matrix
from  scipy.sparse.linalg import spsolve, splu
from p2d_newton import newton
from dataclasses import dataclass


def p2d_fn(Np, Nn, Mp, Mn, Ms, Ma,Mz, fn, jac_fn):
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz)
#    grid = pack_grid(Mp,Np,Mn,Nn,Ms,Ma,Mz)

    
    U = np.hstack( 
            [
                    peq.cavg*np.ones(Mp*(Np+2)), 
                    neq.cavg*np.ones(Mn*(Nn+2)),
                    1000 + np.zeros(Mp + 2),
                    1000 + np.zeros(Ms + 2),
                    1000 + np.zeros(Mn + 2),
                    
                    np.zeros(Mp),
                    np.zeros(Mn),
                    np.zeros(Mp),
                    np.zeros(Mn),
                    
                    
                    np.zeros(Mp+2) + peq.open_circuit_poten(peq.cavg, peq.cavg,Tref,peq.cmax),
                    np.zeros(Mn+2) + neq.open_circuit_poten(neq.cavg, neq.cavg,Tref,neq.cmax),
                    
                    np.zeros(Mp+2) + 0,
                    np.zeros(Ms+2) + 0,
                    np.zeros(Mn+2) + 0,
    
                    Tref + np.zeros(Ma + 2),
                    Tref + np.zeros(Mp + 2),
                    Tref + np.zeros(Ms + 2),
                    Tref + np.zeros(Mn + 2),
                    Tref + np.zeros(Mz + 2)
                    
                    ])
    
     
    Tf = 100; 
    steps = Tf/delta_t;
#    jac_fn = jax.jit(jacfwd(fn))
    voltages = [];
    temps = [];
    start = timeit.default_timer()
    for i  in range(0,int(steps)):
    
        [U, fail] = newton(fn, jac_fn, U)
    
        cmat_pe, cmat_ne,uvec_pe, uvec_sep, uvec_ne, \
        Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
        phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack(U,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        volt = phis_pe[1] - phis_ne[Mn]
        voltages.append(volt)
        temps.append(np.mean(Tvec_pe[1:Mp+1]))
        if (fail == 0):
            pass 
    #        print("timestep:", i)
        else:
            print('Premature end of run\n') 
            break 
        
    end = timeit.default_timer();
    time = end-start
    return U, voltages, temps,time