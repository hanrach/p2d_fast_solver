#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:49:44 2020

@author: hanrach
"""
import jax.numpy as np
from jax.config import config
config.update('jax_enable_x64', True)
from model.settings import Tref
from model.p2d_param import get_battery_sections
import timeit

from utils.unpack import unpack

from naive.p2d_newton import newton_short


def p2d_fn_short(Np, Nn, Mp, Mn, Ms, Ma,Mz, delta_t, fn, jac_fn, Iapp, Tf):
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Mn, Ms, Ma, Mz,delta_t, Iapp)

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
    
     

    steps = Tf/delta_t;

    voltages = [];
    temps = [];

    start_init = timeit.default_timer()
    Jinit = jac_fn(U, U).block_until_ready()
    end_init = timeit.default_timer()

    init_time = end_init - start_init
    solve_time_tot = 0
    jf_tot_time = 0
    overhead_time = 0

    start1 = timeit.default_timer()
    
    for i  in range(0,int(steps)):
    
        U, info = newton_short(fn, jac_fn, U)

        (fail, jf_time, overhead, solve_time) = info
        solve_time_tot += solve_time
        jf_tot_time += jf_time
        overhead_time += overhead

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

        
    end1 = timeit.default_timer();
    tot_time = end1-start1

    time = (tot_time, solve_time_tot, jf_tot_time, overhead_time, init_time)
    print("Done naive simulation\n")
    return U, voltages, temps, time