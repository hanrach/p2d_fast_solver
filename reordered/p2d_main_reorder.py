#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:57:47 2020

@author: hanrach
"""

import jax
import jax.numpy as np
from jax import jacfwd, vmap
from jax.config import config
import time
config.update('jax_enable_x64', True)
from numpy.linalg import norm
from jax.scipy.linalg import solve
import matplotlib.pylab as plt
from model.settings import Tref
import numpy as onp
#from p2d_param import grid_point_setup
import model.coeffs as coeffs
import timeit
from time import perf_counter
#from res_fn_order import fn_fast
#from res_fn_order2 import fn
from utils.unpack import unpack_fast
from scipy.sparse import csc_matrix
from scikits.umfpack import splu
from p2d_newton_reorder import newton_reorder, newton_reorder_short
image_folder = 'images'
video_name = 'video.avi'
from utils.precompute_c import precompute
from dataclasses import dataclass
from model.p2d_param import get_battery_sections
from utils.reorder import reorder_tot




#@nb.jit(nopython=True)
#def to_numpy(array):
#    return onp.asarray(array)
#    

def reorder_newton_step(Np, Nn, Mp, Mn, Ms, Ma,Mz, fn_fast, jac_fn_fast, timer_mode):
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz)
    Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq,neq)
    gamma_p_vec  = gamma_p*np.ones(Mp)
    gamma_n_vec = gamma_n*np.ones(Mn)
    lu_p = splu(csc_matrix(Ap))
    lu_n = splu(csc_matrix(An))
    lu = {"pe":lu_p, "ne": lu_n}
    
    U_fast = np.hstack( 
            [
                
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

    cmat_pe = peq.cavg*np.ones(Mp*(Np+2))
    cmat_ne = neq.cavg*np.ones(Mn*(Nn+2))
    
    idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
    re_idx = np.argsort(idx_tot)
    temp_p_mat = np.tile(temp_p,(Mp,1)).transpose()
    temp_n_mat = np.tile(temp_n,(Mn,1)).transpose()
    cmat_rhs_pe = cmat_format(cmat_pe, Mp, Np)
    cmat_rhs_ne = cmat_format(cmat_ne,Mn, Nn)
    lu_pe = lu["pe"]; lu_ne = lu["ne"]
    start0=time.monotonic()
    cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
    cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
    end0=time.monotonic()
    cI_pe_reshape = np.reshape(cI_pe_vec, [Np+2, Mp], order="F"); cs_pe1 = np.asarray((cI_pe_reshape[Np,:] + cI_pe_reshape[Np+1,:])/2)
    cI_ne_reshape = np.reshape(cI_ne_vec, [Nn+2, Mn], order="F"); cs_ne1 = np.asarray((cI_ne_reshape[Nn,:] + cI_ne_reshape[Nn+1,:])/2)
    
    
#    start = timeit.default_timer()
    U_fast,fail, linsolve_time, eval_time, overhead_time = newton_reorder(fn_fast, jac_fn_fast, U_fast, cs_pe1, cs_ne1, gamma_p_vec,gamma_n_vec,idx_tot,re_idx, timer_mode)
    
    uvec_pe,uvec_sep, uvec_ne,Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc,\
        phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)
   
    cII_p =  form_c2_p( temp_p_mat, jvec_pe, Tvec_pe, Np, Mp, peq)
    cII_n = form_c2_p( temp_n_mat, jvec_ne, Tvec_ne, Nn, Mn,neq)
    cmat_pe = np.reshape(cI_pe_reshape + cII_p, [Mp*(Np+2)], order="F")
    cmat_ne = np.reshape(cI_ne_reshape + cII_n, [Mn*(Nn+2)], order="F")
#    time = end-start
    return U_fast ,fail, linsolve_time + (end0-start0), eval_time, overhead_time


def reorder_newton_step_short(Np, Nn, Mp, Mn, Ms, Ma,Mz, fn_fast, jac_fn_fast):
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz)
    Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq,neq)
    gamma_p_vec  = gamma_p*np.ones(Mp)
    gamma_n_vec = gamma_n*np.ones(Mn)
    lu_p = splu(csc_matrix(Ap))
    lu_n = splu(csc_matrix(An))
    lu = {"pe":lu_p, "ne": lu_n}
    
    U_fast = np.hstack( 
            [
                
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

    cmat_pe = peq.cavg*np.ones(Mp*(Np+2))
    cmat_ne = neq.cavg*np.ones(Mn*(Nn+2))
    idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
    re_idx = np.argsort(idx_tot)
   
    start= timeit.default_timer()
    temp_p_mat = np.tile(temp_p,(Mp,1)).transpose()
    temp_n_mat = np.tile(temp_n,(Mn,1)).transpose()
    cmat_rhs_pe = cmat_format(cmat_pe, Mp, Np)
    cmat_rhs_ne = cmat_format(cmat_ne,Mn, Nn)
    lu_pe = lu["pe"]; lu_ne = lu["ne"]
    
    cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
    cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
    
    cI_pe_reshape = np.reshape(cI_pe_vec, [Np+2, Mp], order="F"); cs_pe1 = np.asarray((cI_pe_reshape[Np,:] + cI_pe_reshape[Np+1,:])/2)
    cI_ne_reshape = np.reshape(cI_ne_vec, [Nn+2, Mn], order="F"); cs_ne1 = np.asarray((cI_ne_reshape[Nn,:] + cI_ne_reshape[Nn+1,:])/2)
    

    U_fast,fail = newton_reorder_short(fn_fast, jac_fn_fast, U_fast, cs_pe1, cs_ne1, gamma_p_vec,gamma_n_vec,idx_tot,re_idx)
    
    uvec_pe,uvec_sep, uvec_ne,Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc,\
        phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)
   
    cII_p =  form_c2_p( temp_p_mat, jvec_pe, Tvec_pe, Np, Mp, peq)
    cII_n = form_c2_p( temp_n_mat, jvec_ne, Tvec_ne, Nn, Mn,neq)
    cmat_pe = np.reshape(cI_pe_reshape + cII_p, [Mp*(Np+2)], order="F")
    cmat_ne = np.reshape(cI_ne_reshape + cII_n, [Mn*(Nn+2)], order="F")
    
    time=timeit.default_timer()-start
    return U_fast ,fail, time


def cmat_format(cmat, M, N):
    val = cmat
    for i in range(0,M):
        val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
        val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)
    return val

def form_c2_p( temp, j, T, N, M, elec):
    # temp is N+2 by M
    val = np.zeros([N+2,M])
    for i in range(0,M):
        Deff = coeffs.solidDiffCoeff(elec.Ds,elec.ED,T[i+1])
        val = jax.ops.index_update(val, jax.ops.index[:,i], -(j[i]*temp[:,i]/Deff))
    return val



def p2d_reorder_fn(Np, Nn, Mp, Mn, Ms, Ma,Mz, fn_fast, jac_fn, timer_mode):
  
    start0 = timeit.default_timer()
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz)
    Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq,neq)
    gamma_p_vec  = gamma_p*np.ones(Mp)
    gamma_n_vec = gamma_n*np.ones(Mn)
    lu_p = splu(csc_matrix(Ap))
    lu_n = splu(csc_matrix(An))
    
    @jax.jit
    def cmat_format_p(cmat):
        val=jax.ops.index_update(cmat, jax.ops.index[0:Mp*(Np+2):Np+2],0)
        val=jax.ops.index_update(val, jax.ops.index[Np+1:Mp*(Np+2):Np+2],0)
#        for i in range(0,M):
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)    
        return val
    
    @jax.jit
    def cmat_format_n(cmat):
        val=jax.ops.index_update(cmat, jax.ops.index[0:Mn*(Nn+2):Nn+2],0)
        val=jax.ops.index_update(val, jax.ops.index[Nn+1:Mn*(Nn+2):Nn+2],0)
#        for i in range(0,M):
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)    
        return val
    
    @jax.jit
    def form_c2_p_jit( temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(peq.Ds*np.ones(Mp), peq.ED*np.ones(Mp), T[1:Mp+1])
        fn=lambda j, temp, Deff: -(j*temp/Deff)
#        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val=vmap(fn,(0,None,0),1)(j,temp,Deff_vec)
      
        return val


    @jax.jit    
    def form_c2_n_jit(temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(neq.Ds*np.ones(Mn), neq.ED*np.ones(Mn), T[1:Mn+1])
        fn=lambda j, temp, Deff: -(j*temp/Deff)
#        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val=vmap(fn,(0,None,0),1)(j,temp,Deff_vec)
        return val
    
    U_fast = np.hstack( 
            [
                
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

    cmat_pe = peq.cavg*np.ones(Mp*(Np+2))
    cmat_ne = neq.cavg*np.ones(Mn*(Nn+2))
    
    lu = {"pe":lu_p, "ne": lu_n}


    idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
    re_idx = np.argsort(idx_tot)
    Tf = 3520; 
    steps = Tf/delta_t;
    voltages = [];
    temps = [];
    end0 = timeit.default_timer()
    
    print("setup time", end0-start0)
#    res_list=[]
    linsolve_t=[]
    eval_t = []
    sp_t=[]
    start1 = timeit.default_timer()
    for i  in range(0,int(steps)):
 
        cmat_rhs_pe = cmat_format_p(cmat_pe)
        cmat_rhs_ne = cmat_format_n(cmat_ne)
        lu_pe = lu["pe"]; lu_ne = lu["ne"]
        
        c_start = timeit.default_timer()
        cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
        cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
        c_end = timeit.default_timer()
        
     
        cs_pe1 = (cI_pe_vec[Np:Mp*(Np+2):Np+2] + cI_pe_vec[Np+1:Mp*(Np+2):Np+2])/2
        cs_ne1 = (cI_ne_vec[Nn:Mn*(Nn+2):Nn+2] + cI_ne_vec[Nn+1:Mn*(Nn+2):Nn+2])/2

        
        U_fast,fail, linsolve_time, eval_time, overhead_time = newton_reorder(fn_fast, jac_fn, U_fast, cs_pe1, cs_ne1, gamma_p_vec,gamma_n_vec,idx_tot,re_idx, timer_mode)
        
        uvec_pe,uvec_sep, uvec_ne,Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc,\
        phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        
        cII_p = form_c2_p_jit(temp_p, jvec_pe, Tvec_pe)
        cII_n = form_c2_n_jit(temp_n, jvec_ne, Tvec_ne)
        cmat_pe = np.reshape(cII_p, [Mp*(Np+2)], order="F") + cI_pe_vec
        cmat_ne = np.reshape(cII_n, [Mn*(Nn+2)], order="F") + cI_ne_vec
        
        linsolve_t.append(linsolve_time + (c_end - c_start))
        eval_t.append(eval_time)
        sp_t.append(overhead_time)
        
        volt = phis_pe[1] - phis_ne[Mn]
        voltages.append(volt)
        temps.append(np.mean(Tvec_pe[1:Mp+1]))
        if (fail == 0):
            pass 
#            print("timestep:", i)
        else:
            print('Premature end of run\n') 
            break 
        
    end1 = timeit.default_timer();
    time = end1-start1
    return U_fast, voltages, temps,time,linsolve_t, eval_t, sp_t

 
def p2d_reorder_fn_short(Np, Nn, Mp, Mn, Ms, Ma,Mz, fn_fast, jac_fn, delta_t, Tf):
    
    
    start0 = timeit.default_timer()
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t)
    Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq,neq)
    gamma_p_vec  = gamma_p*np.ones(Mp)
    gamma_n_vec = gamma_n*np.ones(Mn)
    lu_p = splu(csc_matrix(Ap))
    lu_n = splu(csc_matrix(An))
    
    @jax.jit
    def cmat_format_p(cmat):
        val=jax.ops.index_update(cmat, jax.ops.index[0:Mp*(Np+2):Np+2],0)
        val=jax.ops.index_update(val, jax.ops.index[Np+1:Mp*(Np+2):Np+2],0)
#        for i in range(0,M):
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)    
        return val
    
    @jax.jit
    def cmat_format_n(cmat):
        val=jax.ops.index_update(cmat, jax.ops.index[0:Mn*(Nn+2):Nn+2],0)
        val=jax.ops.index_update(val, jax.ops.index[Nn+1:Mn*(Nn+2):Nn+2],0)
#        for i in range(0,M):
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
#            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)    
        return val
    
    @jax.jit
    def form_c2_p_jit( temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(peq.Ds*np.ones(Mp), peq.ED*np.ones(Mp), T[1:Mp+1])
        fn=lambda j, temp, Deff: -(j*temp/Deff)
#        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val=vmap(fn,(0,None,0),1)(j,temp,Deff_vec)
      
        return val


    @jax.jit    
    def form_c2_n_jit(temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(neq.Ds*np.ones(Mn), neq.ED*np.ones(Mn), T[1:Mn+1])
        fn=lambda j, temp, Deff: -(j*temp/Deff)
#        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val=vmap(fn,(0,None,0),1)(j,temp,Deff_vec)
        return val
    
    U_fast = np.hstack( 
            [
                
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

    cmat_pe = peq.cavg*np.ones(Mp*(Np+2))
    cmat_ne = neq.cavg*np.ones(Mn*(Nn+2))
    
    lu = {"pe":lu_p, "ne": lu_n}

    idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
    re_idx = np.argsort(idx_tot)
#    Tf = 100; 
    steps = Tf/delta_t;
    voltages = [];
    temps = [];
    end0 = timeit.default_timer()
    
    print("setup time", end0-start0)
#    res_list=[]
    jf_tot_time=0
    solve_tot_time=0
    overhead_tot_time=0
    start1 = timeit.default_timer()
    for i  in range(0,int(steps)):

        cmat_rhs_pe = cmat_format_p(cmat_pe)
        cmat_rhs_ne = cmat_format_n(cmat_ne)
        lu_pe = lu["pe"]; lu_ne = lu["ne"]
        

        start=timeit.default_timer()
        cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
        cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
        end=timeit.default_timer()
        c_lintime=end-start
#        print("Solve for c time:", c_lintime);
        
        cs_pe1 = (cI_pe_vec[Np:Mp*(Np+2):Np+2] + cI_pe_vec[Np+1:Mp*(Np+2):Np+2])/2
        cs_ne1 = (cI_ne_vec[Nn:Mn*(Nn+2):Nn+2] + cI_ne_vec[Nn+1:Mn*(Nn+2):Nn+2])/2
        
        start=perf_counter()
        U_fast,info = newton_reorder_short(fn_fast, jac_fn, U_fast, cs_pe1, cs_ne1, gamma_p_vec,gamma_n_vec,idx_tot,re_idx)
        end=perf_counter()
        fail, jf_time, overhead, solve_time =info
        
        jf_tot_time += jf_time;
        overhead_tot_time += overhead
        solve_tot_time += solve_time +c_lintime
#        print("One newton's solve", end-start)
        
        uvec_pe,uvec_sep, uvec_ne,Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc,\
        phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        

        cII_p = form_c2_p_jit(temp_p, jvec_pe, Tvec_pe)
        cII_n = form_c2_n_jit(temp_n, jvec_ne, Tvec_ne)
        cmat_pe = np.reshape(cII_p, [Mp*(Np+2)], order="F") + cI_pe_vec
        cmat_ne = np.reshape(cII_n, [Mn*(Nn+2)], order="F") + cI_ne_vec
        
        volt = phis_pe[1] - phis_ne[Mn]
        voltages.append(volt)
        temps.append(np.mean(Tvec_pe[1:Mp+1]))
        if (fail == 0):
            pass 
#            print("timestep:", i)
        else:
            print('Premature end of run\n') 
            break 
        
    end1 = timeit.default_timer();
    time_final = end1-start1
    
    print("Total time to evualuate Jacobian and f:", jf_tot_time)
    time=(time_final, jf_tot_time, overhead_tot_time, solve_tot_time)
    return U_fast, cmat_pe, cmat_ne, voltages, temps,time
#    return U_fast, cmat_pe, cmat_ne, voltages, temps,time

