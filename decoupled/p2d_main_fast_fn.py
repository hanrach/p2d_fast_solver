#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:58:05 2020

@author: hanrach
"""

import jax
import jax.numpy as np
from jax import vmap
from jax.config import config
config.update('jax_enable_x64', True)
from model.settings import delta_t,Tref
import numpy as onp
import model.coeffs as coeffs
import timeit
from utils.unpack import unpack_vars
from scipy.sparse import csc_matrix
from scikits.umfpack import splu
from decoupled.p2d_newton_fast import newton_fast_sparse
image_folder = 'images'
video_name = 'video.avi'
from utils.precompute_c import precompute
from model.p2d_param import get_battery_sections




def p2d_fast_fn_short(Np, Nn, Mp, Mn, Ms, Ma,Mz, fn_fast, jac_fn, Iapp, Tf):
    start0 = timeit.default_timer()
    peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Mn, Ms, Ma, Mz, 10, Iapp)
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

    @jax.partial(jax.jit, static_argnums=(2, 3,))
    def combine_c(cII, cI_vec, M,N):
        return np.reshape(cII, [M * (N + 2)], order="F") + cI_vec
    
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


    steps = Tf/delta_t;
    voltages = [];
    temps = [];
    end0 = timeit.default_timer()
    
    print("setup time", end0-start0)

    cmat_rhs_pe = cmat_format_p(cmat_pe)
    cmat_rhs_ne = cmat_format_n(cmat_ne)
    lu_pe = lu["pe"];
    lu_ne = lu["ne"]

    cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
    cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))

    cs_pe1 = (cI_pe_vec[Np:Mp * (Np + 2):Np + 2] + cI_pe_vec[Np + 1:Mp * (Np + 2):Np + 2]) / 2
    cs_ne1 = (cI_ne_vec[Nn:Mn * (Nn + 2):Nn + 2] + cI_ne_vec[Nn + 1:Mn * (Nn + 2):Nn + 2]) / 2

    start_init = timeit.default_timer()
    Jinit = jac_fn(U_fast, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec).block_until_ready()
    end_init = timeit.default_timer()

    init_time = end_init - start_init

    solve_time_tot=0
    jf_tot_time=0
    overhead_time = 0


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

        cs_pe1 = (cI_pe_vec[Np:Mp*(Np+2):Np+2] + cI_pe_vec[Np+1:Mp*(Np+2):Np+2])/2
        cs_ne1 = (cI_ne_vec[Nn:Mn*(Nn+2):Nn+2] + cI_ne_vec[Nn+1:Mn*(Nn+2):Nn+2])/2
        
        U_fast, info = newton_fast_sparse(fn_fast, jac_fn, U_fast, cs_pe1, cs_ne1, gamma_p_vec,gamma_n_vec)

        (fail, solve_time, overhead, jf_time) = info

        overhead_time += overhead
        solve_time_tot += solve_time + c_lintime
        jf_tot_time += jf_time 
        
        # uvec_pe,uvec_sep, uvec_ne,Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc,\
        # phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        Tvec_pe, Tvec_ne, phis_pe, phis_ne, jvec_pe, jvec_ne = unpack_vars(U_fast, Mp, Mn, Ms, Ma)
      
        cII_p = form_c2_p_jit(temp_p, jvec_pe, Tvec_pe)
        cII_n = form_c2_n_jit(temp_n, jvec_ne, Tvec_ne)
        # cmat_pe = np.reshape(cII_p, [Mp*(Np+2)], order="F") + cI_pe_vec
        # cmat_ne = np.reshape(cII_n, [Mn*(Nn+2)], order="F") + cI_ne_vec
        cmat_pe = combine_c(cII_p, cI_pe_vec, Mp, Np)
        cmat_ne = combine_c(cII_n, cI_ne_vec, Mn, Nn)
    
        volt = phis_pe[1] - phis_ne[Mn]
        voltages.append(volt)
        temps.append(np.mean(Tvec_pe[1:Mp+1]))
        if (fail == 0):
            pass 
#            print("timestep:", i)
        else:
            print('Premature end of run\n') 
            print("timestep:",i)
            break 
        
    end1 = timeit.default_timer();
    tot_time = (end1-start1)
    time = (tot_time, solve_time_tot, jf_tot_time, overhead_time, init_time)
    print("Done decoupled simulation\n")
    return U_fast, cmat_pe, cmat_ne, voltages, temps,time



