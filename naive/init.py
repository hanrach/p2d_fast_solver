#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:21:54 2020

@author: hanrach
"""

from jax import jacfwd, jit
from naive.residual import ResidualFunction
import jax.numpy as np
from utils.unpack import unpack

def p2d_init_slow(Np, Nn, Mp, Mn, Ms, Ma,Mz, delta_t, Iapp):
    solver = ResidualFunction(Np, Nn, Mp, Mn, Ms, Ma,Mz, delta_t, Iapp)

    def fn(U,Uold):
        
        val= np.zeros(solver.Ntot)
        Mp = solver.Mp; Np = solver.Np
        Mn = solver.Mn; Nn = solver.Nn; Ms = solver.Ms; Ma = solver.Ma; Mz=solver.Mz
        cmat_pe, cmat_ne,\
        uvec_pe, uvec_sep, uvec_ne, \
        Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
        phie_pe, phie_sep, phie_ne, \
        phis_pe, phis_ne, jvec_pe,jvec_ne,eta_pe,eta_ne = unpack(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
        
        cmat_old_pe, cmat_old_ne,\
        uvec_old_pe, uvec_old_sep, uvec_old_ne,\
        Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc,\
        _, _, \
        _, _,\
        _,_,\
        _,_,_= unpack(Uold,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        
        val = solver.res_c_fn(val, cmat_pe, jvec_pe, Tvec_pe, cmat_old_pe, cmat_ne, jvec_ne, Tvec_ne, cmat_old_ne )
           
        val = solver.res_u_pe(val, uvec_pe, Tvec_pe, jvec_pe, uvec_old_pe, uvec_sep, Tvec_sep)
        val = solver.res_u_sep(val, uvec_sep, Tvec_sep, uvec_old_sep, uvec_pe, Tvec_pe, uvec_ne, Tvec_ne)
        val = solver.res_u_ne(val, uvec_ne, Tvec_ne, jvec_ne, uvec_old_ne, uvec_sep, Tvec_sep)
        
        val = solver.res_T_acc(val, Tvec_acc, Tvec_old_acc, Tvec_pe)
        val = solver.res_T_pe(val, Tvec_pe, uvec_pe, phie_pe, phis_pe, jvec_pe, eta_pe, cmat_pe, Tvec_old_pe, Tvec_acc, Tvec_sep)
        val = solver.res_T_sep(val, Tvec_sep, uvec_sep, phie_sep, Tvec_old_sep, Tvec_pe, Tvec_ne )
        val = solver.res_T_ne(val, Tvec_ne, uvec_ne, phie_ne, phis_ne, jvec_ne, eta_ne, cmat_ne, Tvec_old_ne, Tvec_zcc, Tvec_sep)
        val = solver.res_T_zcc(val, Tvec_zcc, Tvec_old_zcc, Tvec_ne)
        
        val = solver.res_phie_pe(val, uvec_pe, phie_pe, Tvec_pe, jvec_pe, uvec_sep,phie_sep, Tvec_sep)
        #    val = res_phie_pe_phi(val, uvec_pe, phie_pe, Tvec_pe, phis_pe, uvec_sep,phie_sep, Tvec_sep)
        val = solver.res_phie_sep(val, uvec_sep, phie_sep, Tvec_sep, phie_pe, phie_ne)
        val = solver.res_phie_ne(val, uvec_ne, phie_ne, Tvec_ne, jvec_ne, uvec_sep, phie_sep, Tvec_sep)
        #    val = res_phie_ne_phi(val, uvec_ne, phie_ne, Tvec_ne, phis_ne, uvec_sep, phie_sep, Tvec_sep)
        
        val = solver.res_phis(val, phis_pe, jvec_pe, phis_ne, jvec_ne)
        
        val = solver.res_j(val, jvec_pe, uvec_pe, Tvec_pe, eta_pe, cmat_pe, jvec_ne, uvec_ne, Tvec_ne, eta_ne, cmat_ne)
        #    val = res_j_phi(val, jvec_pe, uvec_pe, Tvec_pe, phis_pe, phie_pe, cmat_pe, jvec_ne, uvec_ne, Tvec_ne, phis_ne, phie_ne, cmat_ne)
        val = solver.res_eta(val, eta_pe, phis_pe, phie_pe, Tvec_pe, cmat_pe, eta_ne, phis_ne, phie_ne, Tvec_ne, cmat_ne)
        return val
    fn = jit(fn)
    jac_fn = jit(jacfwd(fn))
#    fn = jit(fn)
    return fn, jac_fn
