#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:21:54 2020

@author: hanrach
"""

from jax import jacfwd, jit, jacrev
from residual import ResidualFunction
from res_fn_fast import ResidualFunctionFast
import jax.numpy as np
from unpack import unpack, unpack_fast

def p2d_init_slow(Np, Nn, Mp, Mn, Ms, Ma,Mz):
    solver = ResidualFunction(Np, Nn, Mp, Mn, Ms, Ma,Mz)

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

def p2d_init_fast(Np, Nn, Mp, Mn, Ms, Ma,Mz,delta_t):
    solver_fast = ResidualFunctionFast(Mp, Np, Mn, Nn, Ms, Ma, Mz, delta_t)
   
    def fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n):
    #    val = np.zeros(Ntot)
        val= np.zeros(solver_fast.Ntot)
#        U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n
#        U = arg0["U"]; Uold=arg0["Uold"]; cs_pe1=arg0["cs_pe1"]; cs_ne1=arg0["cs_ne1"]; gamma_p=arg0["gamma_p"]; gamma_n=arg0["gamma_n"]       
#        gamma_p = gamma_p*np.ones(Mp)
#        gamma_n = gamma_n*np.ones(Mn)
        uvec_pe, uvec_sep, uvec_ne, \
        Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
        phie_pe, phie_sep, phie_ne, \
        phis_pe, phis_ne, jvec_pe,jvec_ne,eta_pe,eta_ne = unpack_fast(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    
        
        uvec_old_pe, uvec_old_sep, uvec_old_ne,\
        Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc,\
        _, _, \
        _, _,\
        _,_,\
        _,_,_= unpack_fast(Uold,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        
        
        ''' add direct solve for c'''
    
       
        val = solver_fast.res_u_pe(val, uvec_pe, Tvec_pe, jvec_pe, uvec_old_pe, uvec_sep, Tvec_sep)
        val = solver_fast.res_u_sep(val, uvec_sep, Tvec_sep, uvec_old_sep, uvec_pe, Tvec_pe, uvec_ne, Tvec_ne)
        val = solver_fast.res_u_ne(val, uvec_ne, Tvec_ne, jvec_ne, uvec_old_ne, uvec_sep, Tvec_sep)
        
        val = solver_fast.res_T_acc(val, Tvec_acc, Tvec_old_acc, Tvec_pe)
        val = solver_fast.res_T_pe_fast(val, Tvec_pe, uvec_pe, phie_pe, phis_pe, jvec_pe, eta_pe, cs_pe1, gamma_p, Tvec_old_pe, Tvec_acc, Tvec_sep)
        val = solver_fast.res_T_sep(val, Tvec_sep, uvec_sep, phie_sep, Tvec_old_sep, Tvec_pe, Tvec_ne )
        val = solver_fast.res_T_ne_fast(val, Tvec_ne, uvec_ne, phie_ne, phis_ne, jvec_ne, eta_ne, cs_ne1, gamma_n, Tvec_old_ne, Tvec_zcc, Tvec_sep)
        val = solver_fast.res_T_zcc(val, Tvec_zcc, Tvec_old_zcc, Tvec_ne)
        
        val = solver_fast.res_phie_pe(val, uvec_pe, phie_pe, Tvec_pe, jvec_pe, uvec_sep,phie_sep, Tvec_sep)
        val = solver_fast.res_phie_sep(val, uvec_sep, phie_sep, Tvec_sep, phie_pe, phie_ne)
        val = solver_fast.res_phie_ne(val, uvec_ne, phie_ne, Tvec_ne, jvec_ne, uvec_sep, phie_sep, Tvec_sep)
    
        val = solver_fast.res_phis(val, phis_pe, jvec_pe, phis_ne, jvec_ne)
    
        val = solver_fast.res_j_fast(val, jvec_pe, uvec_pe, Tvec_pe, eta_pe, cs_pe1, gamma_p, jvec_ne, uvec_ne, Tvec_ne, eta_ne, cs_ne1, gamma_n)
        val = solver_fast.res_eta_fast(val, eta_pe, phis_pe, phie_pe, Tvec_pe, jvec_pe, cs_pe1, gamma_p, eta_ne, phis_ne, phie_ne, Tvec_ne, jvec_ne, cs_ne1, gamma_n)
        
        return val
    
    fn_fast=jit(fn_fast)
    jac_fn_fast=jit(jacfwd(fn_fast))
        
    return fn_fast, jac_fn_fast
    
    