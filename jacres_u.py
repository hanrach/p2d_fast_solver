#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:28:52 2020

@author: hanrach
"""


import jax.numpy as np
from jax import vmap, grad
import jax
from jax.config import config
import timeit
#import heatsource as heat
#from batterySection import pe, sep, acc, zcc, ne
#from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, bmat, block_diag, vstack,hstack
config.update("jax_enable_x64", True)
from settings import Iapp, delta_t,F
from build_matrix import build_tridiag,build_diag, empty_square, empty_rec
import matplotlib.pyplot as plt
from unpack import unpack

def jacres_u(U, Uold, peq, neq, sepq, accq, zccq):
    Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
    Np = peq.N; Nn = neq.N
    
    cmat_pe, cmat_ne, \
    uvec_pe, uvec_sep, uvec_ne,\
    Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
    phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, \
    j_pe,j_ne,eta_pe,eta_ne = unpack(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    
    cmat_old_pe, cmat_old_ne,\
    uvec_old_pe, uvec_old_sep, uvec_old_ne,\
    Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc,\
    _, _, \
    _, _,\
    _,_,\
    _,_,_= unpack(Uold,Mp, Np, Mn, Nn, Ms, Ma, Mz)
    
    bc_u0p = peq.bc_zero_neumann(uvec_pe[0],uvec_pe[1])
    res_up = vmap(peq.electrolyte_conc)(uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], j_pe[0:Mp],uvec_old_pe[1:Mp+1])
    bc_uMp = peq.bc_u_sep_p(uvec_pe[Mp], uvec_pe[Mp+1], Tvec_pe[Mp], Tvec_pe[Mp+1], uvec_sep[0],uvec_sep[1],Tvec_sep[0],Tvec_sep[1])
    
    bc_u0s = peq.bc_inter_cont(uvec_pe[Mp], uvec_pe[Mp+1], uvec_sep[0],uvec_sep[1])
    res_us = vmap(sepq.electrolyte_conc)(uvec_sep[0:Ms], uvec_sep[1:Ms+1], uvec_sep[2:Ms+2], Tvec_sep[0:Ms], Tvec_sep[1:Ms+1], Tvec_sep[2:Ms+2], uvec_old_sep[1:Ms+1] )
    bc_uMs = sepq.bc_u_sep_n(uvec_ne[0], uvec_ne[1], Tvec_ne[0],Tvec_ne[1], uvec_sep[Ms], uvec_sep[Ms+1], Tvec_sep[Ms], Tvec_sep[Ms+1])
    
    bc_u0n = neq.bc_inter_cont(uvec_ne[0], uvec_ne[1], uvec_sep[Ms], uvec_sep[Ms+1])
    res_un = vmap(neq.electrolyte_conc)(uvec_ne[0:Mn], uvec_ne[1:Mn+1], uvec_ne[2:Mn+2], Tvec_ne[0:Mn], Tvec_ne[1:Mn+1], Tvec_ne[2:Mn+2], j_ne[0:Mn], uvec_old_ne[1:Mn+1])
    bc_uMn = neq.bc_zero_neumann(uvec_ne[Mn],uvec_ne[Mn+1])
    
    """ positive electrode"""
    arg_up = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], j_pe[0:Mp],uvec_old_pe[1:Mp+1]]
    arg_uMp = [uvec_pe[Mp], uvec_pe[Mp+1], Tvec_pe[Mp], Tvec_pe[Mp+1], uvec_sep[0],uvec_sep[1],Tvec_sep[0],Tvec_sep[1]]
    
    A_up = vmap( grad( peq.electrolyte_conc, range(0,len(arg_up)-1) ) )(*arg_up)
    bc_u0p_grad = np.array([[-1,1]]).T
    bc_uMp_grad =  ((jax.grad(peq.bc_u_sep_p,range(0,len(arg_uMp)))))(*arg_uMp)
    
    #uu
    bc_uup = {"right":bc_u0p_grad[0:2], "left":bc_uMp_grad[0:2]}
    J_uupp = build_tridiag(Mp, A_up[0:3], **bc_uup)
    # interface boundar8
    J_uups = coo_matrix((np.ravel(np.asarray(bc_uMp_grad[4:6])),([Mp+1,Mp+1],[0,1] )),shape=(Mp+2,Ms+2+Mn+2))
    J_uup = hstack([J_uupp,J_uups])
    
    # uT
    bc_uTp = {"left":bc_uMp_grad[2:4]}
    J_uTpp = build_tridiag(Mp,A_up[3:6], **bc_uTp)
    # interface
    J_uTps = coo_matrix((np.ravel(np.asarray(bc_uMp_grad[6:8])),([Mp+1,Mp+1],[0,1] )),shape=(Mp+2,Ms+2+Mn+2))
    J_uTp = hstack([J_uTpp, J_uTps])
    
    # uj
    J_ujp = build_diag(Mp,A_up[6], "long")
    
    """ Separator """
    arg_u0s = [uvec_pe[Mp], uvec_pe[Mp+1], uvec_sep[0],uvec_sep[1]]
    arg_us = [uvec_sep[0:Ms], uvec_sep[1:Ms+1], uvec_sep[2:Ms+2], Tvec_sep[0:Ms], Tvec_sep[1:Ms+1], Tvec_sep[2:Ms+2], uvec_old_sep[1:Ms+1] ]
    arg_uMs = [uvec_ne[0], uvec_ne[1], Tvec_ne[0],Tvec_ne[1], uvec_sep[Ms], uvec_sep[Ms+1], Tvec_sep[Ms], Tvec_sep[Ms+1]]
    
    A_us = vmap( grad( sepq.electrolyte_conc, range(0,len(arg_us)-1) ) )(*arg_us)
    bc_u0s_grad = ( jax.grad(peq.bc_inter_cont,range(0,len(arg_u0s))) )(*arg_u0s)
    bc_uMs_grad = ( jax.grad(sepq.bc_u_sep_n,range(0,len(arg_uMs)) ))(*arg_uMs)
    
    #uu
    bc_uus = {"right":bc_u0s_grad[2:4], "left":bc_uMs_grad[4:6]}
    J_uuss = build_tridiag(Ms, A_us[0:3], **bc_uus)
    # positive sep interface
    J_uusp = coo_matrix((np.ravel(np.asarray(bc_u0s_grad[0:2])),([0,0],[Mp,Mp+1] )),shape=(Ms+2,Mp+2))
    #negative sep interface
    J_uusn = coo_matrix((np.ravel(np.asarray(bc_uMs_grad[0:2])),([Ms+1,Ms+1],[0,1] )),shape=(Ms+2,Mn+2))
    J_uus = hstack([J_uusp,J_uuss,J_uusn])
    
    # uT
    bc_uTs = {"left":bc_uMs_grad[6:8]}
    J_uTss = build_tridiag(Ms,A_us[3:6], **bc_uTs)
#    J_uTsp = coo_matrix((np.ravel(np.asarray(bc_u0s_grad[2:4])),([0,0],[Mp,Mp+1] )),shape=(Ms+2,Mp+2))
    J_uTsp = empty_rec(Ms+2,Mp+2)
    J_uTsn = coo_matrix((np.ravel(np.asarray(bc_uMs_grad[2:4])),([Ms+1,Ms+1],[0,1] )),shape=(Ms+2,Mn+2))
    J_uTs = hstack([J_uTsp,J_uTss,J_uTsn])
    
    """ negative electrode"""
    arg_un = [uvec_ne[0:Mn], uvec_ne[1:Mn+1], uvec_ne[2:Mn+2], Tvec_ne[0:Mn], Tvec_ne[1:Mn+1], Tvec_ne[2:Mn+2], j_ne[0:Mn],uvec_old_ne[1:Mn+1]]
    arg_u0n = [uvec_ne[0], uvec_ne[1],  uvec_sep[Ms],uvec_sep[Ms+1]]
    
    A_un = vmap( grad( neq.electrolyte_conc, range(0,len(arg_un)-1) ) )(*arg_un)
    bc_u0n_grad = grad(neq.bc_inter_cont,range(0,len(arg_u0n)))(*arg_u0n)
    bc_uMn_grad =  np.array([[-1,1]]).T
    
    #uu
    bc_uun = {"right":bc_u0n_grad[0:2], "left":bc_uMn_grad[0:2]}
    J_uunn = build_tridiag(Mn, A_un[0:3], **bc_uun)
    J_uuns = coo_matrix((np.ravel(np.asarray(bc_u0n_grad[2:4])),([0,0],[Mp+2 + Ms, Mp+2 + Ms+1] )),shape=(Mn+2,Ms+2+Mp+2))
    J_uun = hstack([J_uuns, J_uunn])
    # uT
    
    J_uTnn = build_tridiag(Mn,A_un[3:6])
    
    J_uTns = empty_rec(Mn+2,Ms+2+Mp+2)
    J_uTn = hstack([J_uTns, J_uTnn])
    
    # uj
    J_ujn = build_diag(Mn,A_un[6], "long")
    
    res_u = np.hstack((bc_u0p, res_up, bc_uMp, bc_u0s, res_us, bc_uMs, bc_u0n, res_un, bc_uMn))
    
    J_u = hstack([
    empty_rec(Mp+2+Ms+2+Mn+2, Mp*(Np+2) + Mn*(Nn+2)), # c
    vstack([ J_uup, J_uus, J_uun]),
    hstack([
            empty_rec( Mp+2+Ms+2+Mn+2, Ma+2 ), # acc 
            vstack([J_uTp,J_uTs,J_uTn]),
            empty_rec( Mp+2+Ms+2+Mn+2, Mz+2 ) # zcc
            ]),
    empty_rec(Mp+2+Ms+2+Mn+2, Mp +2+Ms+2+Mn+2), #phie
    empty_rec(Mp+2+Ms+2+Mn+2, Mp+2+Mn+2), #phis
    vstack([
            hstack([J_ujp, empty_rec(Mp+2, Mn)]),
            empty_rec(Ms+2, Mp+Mn),
            hstack([empty_rec(Mn+2, Mp), J_ujn])
    ]),
    empty_rec(Mp+2+Ms+2+Mn+2, Mp+Mn)
    ])

    return res_u, J_u
    
    
