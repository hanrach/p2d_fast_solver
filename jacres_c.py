#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:36:46 2020

@author: hanrach
"""
import jax.numpy as np
import numpy as onp
from jax import vmap, grad
import jax
from jax.config import config
import timeit
#from batterySection import pe, sep, acc, zcc, ne
#from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, bmat, block_diag, hstack, vstack, identity, kron
config.update("jax_enable_x64", True)
from settings import Iapp, delta_t,F
from build_matrix import build_tridiag,build_diag, empty_square, empty_rec, build_bidiag, build_tridiag_c
import matplotlib.pylab as plt
import coeffs as coeffs
from unpack import unpack
def jacres_c(U, Uold, peq, neq, sepq, accq, zccq):
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

    
    """ Positive """
    k = delta_t;
    Np = peq.N
    Rp = peq.Rp*(np.linspace(1,N,N)-(1/2))/N;
    rp = peq.Rp*(np.linspace(0,N, N+1))/N
    lambda1_p = k*r[0:N]**2/(R**2*hy**2);
    lambda2_p = k*r[1:N+1]**2/(R**2*hy**2);
    
    res_cp = onp.zeros([(Np+2)*Mp,1])
    for i in range(0,Mp):
        arg_cp = [cmat_pe[0:Np,i], cmat_pe[1:Np+1,i], cmat_pe[2:Np+2,i], cmat_old_pe[1:Np+1, i], lambda1_p, lambda2_p]
        arg_c0p = [cmat_pe[0,i], cmat_pe[1,i]]
        arg_cMp = [cmat_pe[Np,i], cmat_pe[Np+1,i], j_pe[i], Tvec_pe[i+1]]
        
        res_cp[(i)*(Np+2)] = peq.bc_zero_neumann(*arg_c0p)
        res_cp[i*(Np+2)+1: Np + 1 + i*(Np+2)] = np.reshape(peq.solid_conc(*arg_cp), [Np,1])
#        res_cp[i*(Np+2)+1: Np + 1 + i*(Np+2)] = vmap(peq.solid_conc)(*arg_cp)
        res_cp[(Np+1) + i*(Np+2)] = peq.bc_neumann_c(*arg_cMp) 
        
    res_cp = np.asarray(res_cp)
       
    """ Linear Jacobian """
    bc_cp0_grad = np.array([[-1,1]]).T
    A_cp = vmap(grad(peq.solid_conc_2, range(0,3) ))(*arg_cp)
    bc_cpM_grad = np.array([[-1/peq.hy,1/peq.hy]]).T
    
    bcp = {"right":bc_cp0_grad, "left": bc_cpM_grad}
    J_cp_sub = build_tridiag_c(Np,A_cp,**bcp )
    Ip=identity(Mp)
    J_cp = kron(Ip, J_cp_sub)
    
    """ d resc / d j : positive """
    Deff = vmap(coeffs.solidDiffCoeff)(peq.Ds*np.ones([Mp,1]), peq.ED*np.ones([Mp,1]), Tvec_pe[1:Mp+1])
    dcj_p = 1/Deff
    
    # build the jacobian for j
    row_cjp = onp.zeros(Mp, dtype = int) 
    col_cjp = onp.arange(0,Mp)
    for i in range(0, Mp):
        row_cjp[i] = Np + 1 + i*(Np+2)
    J_cjp = coo_matrix((dcj_p.ravel(),(row_cjp,col_cjp)), shape=(Mp*(Np+2), Mp))
    
    
    """ d resc / d T : positive """
    dcT_p =vmap( grad(peq.bc_neumann_c, 3))(cmat_pe[Np,0:Mp], cmat_pe[Np+1,0:Mp], j_pe[0:Mp], Tvec_pe[1:Mp+1])
    #dcT_ps = grad(coeffs.solidDiffCoeff, 2)(peq.Ds, peq.ED, Tvec_pe[1])
    
    # build the jacobian for j
    row_cTp = onp.zeros(Mp, dtype = int) 
    col_cTp = onp.arange(1,Mp+1)
    for i in range(0, Mp):
        row_cTp[i] = Np + 1 + i*(Np+2)
    J_cTp = coo_matrix((dcT_p.ravel(),(row_cTp,col_cTp)), shape=(Mp*(Np+2) + Mn*(Nn+2), Mp+2))
    
    """ Negative """
    res_cn = onp.zeros([(Nn+2)*Mn,1])
    for i in range(0,Mn):
        arg_cn = [cmat_ne[0:Nn,i], cmat_ne[1:Nn+1,i], cmat_ne[2:Nn+2,i], cmat_old_ne[1:Nn+1, i]]
        arg_c0n = [cmat_ne[0,i], cmat_ne[1,i]]
        arg_cMn = [cmat_ne[Nn,i], cmat_ne[Nn+1,i], j_ne[i], Tvec_ne[i+1]]
        res_cn[(i)*(Nn+2)] = neq.bc_zero_neumann(*arg_c0n)
        res_cn[i*(Nn+2)+1: Nn + 1 + i*(Nn+2)] = np.reshape(neq.solid_conc_2(*arg_cn), [Nn,1])
        res_cn[(Nn+1) + i*(Nn+2)] = neq.bc_neumann_c(*arg_cMn)
        
    res_cn = np.asarray(res_cn)
        
    """ Linear Jacobian """
    bc_cn0_grad = np.array([[-1,1]]).T
    A_cn = vmap(grad(neq.solid_conc_2, range(0,3) ))(*arg_cn)
    bc_cnM_grad = np.array([[-1,1]]).T
    
    bcn = {"right":bc_cn0_grad, "left": bc_cnM_grad}
    J_cn_sub = build_tridiag_c(Nn,A_cn,**bcn)
    In=identity(Mn)
    J_cn = kron(In, J_cn_sub)
    
    """ d resc / d j : negative """
    Deffn = vmap(coeffs.solidDiffCoeff)(neq.Ds*np.ones([Mn,1]), neq.ED*np.ones([Mn,1]), Tvec_ne[1:Mn+1])
    dcj_n= neq.hy/Deffn
    
    # build the jacobian for j
    row_cjn = onp.zeros(Mn, dtype = int) 
    col_cjn = onp.arange(0,Mn)
    for i in range(0, Mn):
        row_cjn[i] = Nn + 1 + i*(Nn+2)
    J_cjn = coo_matrix((dcj_n.ravel(),(row_cjn,col_cjn)), shape=(Mn*(Nn+2), Mn))
    
    """ d resc / d T : negative """
    dcT_n =vmap( grad(neq.bc_neumann_c,3))(cmat_ne[Np,0:Mp], cmat_ne[Np+1,0:Mp], j_ne[0:Mp], Tvec_ne[1:Mp+1])
    #dcT_ps = grad(coeffs.solidDiffCoeff, 2)(peq.Ds, peq.ED, Tvec_pe[1])
    
    # build the jacobian for j
    row_cTn = onp.zeros(Mn, dtype = int) 
    col_cTn = onp.arange(1,Mn+1)
    for i in range(0, Mn):
        row_cTn[i] = Nn + 1 + i*(Nn+2)  + Mp*(Np+2)
    J_cTn = coo_matrix((dcT_n.ravel(),(row_cTn,col_cTn)), shape=(Mn*(Nn+2) + Mp*(Np+2), Mn+2))
    
    J_cc = block_diag((J_cp, J_cn))
    J_cj = block_diag((J_cjp, J_cjn))
    J_cT = hstack([
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Ma+2),
            J_cTp,
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Ms+2),
            J_cTn,
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Mz+2)
            ])
            
    res_c = np.vstack((res_cp, res_cn))
            
    J_c = hstack([
            J_cc,
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Mp+2+Mn+2+Ms+2),
            J_cT,
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Mp+2+Mn+2+Ms+2),
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Mp+2+Mn+2),
            J_cj,
            empty_rec(Mp*(Np+2) + Mn*(Nn+2), Mp+Mn)
            ])
            
    return res_c, J_c