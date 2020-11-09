#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:22:58 2020

@author: hanrach
"""

import jax.numpy as np
from jax import vmap, grad
import jax
from jax.config import config
import timeit
#from batterySection import pe, sep, acc, zcc, ne
#from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, bmat, block_diag, hstack, vstack
config.update("jax_enable_x64", True)
from settings import Iapp, delta_t,F
from build_matrix import build_tridiag,build_diag, empty_square, empty_rec, build_bidiag
from ElectrodeEquation import peq,neq
from SeparatorEquation import sepq
from CurrentCollectorEquation import accq,zccq
import matplotlib.pylab as plt

def jacres_phis(acc,pe,sep,ne,zcc):
    Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
    Np = peq.N; Nn = neq.N
    #    def solid_poten(self,phisn, phisc, phisp, j):
    #        hx = self.hx; a = self.a
    #        sigeff = self.sigma*(1-self.eps-self.epsf)
    #        ans = sigeff*( phisn - 2*phisc + phisp)/hx**2 - a*F*j
    #        return ans.reshape()
    #    
    #    def bc_phis(self,phis0, phis1, source):
    #        sigeff = self.sigma*(1-pe.eps-pe.epsf)
    #        bc = sigeff*( phis1 - phis0 )/pe.hx - source
    #        return bc.reshape()
    """ Positive electrode"""
    arg_phis0p = [pe.phisvec[0], pe.phisvec[1], -Iapp]
    arg_phisp =  [pe.phisvec[0:Mp], pe.phisvec[1:Mp+1], pe.phisvec[2:Mp+2], pe.jvec[0:Mp]]
    arg_phisMp = [pe.phisvec[Mp], pe.phisvec[Mp+1]]
    
    bc_phis0p = peq.bc_phis(*arg_phis0p)
    res_phisp = vmap(peq.solid_poten)(*arg_phisp)
    bc_phisMp = peq.bc_zero_neumann(*arg_phisMp)
    
    bc_phis0p_grad = grad(peq.bc_phis,range(0,2))(*arg_phis0p)
    A_phisp = vmap(grad(peq.solid_poten, range(0,4)))(*arg_phisp)
    bc_phisMp_grad = grad(peq.bc_zero_neumann,range(0,2))(*arg_phisMp)
    
    bcphisp = {"right":bc_phis0p_grad, "left":bc_phisMp_grad}
    J_phisphisp = build_tridiag(Mp, A_phisp[0:3], **bcphisp)
    
    J_phisjp = build_diag(Mp,A_phisp[3],"long")
    
    
    """ Negative electrode"""
    arg_phis0n = [ne.phisvec[0], ne.phisvec[1]]
    arg_phisn =  [ne.phisvec[0:Mn], ne.phisvec[1:Mn+1], ne.phisvec[2:Mn+2], ne.jvec[0:Mn]]
    arg_phisMn = [ne.phisvec[Mn], ne.phisvec[Mn+1], -Iapp]
    
    bc_phis0n = neq.bc_zero_neumann(*arg_phis0n)
    res_phisn = vmap(neq.solid_poten)(*arg_phisn)
    bc_phisMn = neq.bc_phis(*arg_phisMn)
    
    bc_phis0n_grad = grad(neq.bc_zero_neumann,range(0,2))(*arg_phis0n)
    A_phisn = vmap(grad(neq.solid_poten, range(0,4)))(*arg_phisn)
    bc_phisMn_grad = grad(neq.bc_phis,range(0,2))(*arg_phisMn)
    
    bcphisn = {"right":bc_phis0n_grad, "left":bc_phisMn_grad}
    J_phisphisn = build_tridiag(Mn, A_phisn[0:3], **bcphisn)
    
    J_phisjn = build_diag(Mn,A_phisn[3],"long")
    
    J_phisphis = block_diag((J_phisphisp, J_phisphisn))
    J_phisj = block_diag((J_phisjp, J_phisjn))
    
    
    res_phis = np.hstack((bc_phis0p, res_phisp, bc_phisMp, bc_phis0n, res_phisn, bc_phisMn))
    J_phis = hstack([
            empty_rec(Mp+2+Mn+2, Mp*(Np+2) + Mn*(Nn+2)), #c
            empty_rec(Mp+2+Mn+2, Mp+2+Ms+2+Mn+2),#u
            empty_rec(Mp+2+Mn+2, Ma+2+Mp+2+Ms+2+Mn+2+Mz+2),#T
            empty_rec(Mp+2+Mn+2, Mp+2+Ms+2+Mn+2),#phie
            J_phisphis,
            J_phisj,
            empty_rec(Mp+2+Mn+2, Mp+Mn)
            ])
    
    return res_phis, J_phis