#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:13:12 2020

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

def jacres_j(acc, pe, sep, ne, zcc):
    Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
    Np = peq.N; Nn = neq.N
    
    arg_jp = [pe.jvec[0:Mp], pe.uvec[1:Mp+1], pe.Tvec[1:Mp+1], pe.etavec[0:Mp],  pe.cmat[Np,:], pe.cmat[Np+1,:], pe.cmax*np.ones([Mp,1])]
    res_jp = vmap(peq.ionic_flux)(*arg_jp)
    A_jp = vmap( (grad(peq.ionic_flux, range(0,len(arg_jp)-1))) )(*arg_jp)
    
    J_jp = build_diag(Mp, A_jp[0], "square")
    J_jup = build_diag(Mp, A_jp[1], "wide")
    J_jTp = build_diag(Mp, A_jp[2], "wide")
    J_jetap = build_diag(Mp, A_jp[3], "square")
    
    col_cp = []
    data_cp = []
#    row_cp = np.repeat( np.arange(0,Mp), 2)
    row_cp = np.repeat( np.arange(0,Mp), 2)
    for i in range(0,Mp):
#        col_cp.append([Np + (Np+2)*(i), Np+1 + (Np+2)*(i) ])
#        data_cp.append([A_jp[4][i], A_jp[5][i]])
        col_cp.append([Np + (Np+2)*i,Np+1 + (Np+2)*(i) ])
        data_cp.append([A_jp[4][i], A_jp[5][i]])
    data_cp =  np.ravel(np.array(data_cp))
    col_cp =  np.ravel(np.array(col_cp))
    J_cp = coo_matrix((data_cp,(row_cp,col_cp)), shape = (Mp, Mp*(Np+2) + Mn*(Nn+2)) )
    
    """ Negative Electrode"""
    
    arg_jn = [ne.jvec[0:Mn], ne.uvec[1:Mn+1], ne.Tvec[1:Mn+1], ne.etavec[0:Mn], ne.cmat[Nn,:], ne.cmat[Nn+1,:], ne.cmax*np.ones([Mn,1])]
    res_jn = vmap(neq.ionic_flux)(*arg_jn)
    A_jn = vmap( (grad(neq.ionic_flux, range(0,len(arg_jn)-1))) )(*arg_jn)
    
    J_jn = build_diag(Mn, A_jn[0], "square")
    J_jun = build_diag(Mn, A_jn[1], "wide")
    J_jTn = build_diag(Mn, A_jn[2], "wide")
    J_jetan = build_diag(Mn, A_jn[3], "square")
    
    col_cn = []
    data_cn = []
    offset = (Np+2)*Mp
    row_cn = np.repeat(np.arange(0,Mn), 2)
    for i in range(0,Mn):
        col_cn.append([ Nn + (Nn+2)*i + offset, Nn+1 + (Nn+2)*(i) + offset ])
        data_cn.append([ A_jn[4][i], A_jn[5][i]])
    data_cn =  np.ravel(np.array(data_cn))
    col_cn =  np.ravel(np.array(col_cn))
    J_cn = coo_matrix((data_cn,(row_cn,col_cn)), shape = (Mn, Mp*(Np+2) + Mn*(Nn+2)) )
    
    
    """" total """
    J_ju = hstack([
            vstack([J_jup,empty_rec(Mn,Mp+2)]),
            empty_rec(Mp+Mn,Ms+2),
            vstack([empty_rec(Mp,Mn+2), J_jun])
            ])
    
    J_jT = hstack([
            empty_rec(Mp+Mn,Ma+2),
            vstack([J_jTp,empty_rec(Mn,Mp+2)]),
            empty_rec(Mp+Mn,Ms+2),
            vstack([empty_rec(Mp,Mn+2), J_jTn]),
            empty_rec(Mp+Mn,Mz+2)
            ])
    
    J_jj = block_diag((J_jp, J_jn))
    
    J_jeta = block_diag((J_jetap, J_jetan))
    
    J_jc = vstack([ J_cp, J_cn ])
    
    res_j = np.hstack((res_jp, res_jn))
    J_j = hstack([
            J_jc,
            J_ju, J_jT,
            empty_rec(Mp+Mn,Mp+2+Ms+2+Mn+2),
            empty_rec(Mp+Mn, Mp+2+Mn+2),
            J_jj, J_jeta
            ])
    
    return res_j, J_j

    