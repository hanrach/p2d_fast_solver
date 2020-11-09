#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:06:42 2020

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

def jacres_eta(acc,pe,sep,ne,zcc):
    Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
    Np = peq.N; Nn = neq.N
    # over_poten(self,eta, phis,phie, T, cs, cmax):
    arg_etap = [pe.etavec[0:Mp], pe.phisvec[1:Mp+1], pe.phievec[1:Mp+1], pe.Tvec[1:Mp+1], pe.cmat[Np,:], pe.cmat[Np+1,:], pe.cmax*np.ones([Mp,1])]
    res_etap = vmap(peq.over_poten)(*arg_etap)
    A_etap = vmap((grad(peq.over_poten, range(0,len(arg_etap)-1))))(*arg_etap)
    
    J_etap = build_diag(Mp,A_etap[0],"square")
    J_etaphisp = build_diag(Mp,A_etap[1], "wide")
    J_etaphiep = build_diag(Mp,A_etap[2], "wide")
#    J_etajp = build_diag(Mp,A_etap[4],"square")
    J_etajp = empty_square(Mp)
    J_etaTp = build_diag(Mp,A_etap[3],"wide")
    
    col_cp = []
    data_cp = []
    row_cp = np.repeat( np.arange(0,Mp), 2)
    for i in range(0,Mp):
        col_cp.append([ Np + (Np+2)*i, Np+1 + (Np+2)*(i) ])
        data_cp.append([ A_etap[4][i], A_etap[5][i]])
    data_cp =  np.ravel(np.array(data_cp))
    col_cp =  np.ravel(np.array(col_cp))
    J_cp = coo_matrix((data_cp,(row_cp,col_cp)), shape = (Mp, Mp*(Np+2) + Mn*(Nn+2)) )
    
    """negative"""
    arg_etan = [ne.etavec[0:Mn], ne.phisvec[1:Mn+1], ne.phievec[1:Mn+1], ne.Tvec[1:Mn+1], ne.cmat[Nn,:], ne.cmat[Nn+1,:], ne.cmax*np.ones([Mn,1])]
    res_etan = vmap(neq.over_poten)(*arg_etan)
    A_etan = vmap((grad(neq.over_poten, range(0,len(arg_etan)-1))))(*arg_etan)
    
    J_etan = build_diag(Mn,A_etan[0],"square")
    J_etaphisn = build_diag(Mn,A_etan[1], "wide")
    J_etaphien = build_diag(Mn,A_etan[2], "wide")
#    J_etajn = build_diag(Mn,A_etan[4],"square")
    J_etajn = empty_square(Mn)
    J_etaTn = build_diag(Mn,A_etan[3],"wide")
    
    col_cn = []
    data_cn = []
    offset = (Np+2)*Mp
    row_cn = np.repeat(np.arange(0,Mn), 2)
    for i in range(0,Mn):
        col_cn.append([  Nn + (Nn+2)*i + offset, Nn+1 + (Nn+2)*(i) + offset ])
        data_cn.append([ A_etan[4][i], A_etan[5][i] ])
    data_cn =  np.ravel(np.array(data_cn))
    col_cn =  np.ravel(np.array(col_cn))
    J_cn = coo_matrix((data_cn,(row_cn,col_cn)), shape = (Mn, Mp*(Np+2) + Mn*(Nn+2)) )
    
    J_etaphis = hstack([
            vstack([J_etaphisp,empty_rec(Mn,Mp+2)]),
            vstack([empty_rec(Mp,Mn+2), J_etaphisn])
            ])
    
    J_etaphie = hstack([
            vstack([J_etaphiep, empty_rec(Mn, Mp+2)]),
            empty_rec(Mn+Mp,Ms+2),
            vstack([empty_rec(Mp, Mn+2), J_etaphien])
            ])
    
    J_etaT = hstack([
            empty_rec(Mp+Mn,Ma+2),
            vstack([J_etaTp,empty_rec(Mn,Mp+2)]),
            empty_rec(Mp+Mn,Ms+2),
            vstack([empty_rec(Mp,Mn+2), J_etaTn]),
            empty_rec(Mp+Mn,Mz+2)
            ])
    
    J_etaeta = block_diag((J_etap, J_etan))
    
    J_etaj = block_diag((J_etajp, J_etajn))
    
    J_etac = vstack([J_cp, J_cn])
    
    res_eta = np.hstack((res_etap, res_etan))
    J_eta = hstack([
            J_etac,
            empty_rec(Mp+Mn,Mp+2+Ms+2+Mn+2),
            J_etaT,
            J_etaphie,
            J_etaphis,
            J_etaj,
            J_etaeta
            ])
    
    return res_eta, J_eta