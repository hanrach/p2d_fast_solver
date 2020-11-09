#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:51:36 2020

@author: hanrach
"""

import jax.numpy as np
from jax import vmap, grad
#import jax
from jax.config import config
#import timeit
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

def jacres_phie(acc,pe,sep, ne,zcc):
    Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
    Np = peq.N; Nn = neq.N
    """ Positive Electrode"""
    #electrolyte_poten(self,un, uc, up, phien, phiec, phiep, Tn, Tc, Tp,j):
    arg_phie0p = [pe.phievec[0], pe.phievec[1]]
    arg_phiep = [pe.uvec[0:Mp], pe.uvec[1:Mp+1], pe.uvec[2:Mp+2], pe.phievec[0:Mp], pe.phievec[1:Mp+1], pe.phievec[2:Mp+2],
                  pe.Tvec[0:Mp], pe.Tvec[1:Mp+1], pe.Tvec[2:Mp+2], pe.jvec[0:Mp]]
    #bc_phie_p(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s, T0_p, T1_p, T0_s, T1_s)
    arg_phieMp = [pe.phievec[Mp], pe.phievec[Mp+1], sep.phievec[0], sep.phievec[1],
                  pe.uvec[Mp], pe.uvec[Mp+1], sep.uvec[0], sep.uvec[1],
                  pe.Tvec[Mp], pe.Tvec[Mp+1], sep.Tvec[0], sep.Tvec[1]]
    
    bc_phie0p = peq.bc_zero_neumann(*arg_phie0p)
    res_phiep = vmap(peq.electrolyte_poten)(*arg_phiep)
    bc_phieMp = peq.bc_phie_p(*arg_phieMp)
    
    bc_phie0p_grad = np.array([[-1,1]]).T
    A_phiep = vmap( (grad(peq.electrolyte_poten, range(0, len(arg_phiep)) )) )(*arg_phiep)
    bc_phieMp_grad = grad(peq.bc_phie_p,range(0, len(arg_phieMp)))(*arg_phieMp)
    
    bcphiep = {"right":bc_phie0p_grad, "left":bc_phieMp_grad[0:2]}
    J_phiephiepp = build_tridiag(Mp, A_phiep[3:6], **bcphiep)
    J_phieps = coo_matrix( (np.ravel(np.asarray(bc_phieMp_grad[2:4])), ([Mp+1,Mp+1], [0,1]) ), shape = (Mp+2,Ms+2+Mn+2) )
    J_phiephiep = hstack([ J_phiephiepp, J_phieps])
    
    bc_phieup = {"left":bc_phieMp_grad[4:6]}
    J_phieupp = build_tridiag(Mp, A_phiep[0:3], **bc_phieup)
    J_phieups = coo_matrix( (np.ravel(np.asarray(bc_phieMp_grad[6:8])), ([Mp+1,Mp+1], [0,1]) ), shape = (Mp+2,Ms+2+Mn+2) )
    J_phieup = hstack([J_phieupp, J_phieups])
    
    bc_phieTp = {"left":bc_phieMp_grad[8:10]}
    J_phieTpp = build_tridiag(Mp, A_phiep[6:9], **bc_phieTp)
    J_phieTps = coo_matrix( (np.ravel(np.asarray(bc_phieMp_grad[10:12])), ([Mp+1,Mp+1], [0,1]) ), shape = (Mp+2,Ms+2+Mn+2) )
    J_phieTp = hstack([J_phieTpp, J_phieTps])
    
    J_phiejp = build_diag(Mp, A_phiep[9], "long")
    
    
    """ Separator """
    
    arg_phie0s = [sep.phievec[0], sep.phievec[1], pe.phievec[Mp], pe.phievec[Mp+1]]
    arg_phies = [sep.uvec[0:Ms], sep.uvec[1:Ms+1], sep.uvec[2:Ms+2], sep.phievec[0:Ms], sep.phievec[1:Ms+1], sep.phievec[2:Ms+2],
                  sep.Tvec[0:Ms], sep.Tvec[1:Ms+1], sep.Tvec[2:Ms+2]]
    # bc_phie_sn(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s, T0_n, T1_n, T0_s, T1_s):
    arg_phieMs = [ne.phievec[0], ne.phievec[1], sep.phievec[Ms], sep.phievec[Ms+1],
                  ne.uvec[0], ne.uvec[1], sep.uvec[Ms], sep.uvec[Ms+1],
                  ne.Tvec[0], ne.Tvec[1], sep.Tvec[Ms], sep.Tvec[Ms+1]]
    
    bc_phie0s = peq.bc_inter_cont(*arg_phie0s)
    res_phies = vmap(sepq.electrolyte_poten)(*arg_phies)
    bc_phieMs = sepq.bc_phie_sn(*arg_phieMs)
    
    bc_phie0s_grad = grad(peq.bc_inter_cont, range(0,len(arg_phie0s)))(*arg_phie0s)
    A_phies = vmap( grad(sepq.electrolyte_poten, range(0, len(arg_phies))) )(*arg_phies)
    bc_phieMs_grad = grad(sepq.bc_phie_sn, range(0, len(arg_phieMs)))(*arg_phieMs)
    
    bcphies = {"right":bc_phie0s_grad[0:2], "left":bc_phieMs_grad[2:4]}
    J_phiephiess = build_tridiag(Ms, A_phies[3:6], **bcphies)
    J_phiephiesp = coo_matrix( (np.ravel(np.asarray(bc_phie0s_grad[2:4])), ([0,0], [Mp,Mp+1]) ), shape = (Ms+2,Mp+2) )
    J_phiephiesn = coo_matrix( (np.ravel(np.asarray(bc_phieMs_grad[0:2])), ([Ms+1, Ms+1], [0,1])), shape = (Ms+2,Mn+2))
    J_phiephies = hstack([J_phiephiesp, J_phiephiess, J_phiephiesn])
    
    bcphieus = {"left":bc_phieMs_grad[6:8]}
    J_phieuss = build_tridiag(Ms,A_phies[0:3], **bcphieus)
    J_phieusp = empty_rec(Ms+2,Mp+2)
    J_phieusn = coo_matrix( (np.ravel(np.asarray(bc_phieMs_grad[4:6])), ([Ms+1, Ms+1], [0,1])), shape = (Ms+2,Mn+2))
    J_phieus = hstack([J_phieusp, J_phieuss, J_phieusn])
    
    bcphieTs = {"left":bc_phieMs_grad[10:12]}
    J_phieTss = build_tridiag(Ms,A_phies[6:9], **bcphieTs)
    J_phieTsp = empty_rec(Ms+2,Mp+2)
    J_phieTsn = coo_matrix( (np.ravel(np.asarray(bc_phieMs_grad[8:10])), ([Ms+1, Ms+1], [0,1])), shape = (Ms+2,Mn+2))
    J_phieTs = hstack([J_phieTsp, J_phieTss, J_phieTsn])
    
    """ Negative Electrode"""
    arg_phie0n = [ne.phievec[0],ne.phievec[1], sep.phievec[Ms], sep.phievec[Ms+1]]
    arg_phien = [ne.uvec[0:Mn], ne.uvec[1:Mn+1], ne.uvec[2:Mn+2], ne.phievec[0:Mn], ne.phievec[1:Mn+1], ne.phievec[2:Mn+2],
                  ne.Tvec[0:Mn], ne.Tvec[1:Mn+1], ne.Tvec[2:Mn+2], ne.jvec[0:Mn]]
    arg_phieMn = [ne.phievec[Mn], ne.phievec[Mn+1]]
    
    bc_phie0n = neq.bc_inter_cont(*arg_phie0n)
    res_phien = vmap(neq.electrolyte_poten)(*arg_phien)
    bc_phieMn = neq.bc_zero_dirichlet(*arg_phieMn)
    
    bc_phie0n_grad = grad(neq.bc_inter_cont,range(0,len(arg_phie0n)))(*arg_phie0n)
    A_phien = vmap( grad(neq.electrolyte_poten, range(0,len(arg_phien))) )(*arg_phien)
    bc_phieMn_grad =np.array([[1/2,1/2]]).T
    
    bcphien = {"right":bc_phie0n_grad[0:2], "left":bc_phieMn_grad}
    J_phiephienn = build_tridiag(Mn, A_phien[3:6], **bcphien)
    J_phiephiens = coo_matrix( (np.ravel(np.asarray(bc_phie0n_grad[2:4])), ([0,0], [Mp+2+Ms,Mp+2+Ms+1]) ), shape = (Mn+2,Mp+2+Ms+2) )
    J_phiephien = hstack([J_phiephiens, J_phiephienn])
    
    J_phieunn = build_tridiag(Mn, A_phien[0:3])
    J_phieuns = empty_rec(Mn+2,Mp+2+Ms+2) 
    J_phieun = hstack([J_phieuns, J_phieunn])
    
    J_phieTnn = build_tridiag(Mn, A_phien[6:9])
    J_phieTns = empty_rec(Mn+2,Mp+2+Ms+2) 
    J_phieTn = hstack([J_phieTns, J_phieTnn])
    
    J_phiejn = build_diag(Mn, A_phien[9], "long")
    
    J_phieu = vstack((J_phieup, J_phieus, J_phieun))
    J_phiephie = vstack((J_phiephiep, J_phiephies, J_phiephien))
    J_phieT = hstack([
            empty_rec(Mp+2+Mn+2+Ms+2,Ma+2),
            vstack((J_phieTp,J_phieTs,J_phieTn)),
            empty_rec(Mp+2+Mn+2+Ms+2,Mz+2)
            ])
    J_phiej = vstack([
            hstack([J_phiejp, empty_rec(Mp+2, Mn)]),
            empty_rec(Ms+2, Mp+Mn),
            hstack([empty_rec(Mn+2, Mp), J_phiejn])
            ])
    
    J_phiec = empty_rec(Mp+2+Ms+2+Mn+2, Mp*(Np+2) + Mn*(Nn+2))
    
    res_phie = np.hstack((bc_phie0p, res_phiep, bc_phieMp,
                          bc_phie0s, res_phies, bc_phieMs,
                          bc_phie0n, res_phien, bc_phieMn))
    J_phie = hstack([
            J_phiec,
            J_phieu, J_phieT, J_phiephie,
            empty_rec(Mp+2+Mn+2+Ms+2,Mp+2+Mn+2),
            J_phiej, empty_rec(Mp+2+Ms+2+Mn+2,Mp+Mn)])
    
    return res_phie, J_phie

