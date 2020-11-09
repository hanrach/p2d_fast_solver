#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:37:02 2020

@author: hanrach
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:47:11 2020

@author: hanrach

TEMPERATURE
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
import matplotlib.pylab as plt
from unpack import unpack
def jacres_T(U, Uold, peq, neq, sepq, accq, zccq):
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

    """ Current Collector a residual """
    
    bc_T0a = accq.bc_temp_a(Tvec_acc[0],Tvec_acc[1])
    res_Ta = vmap(accq.temperature)(Tvec_acc[0:Ma], Tvec_acc[1:Ma+1], Tvec_acc[2:Ma+2], Tvec_old_acc[1:Ma+1])
    bc_TMa = peq.bc_temp_ap(Tvec_acc[Ma], Tvec_acc[Ma+1], Tvec_pe[0], Tvec_pe[1])
    
    """ jacobian"""
    bc_T0a_grad = grad(accq.bc_temp_a,(0,1))(Tvec_acc[0],Tvec_acc[1])  
    A_Ta = vmap( grad(accq.temperature, range(0,3)) )(Tvec_acc[0:Ma], Tvec_acc[1:Ma+1], Tvec_acc[2:Ma+2], Tvec_old_acc[1:Ma+1])
    bc_TMa_grad = grad(peq.bc_temp_ap, range(0,4))(Tvec_acc[Ma], Tvec_acc[Ma+1], Tvec_pe[0], Tvec_pe[1])
    
    bcTa = {"right":bc_T0a_grad[0:2], "left": bc_TMa_grad[0:2]}
    J_TTaa = build_tridiag(Ma,A_Ta[0:3], **bcTa)
    J_TTap = coo_matrix((np.ravel(np.asarray(bc_TMa_grad[2:4])),([Ma+1,Ma+1],[0,1] )),shape=(Ma+2,Mp+2+Ms+2+Mn+2+Mz+2))
    J_TTa = hstack([J_TTaa, J_TTap])        
    
    """ Positive electrode residual """
    arg_T0p = [Tvec_acc[Ma],Tvec_acc[Ma+1], Tvec_pe[0], Tvec_pe[1]]
    #arg_T = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], pe.phievec[0:Mp],pe.phievec[2:Mp+2],\
    #         phis_pe[0:Mp], phis_pe[2:Mp+2], Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], j_pe[0:Mp],\
    #         eta_pe[0:Mp], pe.cs[0:Mp], pe.cmax*np.ones([Mp,1]), Tvec_old_pe[1:Mp+1]]
    arg_Tp = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2],
              phie_pe[0:Mp],phie_pe[2:Mp+2],\
             phis_pe[0:Mp], phis_pe[2:Mp+2],
             Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], 
             j_pe[0:Mp],\
             eta_pe[0:Mp], 
             cmat_pe[Np,:], cmat_pe[Np+1,:],peq.cmax*np.ones([Mp,1]), 
             Tvec_old_pe[1:Mp+1]]
    arg_TMp = [Tvec_pe[Mp], Tvec_pe[Mp+1], Tvec_sep[0], Tvec_sep[1]]
    
    bc_T0p = peq.bc_inter_cont(*arg_T0p)
    res_Tp = vmap(peq.temperature)(*arg_Tp)
    bc_TMp = peq.bc_temp_ps(*arg_TMp)
    
    A_Tp = vmap( grad(peq.temperature, range(0,len(arg_Tp) - 2)) )(*arg_Tp)
    bc_T0p_grad = grad(peq.bc_inter_cont, range(0,len(arg_T0p)))(*arg_T0p)
    bc_TMp_grad = grad(peq.bc_temp_ps,range(0,4))(*arg_TMp)
    
    
    bcTp = {"right":bc_T0p_grad[2:4], "left":bc_TMp_grad[0:2]}
    J_TTpp = build_tridiag(Mp, A_Tp[7:10], **bcTp)
    J_TTpa = coo_matrix((np.ravel(np.asarray(bc_T0p_grad[0:2])),([0,0],[Ma,Ma+1] )),shape=(Mp+2,Ma+2))
    J_TTps = coo_matrix((np.ravel(np.asarray(bc_TMp_grad[2:4])),([Mp+1,Mp+1],[0,1] )),shape=(Mp+2,Ms+2+Mn+2+Mz+2))
    J_TTp = hstack([J_TTpa,J_TTpp,J_TTps])
    
    J_Tup = build_tridiag(Mp, A_Tp[0:3])
    J_Tphiep = build_bidiag(Mp,A_Tp[3:5])
    J_Tphisp = build_bidiag(Mp,A_Tp[5:7])
    J_Tjp = build_diag(Mp, A_Tp[10], "long")
    J_Tetap = build_diag(Mp,A_Tp[11], "long")
    
    col_cp = []
    data_cp = []
    row_cp= np.repeat( Ma+2 + np.arange(1,Mp+1), 2)
    for i in range(0,Mp):
        col_cp.append([Np + (Np+2)*i, Np+1 + (Np+2)*(i)])
        data_cp.append([A_Tp[12][i], A_Tp[13][i]])
    data_cp =  np.ravel(np.array(data_cp))
    col_cp =  np.ravel(np.array(col_cp))
    J_cp = coo_matrix((data_cp,(row_cp,col_cp)), shape = (Ma+2 + Mp+2,Mp*(Np+2) + Mn*(Nn+2) ) )
        
    """ Separator residual """
    
    arg_T0s = [Tvec_pe[Mp], Tvec_pe[Mp+1], Tvec_sep[0], Tvec_sep[1]]
    arg_Ts = [uvec_sep[0:Ms], uvec_sep[1:Ms+1], uvec_sep[2:Ms+2], phie_sep[0:Ms], phie_sep[2:Ms+2],\
              Tvec_sep[0:Ms], Tvec_sep[1:Ms+1], Tvec_sep[2:Ms+2], Tvec_old_sep[1:Ms+1]]
    arg_TMs = [Tvec_sep[Ms], Tvec_sep[Ms+1], Tvec_ne[0], Tvec_ne[1]]
    
    bc_T0s = peq.bc_inter_cont(*arg_T0s)
    res_Ts = vmap(sepq.temperature)(*arg_Ts)
    bc_TMs = sepq.bc_temp_sn(*arg_TMs)
    
    bc_T0s_grad = grad(peq.bc_inter_cont, range(0,4))(*arg_T0s)
    bc_TMs_grad = grad(sepq.bc_temp_sn, range(0,4))(*arg_TMs)
    A_Ts = vmap( grad(sepq.temperature, range(0, len(arg_Ts) -1)) )(*arg_Ts)
    
    bcTs = {"right": bc_T0s_grad[2:4], "left": bc_TMs_grad[0:2]}
    J_TTss = build_tridiag(Ms, A_Ts[5:8], **bcTs)
    J_TTsp = coo_matrix((np.ravel(np.asarray(bc_T0s_grad[0:2])), ([0,0],[Ma+2+Mp, Ma+2+Mp+1]) ), shape = (Ms+2,Ma+2+Mp+2) )
    J_TTsn = coo_matrix((np.ravel(np.asarray(bc_TMs_grad[2:4])), ([Ms+1,Ms+1],[0, 1]) ), shape = (Ms+2,Mn+2+Mz+2) )
    J_TTs = hstack([J_TTsp, J_TTss, J_TTsn])
    
    J_Tus = build_tridiag(Ms, A_Ts[0:3])
    J_Tphies = build_bidiag(Ms, A_Ts[3:5])
    
    """ Negative residual """
    arg_T0n = [Tvec_sep[Ms], Tvec_sep[Ms+1], Tvec_ne[0], Tvec_ne[1]]
    arg_Tn = [uvec_ne[0:Mn], uvec_ne[1:Mn+1], uvec_ne[2:Mn+2], phie_ne[0:Mn],phie_ne[2:Mn+2],\
             phis_ne[0:Mn], phis_ne[2:Mn+2], Tvec_ne[0:Mn], Tvec_ne[1:Mn+1], Tvec_ne[2:Mn+2], j_ne[0:Mn],\
             eta_ne[0:Mn], cmat_ne[Nn,:], cmat_ne[Nn+1,:], neq.cmax*np.ones([Mn,1]), Tvec_old_ne[1:Mn+1]]
    arg_TMn = [Tvec_ne[Mn], Tvec_ne[Mn+1], Tvec_zcc[0], Tvec_zcc[1]]
    
    bc_T0n = neq.bc_inter_cont(*arg_T0n)
    res_Tn = vmap(neq.temperature)(*arg_Tn)
    bc_TMn = neq.bc_temp_n(*arg_TMn)
    
    """jacobian"""
    A_Tn = vmap( grad(neq.temperature, range(0,len(arg_Tn) - 2)) )(*arg_Tn)
    bc_T0n_grad = grad(neq.bc_inter_cont, range(0,4))(*arg_T0n)
    bc_TMn_grad = grad(neq.bc_temp_n, range(0,4))(*arg_TMn)
    
    bcTn = {"right":bc_T0n_grad[2:4], "left":bc_TMn_grad[0:2]}
    J_TTnn = build_tridiag(Mn, A_Tn[7:10], **bcTn)
    J_TTns = coo_matrix((np.ravel(np.asarray(bc_T0n_grad[0:2])), ([0,0],[Ma+2+Mp+2+Ms, Ma+2+Mp+2+Ms+1]) ), shape = (Mn+2,Ma+2+Mp+2+Ms+2) )
    J_TTnz = coo_matrix((np.ravel(np.asarray(bc_TMn_grad[2:4])), ([Mn+1,Mn+1],[0, 1]) ), shape = (Mn+2,Mz+2) )
    J_TTn = hstack([J_TTns, J_TTnn, J_TTnz])
    
    J_Tun = build_tridiag(Mn, A_Tn[0:3])
    J_Tphien = build_bidiag(Mn,A_Tn[3:5])
    J_Tphisn = build_bidiag(Mn,A_Tn[5:7])
    J_Tjn = build_diag(Mn, A_Tn[10], "long")
    J_Tetan = build_diag(Mn,A_Tn[11], "long")
    
    col_cn = []
    data_cn = []
    row_cn = np.repeat( np.arange(1,Mn+1), 2)
    offset = Mp*(Np+2)
    for i in range(0,Mn):
        col_cn.append([Nn + (Nn+2)*i + offset, Nn+1 + (Nn+2)*(i)  + offset ])
        data_cn.append([A_Tn[12][i], A_Tn[13][i]])
    data_cn =  np.ravel(np.array(data_cn))
    col_cn =  np.ravel(np.array(col_cn))
    J_cn = coo_matrix((data_cn,(row_cn,col_cn)), shape = (Mz+2 + Mn+2,Mn*(Nn+2) + offset ) )
    
    """ Current collector z residual """
    arg_T0z = [Tvec_ne[Mn], Tvec_ne[Mn+1], Tvec_zcc[0], Tvec_zcc[1]]
    arg_Tz = [Tvec_zcc[0:Mz], Tvec_zcc[1:Mz+1], Tvec_zcc[2:Mz+2], Tvec_old_zcc[1:Mz+1]]
    arg_TMz = [Tvec_zcc[Mz],Tvec_zcc[Mz+1]]
    
    bc_T0z = neq.bc_inter_cont(*arg_T0z)
    res_Tz = vmap(zccq.temperature)(*arg_Tz)
    bc_TMz = zccq.bc_temp_z(*arg_TMz)
    
    """ jacobian"""
    bc_T0z_grad = grad(neq.bc_inter_cont, range(0,4))(*arg_T0z)  
    A_Tz = vmap( grad(zccq.temperature, range(0,3)) )(*arg_Tz)
    bc_TMz_grad = grad(zccq.bc_temp_z, (0,1))(*arg_TMz)
    
    bcTz = {"right":bc_T0z_grad[2:4], "left": bc_TMz_grad[0:2]}
    J_TTzz = build_tridiag(Mz,A_Tz[0:3], **bcTz)
    J_TTzn = coo_matrix((np.ravel(np.asarray(bc_T0z_grad[0:2])), ([0,0],[Ma+2+Mp+2+Ms+2+Mn, Ma+2+Mp+2+Ms+2+Mn+1]) ), shape = (Mz+2,Ma+2+Mp+2+Ms+2+Mn+2) )
 
    J_TTz = hstack([J_TTzn, J_TTzz])
    
    
    J_Tu = bmat([ [empty_rec(Ma+2, (Mp+2)+(Ms+2)+(Mn+2))],\
                   [block_diag((J_Tup, J_Tus, J_Tun))],\
                   [empty_rec(Mz+2, (Mp+2) + (Ms+2) + (Mn+2))]
                   ])
    J_Tphie = bmat([ [empty_rec(Ma+2, (Mp+2)+(Ms+2)+(Mn+2))],\
                   [block_diag((J_Tphiep, J_Tphies, J_Tphien))],\
                   [empty_rec(Mz+2, (Mp+2) + (Ms+2) + (Mn+2))]
                   ])
    J_Tphis = vstack([
            empty_rec(Ma+2, (Mp+2)+(Mn+2)),
            hstack([J_Tphisp, empty_rec(Mp+2,Mn+2)]),
            empty_rec(Ms+2,(Mp+2)+(Mn+2)),
            hstack([empty_rec(Mn+2,Mp+2), J_Tphisn]),
            empty_rec(Mz+2, (Mp+2)+(Mn+2))
            ])
    
    J_Tj = vstack([
            empty_rec(Ma+2,Mp+Mn),
            hstack([J_Tjp, empty_rec(Mp+2,Mn)]),
            empty_rec(Ms+2, Mp+Mn),
            hstack([empty_rec(Mn+2,Mp), J_Tjn]),
            empty_rec(Mz+2,Mp+Mn)
            ])
    
    J_Teta = vstack([
            empty_rec(Ma+2,Mp+Mn),
            hstack([J_Tetap, empty_rec(Mp+2,Mn)]),
            empty_rec(Ms+2, Mp+Mn),
            hstack([empty_rec(Mn+2,Mp), J_Tetan]),
            empty_rec(Mz+2,Mp+Mn)
            ])
    
    J_TT = vstack([J_TTa, J_TTp, J_TTs, J_TTn, J_TTz])
    
    J_Tc = vstack([J_cp,
                   empty_rec(Ms+2,Mp*(Np+2) + Mn*(Nn+2)),
                   J_cn
                   ])
        
    res_T = np.hstack((bc_T0a,res_Ta,bc_TMa,
                       bc_T0p, res_Tp, bc_TMp,
                       bc_T0s, res_Ts, bc_TMs,
                       bc_T0n, res_Tn, bc_TMn,
                       bc_T0z, res_Tz, bc_TMz))
    
    J_T = hstack([J_Tc,J_Tu, J_TT, J_Tphie, J_Tphis, J_Tj, J_Teta])
    
    return res_T, J_T

#plt.figure(figsize=(20,10)); plt.spy(J_T, markersize=1);plt.savefig('Tsparsity.png')