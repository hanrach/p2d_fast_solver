 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:25:53 2020

@author: hanrach
"""
import jax
import numpy as onp
from jax.config import config
config.update('jax_enable_x64', True)
from init import p2d_init_slow, p2d_init_fast
from p2d_newton import newton
from p2d_newton_fast import  newton_fast2


import timeit
from scipy.linalg import solve_banded
from p2d_param import get_battery_sections
from precompute_c import precompute
import matplotlib.pylab as plt
import matplotlib.font_manager
from  scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csr_matrix, csc_matrix
import jax.numpy as np
from settings import Tref
from reorder import reorder_tot
from banded_matrix import diagonal_form
from unpack import unpack_fast
Np = 5
Nn = 5
Mp = 5
Ms = 5
Mn = 5
Ma = 5
Mz = 5
#
#i=0
peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz, 10)
Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq,neq)
gamma_p_vec  = gamma_p*np.ones(Mp)
gamma_n_vec = gamma_n*np.ones(Mn)
lu_p = splu(csc_matrix(Ap))
lu_n = splu(csc_matrix(An))

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

c_pe = peq.cavg*np.ones(Mp*(Np+2))
c_ne = neq.cavg*np.ones(Mn*(Nn+2))

lu = {"pe":lu_p, "ne": lu_n}
def cmat_format(cmat, M, N):
    val = cmat
    for i in range(0,M):
        val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
        val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)
    return val
    
cmat_rhs_pe = cmat_format(c_pe, Mp, Np)
cmat_rhs_ne = cmat_format(c_ne,Mn, Nn)
lu_pe = lu["pe"]; lu_ne = lu["ne"]
cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
cI_pe_reshape = np.reshape(cI_pe_vec, [Np+2, Mp], order="F"); cs_pe1 = np.asarray((cI_pe_reshape[Np,:] + cI_pe_reshape[Np+1,:])/2)
cI_ne_reshape = np.reshape(cI_ne_vec, [Nn+2, Mn], order="F"); cs_ne1 = np.asarray((cI_ne_reshape[Nn,:] + cI_ne_reshape[Nn+1,:])/2)


fn_fast, jac_fn_fast = p2d_init_fast(Np, Nn, Mp, Mn, Ms, Ma,Mz, 10)

U_fast_new, fail, time, eval_time, sp_time= newton_fast2(fn_fast, jac_fn_fast, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec, "wall time")
Jfast1=jac_fn_fast(U_fast_new,U_fast,cs_pe1,cs_ne1,gamma_p_vec, gamma_n_vec)
yfast1=fn_fast(U_fast_new,U_fast,cs_pe1,cs_ne1,gamma_p_vec, gamma_n_vec)

    
#Unew, fail, time, eval_time_slow, sp_time_slow = newton(fn, jac_fn, U, "wall time")
#Jslow = jac_fn(Unew, U)
#yslow = fn(Unew, U)
    


idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
Jreorder=np.zeros([len(U_fast), len(U_fast)])
Jreorder = Jfast1[:,idx_tot]
Jreorder = Jreorder[idx_tot,:]

Jreorder_onp = onp.asarray(Jreorder)
Jab = diagonal_form((11,11), Jreorder_onp)
idx_original=np.argsort(idx_tot)
yfast_reorder =  yfast1[idx_tot]
yab=solve_banded((11,11), Jab,yfast_reorder)


uvec_pe, uvec_sep, uvec_ne, \
            Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
            phie_pe, phie_sep, phie_ne, \
            phis_pe, phis_ne, jvec_pe,jvec_ne,eta_pe,eta_ne = unpack_fast(U_fast_new, Mp, Np, Mn, Nn, Ms, Ma, Mz)
uvec_old_pe, uvec_old_sep, uvec_old_ne, \
            Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc, \
            phie_old_pe, phie_old_sep, phie_old_ne, \
            phis_old_pe, phis_old_ne, jvec_old_pe,jvec_old_ne,eta_old_pe,eta_old_ne = unpack_fast(U_fast, Mp, Np, Mn, Nn, Ms, Ma, Mz)

arg_up = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], jvec_pe[0:Mp],uvec_old_pe[1:Mp+1]]
dup = jax.vmap( jax.grad(peq.electrolyte_conc, argnums=range(0,8)) )(*arg_up)

def find_indices(dgrad):
    dup_idx= {}
    for (i,partials) in enumerate(dgrad):
        idx = []
        for (derivative) in (partials):
            temp = np.where(np.isclose(derivative,Jab, atol=1e-16))
            idx.append((temp[0],temp[1]))
        dup_idx[i] = idx
    return dup_idx

dup_idx = find_indices(dup)
    
    
#ionic_flux_fast(self,j,u,T,eta,cs1, gamma_c,cmax)

    
    
arg_jp = [jvec_pe[0:Mp], uvec_pe[1:Mp+1], Tvec_pe[1:Mp+1], eta_pe[0:Mp],  cs_pe1, gamma_p_vec, peq.cmax*np.ones([Mp,1])]
djp = jax.vmap( jax.grad(peq.ionic_flux_fast, argnums=range(0,4) ))(*arg_jp)
djp_idx = find_indices(djp)

arg_etap = [eta_pe[0:Mp], phis_pe[1:Mp+1],phie_pe[1:Mp+1], Tvec_pe[1:Mp+1], jvec_pe[0:Mp], cs_pe1, gamma_p_vec, peq.cmax*np.ones([Mp,1])]
detap= jax.vmap( jax.grad(peq.over_poten_fast, argnums=range(0,5)))(*arg_etap)
detap_idx = find_indices(detap)

arg_phisp =  [phis_pe[0:Mp], phis_pe[1:Mp+1], phis_pe[2:Mp+2], jvec_pe[0:Mp]]
dphisp = jax.vmap ( jax.grad(peq.solid_poten, argnums=range(0,len(arg_phisp))))(*arg_phisp)
dphisp_idx = find_indices(dphisp)

arg_phiep = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], phie_pe[0:Mp], phie_pe[1:Mp+1], phie_pe[2:Mp+2],
                  Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], jvec_pe[0:Mp]]
dphiep = jax.vmap(jax.grad(peq.electrolyte_poten, argnums=range(0, len(arg_phiep))))(*arg_phiep)
dphiep_idx = find_indices(dphiep)

#temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
arg_Tp = arg_Tp = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2],
              phie_pe[0:Mp],phie_pe[2:Mp+2],\
             phis_pe[0:Mp], phis_pe[2:Mp+2],
             Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], 
             jvec_pe[0:Mp],\
             eta_pe[0:Mp], 
             cs_pe1,gamma_p_vec,peq.cmax*np.ones([Mp,1]), 
             Tvec_old_pe[1:Mp+1]]

dTp = jax.vmap( jax.grad(peq.temperature_fast, argnums=range(0,12) ))(*arg_Tp)
dTp_idx = find_indices(dTp)

Jnew = np.zeros([23, len(U_fast)])


# Ma + 2 + 4 + 6*i, i=1, ... , Mp -1 

from jax import lax
@jax.jit
def array_update(state,update_element):
    element, ind = update_element
    J, start_index, row = state
    return (jax.ops.index_update(J, jax.ops.index[row,start_index+ 6*ind], element), start_index, row), ind


@jax.jit
def build_dudT(J, partial):
    ranger = np.arange(1,Mp)
#    ranger = Ma+1 + 5+5 + 6*ind
    start_index = Ma + 1 + 10; row=0;
    state = (J, start_index, row)
    J,_,_ = lax.scan(array_update, state, (partial[0:-1], ranger))[0]
    J = jax.ops.index_update(J,jax.ops.index[2, Ma+1 + 5 + 6*Mp +3],partial[-1])
    return J

@jax.jit
def array_update2(J, update_element):
        element, ind = update_element
        return jax.ops.index_update(J, jax.ops.index[17,Ma + 1 + 5 + 6 + 6*ind], element), ind

@jax.jit
def build_dudum(J, partial):
    ranger = np.arange(1,Mp)
    return lax.scan(array_update2, J, (partial, ranger))[0]


@jax.jit
def array_update_dudT(J,update_element):
    element, ind = update_element
    return jax.ops.index_update(J, jax.ops.index[0,Ma+1 + 5+5 + 6*ind], element), ind

@jax.jit
def array_update_dudum(J, update_element):
        element, ind = update_element
        return jax.ops.index_update(J, jax.ops.index[17,Ma + 1 + 5 + 6 + 6*ind], element), ind

def lax_scan_update(fn, state, xs):
    J,_,_ = lax.scan(array_update, state, xs)[0]
    return J

@jax.jit
def build_dup(J,partial):
    ranger_p = np.arange(1,Mp)
    # dudTp
    J,_,_ = lax.scan(array_update, (J,Ma + 1 + 10, 0), (partial[5][0:-1], ranger_p))[0]
    J = jax.ops.index_update(J,jax.ops.index[2, Ma+1 + 5 + 6*Mp +3],partial[5][-1])
    #dudum
    ranger_m = np.arange(0,Mp-1)
    J,_,_ = lax.scan(array_update, (J, Ma + 1 + 5, 17), (partial[0][1:Mp], ranger_m))[0]
    J =  jax.ops.index_update(J,jax.ops.index[16, Ma+1],partial[0][0])
    #duduc
    ranger_c = np.arange(0,Mp)
    J,_,_ = lax.scan(array_update, (J, Ma + 1 + 5, 11), (partial[1][0:Mp], ranger_c))[0]
    #dudup
    J,_,_= lax.scan(array_update, (J, Ma + 1 + 5 + 6, 12), (partial[2][0:Mp], ranger_c))[0]
    #dudTm
    J,_,_ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 12), (partial[3][1:Mp], ranger_m) )[0]
    # dup[3][0] not assigned
    
    #dudTc
    J,_,_ = lax.scan(array_update, (J, Ma + 1 + 5 + 6,6), (partial[4][0:Mp], ranger_c))[0]
    
    #dudj
    J,_,_ = lax.scan(array_update, (J, Ma + 1 + 5 + 1,10), (partial[6][0:Mp], ranger_c))[0]
    return J
    
@jax.jit
def build_djp(J, partial):
    ranger_c = np.arange(0,Mp)
    J,_,_ = lax.scan(array_update, (J,Ma + 1 + 5 + 1,11 ), (partial[0][0:Mp], ranger_c))[0]
    J,_,_ = lax.scan(array_update, (J,Ma + 1 + 5 , 12 ), (partial[1][0:Mp], ranger_c))[0]
    J,_,_ = lax.scan(array_update, (J,Ma + 1 + 5 + 5, 7 ), (partial[0][0:Mp], ranger_c))[0]
    return J

@jax.jit
def build_detap(J,partial):
    
    return J
@jax.jit
def compute_der(U):
    Jnew = np.zeros([23, len(U_fast)])

    uvec_pe, uvec_sep, uvec_ne, \
            Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
            phie_pe, phie_sep, phie_ne, \
            phis_pe, phis_ne, jvec_pe,jvec_ne,eta_pe,eta_ne = unpack_fast(U_fast_new, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    uvec_old_pe, uvec_old_sep, uvec_old_ne, \
            Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc, \
            phie_old_pe, phie_old_sep, phie_old_ne, \
            phis_old_pe, phis_old_ne, jvec_old_pe,jvec_old_ne,eta_old_pe,eta_old_ne = unpack_fast(U_fast, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    
    bc_u0p = peq.bc_zero_neumann(uvec_pe[0],uvec_pe[1])
    arg_up = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], jvec_pe[0:Mp],uvec_old_pe[1:Mp+1]]
    dup = jax.vmap( jax.grad(peq.electrolyte_conc, argnums=range(0,8)) )(*arg_up)
    bc_uMp = peq.bc_u_sep_p(uvec_pe[Mp], uvec_pe[Mp+1], Tvec_pe[Mp], Tvec_pe[Mp+1], uvec_sep[0],uvec_sep[1],Tvec_sep[0],Tvec_sep[1])
    
    Jnew = build_dup(Jnew,dup)
    
    arg_jp = [jvec_pe[0:Mp], uvec_pe[1:Mp+1], Tvec_pe[1:Mp+1], eta_pe[0:Mp],  cs_pe1, gamma_p_vec, peq.cmax*np.ones([Mp,1])]
    djp= jax.vmap( jax.grad(peq.ionic_flux_fast, argnums=range(0,4) ))(*arg_jp)
    
    Jnew = build_djp(Jnew, djp)
    
    arg_etap = [eta_pe[0:Mp], phis_pe[1:Mp+1],phie_pe[1:Mp+1], Tvec_pe[1:Mp+1], jvec_pe[0:Mp], cs_pe1, gamma_p_vec, peq.cmax*np.ones([Mp,1])]
    detap = jax.vmap( jax.grad(peq.over_poten_fast, argnums=range(0,5)))(*arg_etap)

    arg_phisp =  [phis_pe[0:Mp], phis_pe[1:Mp+1], phis_pe[2:Mp+2], jvec_pe[0:Mp]]
    dphisp = jax.vmap ( jax.grad(peq.solid_poten, argnums=range(0,len(arg_phisp))))(*arg_phisp)

    
    arg_phiep = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2], phie_pe[0:Mp], phie_pe[1:Mp+1], phie_pe[2:Mp+2],
                      Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], jvec_pe[0:Mp]]
    dphiep = jax.vmap(jax.grad(peq.electrolyte_poten, argnums=range(0, len(arg_phiep))))(*arg_phiep)

    
    #temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
    arg_Tp = arg_Tp = [uvec_pe[0:Mp], uvec_pe[1:Mp+1], uvec_pe[2:Mp+2],
                  phie_pe[0:Mp],phie_pe[2:Mp+2],\
                 phis_pe[0:Mp], phis_pe[2:Mp+2],
                 Tvec_pe[0:Mp], Tvec_pe[1:Mp+1], Tvec_pe[2:Mp+2], 
                 jvec_pe[0:Mp],\
                 eta_pe[0:Mp], 
                 cs_pe1,gamma_p_vec,peq.cmax*np.ones([Mp,1]), 
                 Tvec_old_pe[1:Mp+1]]
    
    dTp = jax.vmap( jax.grad(peq.temperature_fast, argnums=range(0,12) ))(*arg_Tp)
    return Jnew        


