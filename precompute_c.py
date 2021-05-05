#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 23:15:36 2020

@author: hanrach
"""

import jax.numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import spsolve
#from jax.scipy.linalg import solve
#peq = ElectrodeEquation(p_electrode_constants(),p_electrode_grid_param(10,10), "positive",25751, 51554)
#neq = ElectrodeEquation(n_electrode_constants(),n_electrode_grid_param(10,10), "negative", 26128, 30555)
#sepq = SeparatorEquation(sep_constants(),sep_grid_param(5))
#accq = CurrentCollectorEquation(a_cc_constants(),cc_grid_param(5))
#zccq = CurrentCollectorEquation(z_cc_constants(),cc_grid_param(5))
#
#Mp = peq.M; Mn = neq.M; Ms = sepq.M; Ma = accq.M; Mz = zccq.M
#Np = peq.N; Nn = neq.N
#hy = peq.hy
#Ds = peq.Ds
##lambda1  = delta_t*peq.rpts_end[0:peq.N]**2/peq.rpts_mid**2/hy**2;
##lambda2  = delta_t*peq.rpts_end[1:peq.N+1]**2/peq.rpts_mid**2/hy**2;
#
##R = np.arange(0,peq.Rp + peq.hy, peq.hy) + peq.hy/2 ; R = R[0:-1]
#R = peq.Rp*(np.linspace(1,Np,Np)-(1/2))/Np;
#r = peq.Rp*(np.linspace(0,Np, Np+1))/Np
#lambda1 = delta_t*r[0:Np]**2/(R**2*peq.hy**2);
#lambda2 = delta_t*r[1:Np+1]**2/(R**2*peq.hy**2);
#        
#hy_n = neq.hy
#Ds_n = neq.Ds
##lambda1_n  = delta_t*neq.rpts_end[0:neq.N]**2/neq.rpts_mid**2/hy_n**2;
##lambda2_n  = delta_t*neq.rpts_end[1:neq.N+1]**2/neq.rpts_mid**2/hy_n**2;
##Rn = np.arange(0,neq.Rp + neq.hy, neq.hy) + neq.hy/2 ; Rn = Rn[0:-1]
#Rn = neq.Rp*(np.linspace(1,Nn,Nn)-(1/2))/Nn;
#rn = neq.Rp*(np.linspace(0,Nn, Nn+1))/Nn
#lambda1_n = delta_t*rn[0:Nn]**2/(Rn**2*neq.hy**2);
#lambda2_n = delta_t*rn[1:Nn+1]**2/(Rn**2*neq.hy**2);
#
#Ucp = peq.cavg*np.ones(Mp*(Np+2)) 
#Ucn = neq.cavg*np.ones(Mn*(Nn+2))
#    
##def linear_c(cn,cc,cp,elec):
##    hy = elec.hy
##    Ds = elec.Ds
##    lambda1  = delta_t*elec.rpts_end[0:elec.N]**2/elec.rpts_mid**2/hy**2;
##    lambda2  = delta_t*elec.rpts_end[1:elec.N+1]**2/elec.rpts_mid**2/hy**2;
##    k = delta_t;
##    N = elec.N
##    R = np.arange(0,elec.Rp + elec.hy, elec.hy) + elec.hy/2 ; R = R[0:-1]
##    r = elec.Rp*(np.linspace(0,N, N+1))/N
##    lambda1 = k*r[0:N]**2/(R**2*hy**2);
##    lambda2 = k*r[1:N+1]**2/(R**2*hy**2);
##    ans = (cc)  + Ds*( cc*(lambda2 + lambda1) - lambda2*cp - lambda1*cn)
##    return ans
#
## automatic jacobian
##def fn_c_pe(Uc):
##    cmat_pe = np.reshape(Uc, [Np+2, Mp], order="F") 
##    val = np.zeros(Mp*(Np+2))
##    for i in range(0,Mp):
##        val = jax.ops.index_update(val, jax.ops.index[i*(Np+2)], peq.bc_zero_neumann(cmat_pe[0,i], cmat_pe[1,i]) )
##        val = jax.ops.index_update(val, jax.ops.index[i*(Np+2) + Np+1], peq.bc_zero_neumann(cmat_pe[Np,i], cmat_pe[Np+1,i])/hy) 
##        res_c = linear_c(cmat_pe[0:Np,i], cmat_pe[1:Np+1,i], cmat_pe[2:Np+2,i],peq)
##        val = jax.ops.index_update(val, jax.ops.index[i*(Np+2)+1 : Np+1 + i*(Np+2) ], res_c)
##    return val
##
##y = fn_c_pe(Ucp)
##J = jacfwd(fn_c_pe)(Ucp)
##
##
#
## manual Jacobian
#row = np.hstack([0,0,np.arange(1,Np+1,1),np.arange(1,Np+1,1),np.arange(1,Np+1,1),Np+1,Np+1])
#col=np.hstack([0,1,np.arange(1,Np+1,1),np.arange(1,Np+1,1)-1,np.arange(1,Np+1,1)+1,Np,Np+1])
#data = np.hstack([-1,1,
#        1+Ds*(lambda1+lambda2),
#        -Ds*lambda1,
#        -Ds*lambda2,
#        -1/hy,1/hy]);
#    
#row_n = np.hstack([0,0,np.arange(1,Nn+1,1),np.arange(1,Nn+1,1),np.arange(1,Nn+1,1),Nn+1,Nn+1])
#col_n=np.hstack([0,1,np.arange(1,Nn+1,1),np.arange(1,Nn+1,1)-1,np.arange(1,Nn+1,1)+1,Nn,Nn+1])
#data_n = np.hstack([-1,1,
#         1+Ds_n*(lambda1_n+lambda2_n),
#         -Ds_n*lambda1_n,
#        -Ds_n*lambda2_n,
#        -1/hy_n,1/hy_n]);
#    
##J1 = J[0:Np+2, 0:Np+2]
#Ape = csr_matrix((data, (row, col)))
#Ane = csr_matrix((data_n, (row_n, col_n)))
#Ap = kron( identity(Mp), Ape)
#An = kron(identity(Mn), Ane)
#vec_p = np.hstack([np.zeros(Np+1), 1])
#vec_n = np.hstack([np.zeros(Nn+1), 1])
#temp_p = spsolve(Ape,vec_p); gamma_p = (temp_p[Np] + temp_p[Np+1])/2
#temp_n = spsolve(Ane,vec_n); gamma_n = (temp_n[Nn] + temp_n[Nn+1])/2

def precompute(peq,neq):
    Mp = peq.M; Mn = neq.M; 
    Np = peq.N; Nn = neq.N
    hy = peq.hy
    Ds = peq.Ds;
    delta_t=peq.delta_t
    
    #R = np.arange(0,peq.Rp + peq.hy, peq.hy) + peq.hy/2 ; R = R[0:-1]
    R = peq.Rp*(np.linspace(1,Np,Np)-(1/2))/Np;
    r = peq.Rp*(np.linspace(0,Np, Np+1))/Np
    lambda1 = delta_t*r[0:Np]**2/(R**2*peq.hy**2);
    lambda2 = delta_t*r[1:Np+1]**2/(R**2*peq.hy**2);
            
    hy_n = neq.hy
    Ds_n = neq.Ds
    Rn = neq.Rp*(np.linspace(1,Nn,Nn)-(1/2))/Nn;
    rn = neq.Rp*(np.linspace(0,Nn, Nn+1))/Nn
    lambda1_n = delta_t*rn[0:Nn]**2/(Rn**2*neq.hy**2);
    lambda2_n = delta_t*rn[1:Nn+1]**2/(Rn**2*neq.hy**2);

    row = np.hstack([0,0,np.arange(1,Np+1,1),np.arange(1,Np+1,1),np.arange(1,Np+1,1),Np+1,Np+1])
    col=np.hstack([0,1,np.arange(1,Np+1,1),np.arange(1,Np+1,1)-1,np.arange(1,Np+1,1)+1,Np,Np+1])
    data = np.hstack([-1,1,
            1+Ds*(lambda1+lambda2),
            -Ds*lambda1,
            -Ds*lambda2,
            -1/hy,1/hy]);
        
    row_n = np.hstack([0,0,np.arange(1,Nn+1,1),np.arange(1,Nn+1,1),np.arange(1,Nn+1,1),Nn+1,Nn+1])
    col_n=np.hstack([0,1,np.arange(1,Nn+1,1),np.arange(1,Nn+1,1)-1,np.arange(1,Nn+1,1)+1,Nn,Nn+1])
    data_n = np.hstack([-1,1,
             1+Ds_n*(lambda1_n+lambda2_n),
             -Ds_n*lambda1_n,
            -Ds_n*lambda2_n,
            -1/hy_n,1/hy_n]);
        
    #J1 = J[0:Np+2, 0:Np+2]
    Ape = csr_matrix((data, (row, col)))
    Ane = csr_matrix((data_n, (row_n, col_n)))
    Ap = kron( identity(Mp), Ape)
    An = kron(identity(Mn), Ane)
    vec_p = np.hstack([np.zeros(Np+1), 1])
    vec_n = np.hstack([np.zeros(Nn+1), 1])
    temp_p = spsolve(Ape,vec_p); gamma_p = (temp_p[Np] + temp_p[Np+1])/2
    temp_n = spsolve(Ane,vec_n); gamma_n = (temp_n[Nn] + temp_n[Nn+1])/2
    temp_p_mat = np.tile(temp_p,(Mp,1)).transpose()
    temp_n_mat = np.tile(temp_n,(Mn,1)).transpose()
    
    return Ap, An, gamma_n, gamma_p, temp_p, temp_n