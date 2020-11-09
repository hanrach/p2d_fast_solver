#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:53:45 2020

@author: hanrach
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:19:41 2020

@author: hanrach
"""
import jax
import jax.numpy as np
from jax import jacfwd
from jax.config import config
config.update('jax_enable_x64', True)
from jax.numpy.linalg import norm
from jax.scipy.linalg import solve
from p2d_param import get_battery_sections
import timeit
from settings import Tref
from unpack import unpack, unpack_fast
from scipy.sparse import csr_matrix, csc_matrix
from unpack import unpack
import jax
from jax.scipy.linalg import solve
import jax.numpy as np
from jax.numpy.linalg import norm
from jax import jacrev
from jax.config import config
config.update('jax_enable_x64', True)
from p2d_main_fn import p2d_fn
#from res_fn_order2 import fn
from residual import ResidualFunction

Np = 50
Nn = 50
Mp = 50
Ms = 10
Mn = 50
Ma = 5
Mz = 5

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

def newton(fn, jac_fn, U):
    maxit=20
    tol = 1e-8
    count = 0
    res = 100
    fail = 0
    Uold = U
 
    start =timeit.default_timer()     
    J =  jac_fn(U, Uold)
    y = fn(U,Uold)
    res0 = norm(y/norm(U,np.inf),np.inf)
    delta = solve(J,y)
    U = U - delta
    count = count + 1
    end = timeit.default_timer()
    print("time elapsed in first loop", end-start)
    print(count, res0)
    while(count < maxit and  res > tol):
        start1 =timeit.default_timer() 
        J =  jac_fn(U, Uold)
        y = fn(U,Uold)
        res = norm(y/norm(U,np.inf),np.inf)
        delta = solve(J,y)
        U = U - delta
        count = count + 1
        end1 =timeit.default_timer() 
        print(count, res)
        print("time per loop", end1-start1)
        
    if fail ==0 and np.any(np.isnan(delta)):
        fail = 1
        print("nan solution")
        
    if fail == 0 and max(abs(np.imag(delta))) > 0:
            fail = 1
            print("solution complex")
    
    if fail == 0 and res > tol:
        fail = 1;
        print('Newton fail: no convergence')
    else:
        fail == 0 
        
    return U, fail


# Do one newton step        
jac_fn = jax.jit(jacfwd(fn))
fn = jax.jit(fn)


peq, neq, sepq, accq, zccq= get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz)

Ntot_pe = (Np+2)*(Mp) + 4*(Mp + 2) + 2*(Mp)
Ntot_ne = (Nn+2)*(Mn) + 4*(Mn + 2) + 2*(Mn)

Ntot_sep =  3*(Ms + 2)
Ntot_acc =Ma+ 2
Ntot_zcc = Mz+ 2
Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc

U = np.hstack( 
        [
                peq.cavg*np.ones(Mp*(Np+2)), 
                neq.cavg*np.ones(Mn*(Nn+2)),
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
 
[sol, fail] = newton(fn, jac_fn, U)
print("Matrix of size",Ntot**2)

#print("time elapsed", end-start)
#res_c, J_c = jacres_c(U,U, peq, neq, sepq, accq, zccq) 
#res_u, J_u = jacres_u(U,U, peq, neq, sepq, accq, zccq)
#res_T, J_T = jacres_T(U,U, peq, neq, sepq, accq, zccq)