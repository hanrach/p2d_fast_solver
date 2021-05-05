#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:08:59 2020

@author: hanrach
"""
from functools import partial
from jax.scipy.linalg import solve
import jax.numpy as np
#from scipy.linalg import solve
from jax.numpy.linalg import norm
import jax
from scipy.sparse import csr_matrix, csc_matrix
#from  scipy.sparse.linalg import  splu   
from scikits.umfpack import spsolve
from jax import vmap
import numpy as onp
import coeffs
from unpack import unpack_fast
import timeit
import time
import numpy as onp

def cmat_format(cmat, M, N):
    val = cmat
    for i in range(0,M):
        val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
        val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)
    return val

def form_c2_p( temp, j, T, N, M, elec):
    # temp is N+2 by M
    val = np.zeros([N+2,M])
    for i in range(0,M):
        Deff = coeffs.solidDiffCoeff(elec.Ds,elec.ED,T[i+1])
        val = jax.ops.index_update(val, jax.ops.index[:,i], -(j[i]*temp[:,i]/Deff))
    return val

#def form_c2_n( temp, j, T, N, M, neq):
#    # temp is N+2 by M
#    val = np.zeros([N+2,M])
#    for i in range(0,M):
#        Deff = coeffs.solidDiffCoeff(neq.Ds,neq.ED,T[i+1])
#        val = jax.ops.index_update(val, jax.ops.index[:,i], -(j[i]*temp[:,i]/Deff))
#    return val

def form_c1_p(cI, T, N, M, elec):
    val = np.zeros([N+2,M])
    
    for i in range(0,M):
        Deff = coeffs.solidDiffCoeff(elec.Ds,elec.ED,T[i+1])
        val = jax.ops.index_update(val, jax.ops.index[:,i], (Deff*cI[:,i]))
    return val
#
#def form_c1_n(cI, T, N, M):
#    val = np.zeros([N+2,M])
#    
#    for i in range(0,M):
#        Deff = coeffs.solidDiffCoeff(neq.Ds,neq.ED,T[i+1])
#        val = jax.ops.index_update(val, jax.ops.index[:,i], (Deff*cI[:,i]))
#    return val

#def form_c_p(c, T, N, M):
#    val = np.zeros([N+2,M])
#    
#    for i in range(0,M):
#        Deff = coeffs.solidDiffCoeff(peq.Ds,peq.ED,T[i+1])
#        val = jax.ops.index_update(val, jax.ops.index[:,i], (c[:,i]/Deff))
#    return val
#    


def newton_fast(fn_fast, jac_fn_fast, U, cmat_pe, cmat_ne, lu, gamma_p, gamma_n, temp_p, temp_n, Mp, Np, Mn, Nn, Ms, Ma, Mz, peq, neq):
    maxit=10
    tol = 1e-7
    count = 0
    res = 100
    fail = 0
#    U = jax.ops.index_update(U, jax.ops.index[jp0:etap0], U[jp0:etap0]*2**(-16))
    Uold = U
    cmat_rhs_pe = cmat_format(cmat_pe, Mp, Np)
    cmat_rhs_ne = cmat_format(cmat_ne,Mn, Nn)
    lu_pe = lu["pe"]; lu_ne = lu["ne"]
    start0=time.time()
    cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
    cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
    end0=time.time()
    cI_pe_reshape = np.reshape(cI_pe_vec, [Np+2, Mp], order="F"); cs_pe1 = np.asarray((cI_pe_reshape[Np,:] + cI_pe_reshape[Np+1,:])/2)
    cI_ne_reshape = np.reshape(cI_ne_vec, [Nn+2, Mn], order="F"); cs_ne1 = np.asarray((cI_ne_reshape[Nn,:] + cI_ne_reshape[Nn+1,:])/2)
    
    temp_p_mat = np.tile(temp_p,(Mp,1)).transpose()
    temp_n_mat = np.tile(temp_n,(Mn,1)).transpose()
    start = timeit.default_timer()
    J =  jac_fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
    y = fn_fast(U,Uold,cs_pe1, cs_ne1, gamma_p, gamma_n)  
    delta = solve(J,y)
    U = U - delta;
    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1
    end = timeit.default_timer()
#    print("time elapsed first loop:", end-start)
#    print(count, res0)
#    newton_time_list = []
    while(count < maxit and res > tol):
        start1 = time.time()
        J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
        y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
        end1 = time.time()
        res = norm(y/norm(U,np.inf),np.inf)
#        start1 = timeit.default_timer()
        delta = solve(J,y)
#        end1 = timeit.default_timer()
        U = U - delta
        
        count = count + 1
#        print(count, res)
#        print("time per loop", end1-start1)
#        newton_time_list.append(end1-start1)
        
#    avg_time=sum(newton_time_list)/len(newton_time_list)
    avg_time=(end1-start1) + (end0-start0)
    start_time=end-start
#    start_time=0
    if fail ==0 and np.any(np.isnan(delta)):
        fail = 1
#        print("nan solution")
        
    if fail == 0 and max(abs(np.imag(delta))) > 0:
            fail = 1
#            print("solution complex")
    
    if fail == 0 and res > tol:
        fail = 1;
#        print('Newton fail: no convergence')
    else:
        fail == 0 
        
    uvec_pe, uvec_sep, uvec_ne, \
    Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
    phie_pe, phie_sep, phie_ne, \
    phis_pe, phis_ne, jvec_pe,jvec_ne,eta_pe,eta_ne = unpack_fast(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    cII_p =  form_c2_p( temp_p_mat, jvec_pe, Tvec_pe, Np, Mp, peq)
    cII_n = form_c2_p( temp_n_mat, jvec_ne, Tvec_ne, Nn, Mn,neq)
#    cI_p = form_c1_p(cI_pe_reshape,Tvec_pe,Np,Mp)
#    cI_n = form_c1_n(cI_ne_reshape,Tvec_ne,Nn,Mn)
#    cmat_pe = np.reshape(cI_p + cII_p, [Mp*(Np+2)], order="F")
#    cmat_ne = np.reshape(cI_n + cII_n, [Mn*(Nn+2)], order="F")
    cmat_pe = np.reshape(cI_pe_reshape + cII_p, [Mp*(Np+2)], order="F")
    cmat_ne = np.reshape(cI_ne_reshape + cII_n, [Mn*(Nn+2)], order="F")
    
    return U,cmat_pe, cmat_ne, fail, avg_time


def newton_fast2(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n, time_mode):
    maxit=10
    tol = 1e-7
    count = 0
    res = 100
    fail = 0
    Uold = U
#    start = timeit.default_timer()
    linsolve_time_list = []
    diff_time_list=[]
    sp_time_list=[0]
    
    diff0s=timeit.default_timer()
    J =  jac_fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
    y = fn_fast(U,Uold,cs_pe1, cs_ne1, gamma_p, gamma_n)  
    diff0e=timeit.default_timer()
    diff_time_list.append(diff0e-diff0s)
    
    lin0s=timeit.default_timer()
    delta = solve(J,y)
    lin0e=timeit.default_timer()
    linsolve_time_list.append(lin0e-lin0s)
    
    U = U - delta;
    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1

    while(count < maxit and res > tol):
        if (time_mode == "jax time"):
            start_diff = timeit.default_timer()
            J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            end_diff = timeit.default_timer()
            diff_time_list.append(end_diff-start_diff)
            
            
            res = norm(y/norm(U,np.inf),np.inf)
            
            start1 = timeit.default_timer()
            delta = solve(J,y)
            end1 = timeit.default_timer()
            linsolve_time_list.append(end1-start1)
            
        if (time_mode == "wall time"):
            start_diff = time.monotonic()
            J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            end_diff = time.monotonic()
            diff_time_list.append(end_diff-start_diff)
            
            start_sp = time.monotonic()
            Jsparse=csr_matrix(J)
            end_sp = time.monotonic()
            sp_time_list.append(end_sp - start_sp)

            res = norm(y/norm(U,np.inf),np.inf)
            
            start1 = time.monotonic()
            delta = spsolve(Jsparse,y)
            end1 = time.monotonic()
            linsolve_time_list.append(end1-start1)
            
        if (time_mode == "process time"):
            start_diff = timeit.default_timer()
            J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            end_diff = timeit.default_timer()
            diff_time_list.append(end_diff-start_diff)
            
            start_sp = timeit.default_timer()
            Jsparse=csr_matrix(J)
            end_sp = timeit.default_timer()
            sp_time_list.append(end_sp - start_sp)

            res = norm(y/norm(U,np.inf),np.inf)
            
            start1 = timeit.default_timer()
            delta = spsolve(Jsparse,y)
            end1 = timeit.default_timer()
            linsolve_time_list.append(end1-start1)
            
        U = U - delta
        count = count + 1
#        print(count, res)
#        print("time per loop", end1-start1)
        
    avg_time=sum(linsolve_time_list)
    diff_time=sum(diff_time_list)
    sparsify_time = sum(sp_time_list)

    if fail ==0 and np.any(np.isnan(delta)):
        fail = 1
#        print("nan solution")
        
    if fail == 0 and max(abs(np.imag(delta))) > 0:
            fail = 1
#            print("solution complex")
    
    if fail == 0 and res > tol:
        fail = 1;
#        print('Newton fail: no convergence')
    else:
        fail == 0 
        
    
    return U, fail, avg_time, diff_time, sparsify_time


def newton_fast_dense(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n):
    maxit=10
    tol = 1e-7
    count = 0
    res = 100
    fail = 0
    Uold = U
    
    start = timeit.default_timer()
    J =  jac_fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    y = fn_fast(U,Uold,cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    end = timeit.default_timer();
    jf_time0 = end-start
    
    start=timeit.default_timer()
    Jsparse=csr_matrix(J)
    end=timeit.default_timer()
    overhead0= end-start
    
    start=timeit.default_timer()
#    delta = onp.linalg.solve(J,y)
    delta = spsolve(Jsparse,y)
    end = timeit.default_timer()
    
    U = U - delta;
    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1
    
    solve_time = end-start;
    overhead = overhead0
    jf_time=jf_time0
    while(count < maxit and res > tol):
                 
        start=timeit.default_timer()
        J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        end=timeit.default_timer()
        jf_time += end-start


        start=timeit.default_timer()
        Jsparse=csr_matrix(J)
        end = timeit.default_timer()
        overhead += end-start
        
        res = norm(y/norm(U,np.inf),np.inf)
        
        
        
        start=timeit.default_timer()
#        delta = onp.linalg.solve(J,y)
        delta = spsolve(Jsparse,y)
        end= timeit.default_timer()
        solve_time += end-start;

        U = U - delta
        count = count + 1
#        print(count, res)
#        print("time per loop", end1-start1)
        

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
        
    info=(fail, solve_time, overhead, jf_time)
    return U, info

def newton_fast_short(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n):
    maxit=10
    tol = 1e-7
    count = 0
    res = 100
    fail = 0
    Uold = U
    
    start1 = timeit.default_timer()
    J =  jac_fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    y = fn_fast(U,Uold,cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    end1 = timeit.default_timer()
    jftime0=end1-start1
   
    start=timeit.default_timer()
    delta = solve(J,y).block_until_ready()
    end=timeit.default_timer()
    solve0=end-start
    
    U = U - delta;
    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1

    solve_time = solve0
    jf_time = jftime0
    while(count < maxit and res > tol):
                 
        start1=timeit.default_timer()
        J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        end1=timeit.default_timer()
        jf_time += ( end1-start1);

        res = norm(y/norm(U,np.inf),np.inf)

        start=timeit.default_timer()
        delta = solve(J,y).block_until_ready()
        end=timeit.default_timer()
        solve_time += end-start;

        U = U - delta
        count = count + 1
#        print(count, res)
#        print("time per loop", end1-start1)
        

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
        
    info = (fail, solve_time, jf_time)
    return U, info



