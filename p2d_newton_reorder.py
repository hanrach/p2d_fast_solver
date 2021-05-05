#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:40:30 2020

@author: hanrach
"""
from functools import partial
from jax.scipy.linalg import solve
import jax.numpy as np
from scipy.linalg import solve_banded
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
import numba as nb
#from untitled0 import diagonal_form_jax

#@nb.jit(nopython=True)
#def to_numpy(array):
#    return onp.asarray(array)

@nb.njit()
#@partial(jax.jit, static_argnums=(0,))
def diagonal_form(l_and_u, a):
    """
    a is a numpy square matrix
    this function converts a square matrix to diagonal ordered form
    returned matrix in ab shape which can be used directly for scipy.linalg.solve_banded
    """
    n = a.shape[1]
#    assert(onp.all(a.shape ==(n,n)))
    
    (nlower,nupper)=l_and_u
    diagonal_ordered = onp.zeros((nlower + nupper + 1, n), dtype=a.dtype)
    for i in range(1, nupper + 1):
        for j in range(n - i):
            diagonal_ordered[nupper - i, i + j] = a[j, i + j]
        
    for i in range(n):
        diagonal_ordered[nupper, i] = a[i, i]
    
    for i in range(nlower):
        for j in range(n - i - 1):
            diagonal_ordered[nupper + 1 + i, j] = a[i + j + 1, j]
        
    return diagonal_ordered



@jax.jit
def process_J(J0, y, idx):
    J1 = J0[:,idx]
    J1=J1[idx,:]
    return J1

@jax.jit
def process_y(y,idx):
    y1=y[idx]
    return y1

@jax.jit
def reorder_vec(b, idx):
    return b[idx]



def newton_reorder(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n, idx, re_idx, time_mode):
    maxit=10
    tol = 1e-7
    count = 0
    res = 100
    fail = 0
    Uold = U
#    start = timeit.default_timer()
    linsolve_time_list = []
    diff_time_list=[]
    sp_time_list=[]
    
    diff0_s=timeit.default_timer()
    J =  jac_fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
    y = fn_fast(U,Uold,cs_pe1, cs_ne1, gamma_p, gamma_n)  
    diff0_e = timeit.default_timer()
    diff_time_list.append(diff0_e-diff0_s)
    
    sp0_s = timeit.default_timer()
    J, y= process_vec(J, y, idx); J = onp.asarray(J)
    J = diagonal_form((11,11), J)
    sp0_e = timeit.default_timer()
    sp_time_list.append(sp0_e-sp0_s)
    
    start1 = timeit.default_timer()
    delta = solve_banded((11,11),J,y)
    end1 = timeit.default_timer()
    linsolve_time_list.append(end1-start1)
#    delta = solve(J,y)
    delta_reordered=reorder_vec(delta, re_idx)
    U = U - delta_reordered
    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1


    while(count < maxit and res > tol):
        if (time_mode == "wall time"):
            start_diff = time.monotonic()
            J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            end_diff = time.monotonic()
            diff_time_list.append(end_diff-start_diff)
            
            start_sp = time.monotonic()
            J, y= process_vec(J, y, idx); 
#            J = onp.asarray(J)
            J = diagonal_form((11,11), J)
            end_sp = time.monotonic()
            sp_time_list.append(end_sp - start_sp)

            res = norm(y/norm(U,np.inf),np.inf)
            
            start1 = time.monotonic()
            delta = solve_banded((11,11),J,y)
            end1 = time.monotonic()
            linsolve_time_list.append(end1-start1)
            
            
        if (time_mode == "process time"):
            start_diff = timeit.default_timer()
            J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n)
            end_diff = timeit.default_timer()
            diff_time_list.append(end_diff-start_diff)
            
            start_sp = timeit.default_timer()
            J, y= process_vec(J, y, idx); J = onp.asarray(J)
            J = diagonal_form((11,11), J)
            end_sp = timeit.default_timer()
            sp_time_list.append(end_sp - start_sp)

            res = norm(y/norm(U,np.inf),np.inf)
            
            start1 = timeit.default_timer()
            delta = solve_banded((11,11),J,y)
            end1 = timeit.default_timer()
            linsolve_time_list.append(end1-start1)
        
        delta_reordered = reorder_vec(delta, re_idx)
        U = U - delta_reordered
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


def newton_reorder_short(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n, idx, re_idx):
    maxit=10
    tol = 1e-7
    count = 0
    res = 100
    fail = 0
    Uold = U
#    start = timeit.default_timer()
    start=timeit.default_timer();
    J =  jac_fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    y = fn_fast(U,Uold,cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    end=timeit.default_timer();
    jf_time0=end-start
#    print("Initial J and f evaluation:", end-start);
    
    start=timeit.default_timer();
    J= process_J(J, y, idx).block_until_ready(); J = onp.asarray(J); 
    y=process_y(y,idx).block_until_ready()
    end=timeit.default_timer();
    overhead_r=end-start
#    print("Reordering overhead:", end-start)
    
    start=timeit.default_timer();
    J = diagonal_form((11,11), J)
    end=timeit.default_timer();
    overhead0=overhead_r + (end-start)
#    print("Convert to diagonal form:", end-start)
    
    start=timeit.default_timer();
    delta = solve_banded((11,11),J,y)
    end=timeit.default_timer();
    solve_time0 = end-start
#    print("banded solve time linear solver:", end-start)
#    delta = solve(J,y)
    delta_reordered=reorder_vec(delta, re_idx)
    U = U - delta_reordered
#    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1

    jf_time = jf_time0;
    overhead= overhead0;
    solve_time = solve_time0;
    while(count < maxit and res > tol):


        
        start = timeit.default_timer();
        J =  jac_fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        y = fn_fast(U,Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        end= timeit.default_timer();
        jf_time += end-start;

        start = timeit.default_timer();
        J= process_J(J, y, idx).block_until_ready();
        y = process_y(y,idx).block_until_ready();
        J = onp.asarray(J);  
        J = diagonal_form((11,11), J)
        end = timeit.default_timer();
        overhead += end-start

        res = norm(y/norm(U,np.inf),np.inf)
       
        start = timeit.default_timer();
        delta = solve_banded((11,11),J,y)
        end= timeit.default_timer();
        solve_time += end-start
        
        start=timeit.default_timer()
        delta_reordered = reorder_vec(delta, re_idx).block_until_ready()
        end = timeit.default_timer()
        overhead += end-start
        
        U = U - delta_reordered
        count = count + 1
#        print(count, res)
    
#    print("Total to evaluate Jacobian:",jf_time)
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
        
    info=(fail,jf_time, overhead, solve_time)
    return U, info