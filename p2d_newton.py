#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:32:07 2020

@author: hanrach
"""
from jax.scipy.linalg import solve
import jax.numpy as np
from jax.numpy.linalg import norm
from jax import lax
import collections
import jax
import timeit
from scipy.sparse import csr_matrix, csc_matrix
from  scipy.sparse.linalg import spsolve, splu    
def newton(fn, jac_fn, U):
    maxit=20
    tol = 1e-8
    count = 0
    res = 100
    fail = 0

    Uold = U
    maxit=5
#    
#    @jax.jit
#    def body_fun(U,Uold):
#        J =  jac_fn(U, Uold)
#        y = fn(U,Uold)
#        res = norm(y/norm(U,np.inf),np.inf)
#        delta = solve(J,y)
#        U = U - delta
#        return U, res
#   
    print("here")
    start =timeit.default_timer()     
    J =  jac_fn(U, Uold)
    print("computed jacobian")
    y = fn(U,Uold)
    res0 = norm(y/norm(U,np.inf),np.inf)
    delta = solve(J,y)
    U = U - delta
    count = count + 1
    end = timeit.default_timer()
    print("time elapsed in first loop", end-start)
    print(count, res0)
    while(count < maxit and  res > tol):
#        U, res, delta = body_fun(U,Uold)\
        start1 =timeit.default_timer() 
        J =  jac_fn(U, Uold)
        y = fn(U,Uold)
        res = norm(y/norm(U,np.inf),np.inf)
        delta = solve(J,y)
        U = U - delta
        count = count + 1
        end1 =timeit.default_timer() 
        print("time per loop", end1-start1)
        print(count, res)
    
        
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


def newton_while_lax(fn, jac_fn, U, maxit, tol):
    
    
    count = 0
    res = 100
    fail = 0
    
    val = (U, count, res, fail)

    Uold = U
    J =  jac_fn(U, Uold)
    y = fn(U,Uold)  
    delta = solve(J,y)
    U = U - delta;
#    res0 = norm(y/norm(U,np.inf),np.inf)
    
   
    def cond_fun(val):
        U, count, res, _ = val
        res = norm(y/norm(U,np.inf),np.inf)
        print("res:",res)
        return np.logical_and(res > tol, count < maxit)
#    
   
    def body_fun(val):
        U, count, res, fail = val
        J = jac_fn(U,Uold);
        y = fn(U,Uold)
        delta = solve(J,y)
        U = U - delta
        res = norm(y/norm(U,np.inf),np.inf)
        count = count + 1
        print(count, res)
        val = U, count, res, fail
      
        return val
    
    val =lax.while_loop(cond_fun, body_fun, val )
    U, count, res, _ = val
   
        
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

def damped_newton(fn, jac_fn, U):
    maxit=10
    tol = 1e-8
    count = 0
    res = 100
    fail = 0
#    U = jax.ops.index_update(U, jax.ops.index[jp02:etap02], U[jp02:etap02]*2**(-16))
    Uold = U
    J =  jac_fn(U, Uold)
    y = fn(U,Uold)  
    delta = solve(J,y)
    U = U - delta;
    res0 = norm(y/norm(U,np.inf),np.inf)
    print(count, res0)
    while(count < maxit and res > tol):
        J =  jac_fn(U, Uold)
        y = fn(U,Uold)        
        res = norm(y/norm(U,np.inf),np.inf)
#        res=norm(y, np.inf)
        print(count, res)
        delta = solve(J,y)
        
#        alpha = 1.0
#        while (norm( fn(U - alpha*delta,Uold )) > (1-alpha*0.5)*norm(y)):
##            print("norm1",norm( fn(U - alpha*delta,Uold )))
##            print("norm2", (1-alpha*0.5)*norm(y) )
#            alpha = alpha/2;
##            print("alpha",alpha)
#            if (alpha < 1e-8):
#                break;
#                
#        U = U - alpha*delta
        U = U - delta;
        count = count + 1
    
        
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

def newton_tol(fn, jac_fn, U,tol):
    maxit=20
    count = 0
    res = 100
    fail = 0
    Uold = U
    
    while(count < maxit and res > tol):
        J =  jac_fn(U, Uold)
#        J = jacrev(fn)(U,Uold)
#        Jsparse = csr_matrix(J)
        y = fn(U,Uold)
        res = max(abs(y/norm(y,2)))
        print(count, res)
        delta = solve(J,y)
#        delta = jitsolve(J,fn(U, Uold))
#        delta = spsolve(csr_matrix(J),fn(U,Uold))
        U = U - delta
        count = count + 1
    
        
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
