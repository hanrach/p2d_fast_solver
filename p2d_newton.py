#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:32:07 2020

@author: hanrach
"""
from jax.scipy.linalg import solve
import jax.numpy as np
from numpy.linalg import norm



def newton(fn, jac_fn, U):
    maxit=20
    tol = 1e-7
    count = 0
    res = 100
    fail = 0

    Uold = U
    maxit=5
    while(count < maxit):
        J =  jac_fn(U, Uold)

        y = fn(U,Uold)

        res = norm(y/norm(U,np.inf),np.inf)

        delta = solve(J,y)

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
