#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:04:49 2020

@author: hanrach
"""
#from __future__ import print_function
from jax.scipy.linalg import solve
from jax.experimental import host_callback
import jax.numpy as np
from jax.numpy.linalg import norm
from jax import lax, jacfwd
import collections
import jax
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
NewtonInfo = collections.namedtuple(
        'NewtonInfo', [
                'count',
                'converged',
                'fail',
                'U'])

def tap_fn(count):
    return print("count",count)
    
def lax_newton(fn, jac_fn, U, maxit, tol):
    
    Uold = U
    state = NewtonInfo(count = 0,
                       converged = 0,
                       fail = 0,
                       U = U)

    
#    jac_fn = jacfwd(fn)
    def body(state):
        J =  jac_fn(state.U, Uold)
        y = fn(state.U,Uold)  
        delta = solve(J,y)
#        delta = spsolve(csr_matrix(np.asarray(J)),y)
        U = state.U - delta;
        res = norm(y/norm(U,np.inf),np.inf);
        converged1 = res < tol
        state._replace(count = state.count + 1,
                       converged = converged1,
                       fail = np.any(np.isnan(delta)),
                       U = U)
        
#        print(state.count, state.res)
        return state

    J = jac_fn(state.U, Uold);
    y = fn(state.U, Uold)
    delta = solve(J,y)
#    delta = spsolve(csr_matrix(np.asarray(J)),y)
    U = state.U - delta
    state._replace(U = U)
    state = lax.while_loop(lambda state: np.logical_and(np.logical_and(~ state.converged, ~state.fail), state.count < maxit),
                           body,
                           state)
    
    return state

