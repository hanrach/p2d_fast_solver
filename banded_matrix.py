#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 23:15:36 2020

@author: hanrach
"""
import jax.numpy as np
import numpy as onp
from functools import partial
import numba as nb

@nb.njit()
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


