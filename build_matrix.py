#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:08:46 2020

@author: hanrach
"""

import jax.numpy as np
from jax.config import config
from scipy.sparse import coo_matrix
config.update("jax_enable_x64", True)

def build_tridiag(M, A, **kwargs):
    if (len(kwargs) == 2):
        bc1 = kwargs["right"];
        bc2 = kwargs["left"];
        data = np.hstack((bc1[0], bc1[1],\
                                     A[0],A[1],A[2], \
                                     bc2[0], bc2[1]))
        row = np.hstack((0,0,\
                         np.tile(np.arange(1,M+1),3),\
                         M+1, M+1))
        col = np.hstack((0,1,\
                         np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2), \
                         M,M+1))
    elif (len(kwargs) == 1):
        if (next(iter(kwargs.items()))[0] == "right"):
            bc1 = kwargs["right"];
            data = np.vstack((bc1[0], bc1[1],\
                                     A[0],A[1],A[2]))
            row = np.hstack((0,0,\
                             np.tile(np.arange(1,M+1),3)))
            col = np.hstack((0,1,\
                             np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2)))
        if (next(iter(kwargs.items()))[0] == "left"):
            bc2 = kwargs["left"]
            data = np.hstack((A[0],A[1],A[2],\
                                             bc2[0], bc2[1]))
            row = np.hstack((np.tile(np.arange(1,M+1),3),\
                             M+1, M+1))
            col = np.hstack((np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2),\
                             M, M+1))
    elif (len(kwargs) == 0):
        data = np.hstack((A[0], A[1], A[2]))
        row = np.hstack((np.tile(np.arange(1,M+1),3)))
        col = np.hstack((np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2)))
            
    return coo_matrix((data,(row,col)),shape=(M+2,M+2))

def build_diag(M, A, shape):
    try:
        data = np.concatenate(A).ravel()
    except:
        data = A;
    if (shape == "long"):
        row = np.arange(1,M+1)
        col = np.arange(0,M)
        Jac = coo_matrix((data, (row,col)), shape = (M+2, M))
    elif (shape == "wide"):
        row = np.arange(0,M)
        col = np.arange(1,M+1)
        Jac = coo_matrix((data, (row,col)), shape = (M, M+2))
    elif (shape == "square"):
        row = np.arange(0,M)
        col = np.arange(0,M)
        Jac = coo_matrix((data, (row,col)), shape = (M, M))
    return Jac

def build_bidiag(M, A):
    data = np.hstack((A[0],A[1]))
    row = np.hstack((np.tile(np.arange(1,M+1),2)))
    col = np.hstack((np.arange(0,M),np.arange(2,M+2)))
    Jac = coo_matrix((data,(row,col)),shape=(M+2,M+2))
    return Jac

def build_tridiag_c(M, A, **kwargs):
    bc1 = kwargs["right"];
    bc2 = kwargs["left"];
    
    data = np.hstack((bc1[0], bc1[1],\
                                 A[0],A[1],A[2], \
                                 bc2[0], bc2[1]))
    row = np.hstack((0,0,\
                     np.tile(np.arange(1,M+1),3),\
                     M+1, M+1))
    col = np.hstack((0,1,\
                     np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2), \
                     M,M+1))
    return coo_matrix((data,(row,col)),shape=(M+2,M+2))


def build_tridiag_interface(M, A, **kwargs):
    if (len(kwargs) == 2):
        bc1 = kwargs["right"];
        bc2 = kwargs["left"];
        data = np.hstack((bc1[0], bc1[1],\
                                     A[0],A[1],A[2], \
                                     bc2[0], bc2[1]))
        row = np.hstack((0,0,\
                         np.tile(np.arange(1,M+1),3),\
                         M+1, M+1))
        col = np.hstack((0,1,\
                         np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2), \
                         M,M+1))
    elif (len(kwargs) == 1):
        if (next(iter(kwargs.items()))[0] == "right"):
            bc1 = kwargs["right"];
            data = np.concatenate(np.vstack((bc1[0], bc1[1],\
                                     A[0],A[1],A[2]))).ravel()
            row = np.hstack((0,0,\
                             np.tile(np.arange(1,M+1),3)))
            col = np.hstack((0,1,\
                             np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2)))
        if (next(iter(kwargs.items()))[0] == "left"):
            bc2 = kwargs["left"]
            data = np.concatenate(np.vstack((A[0],A[1],A[2],\
                                             bc2[0], bc2[1]))).ravel()
            row = np.hstack((np.tile(np.arange(1,M+1),3),\
                             M+1, M+1))
            col = np.hstack((np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2),\
                             M, M+1))
    elif (len(kwargs) == 0):
        data = np.concatenate(np.vstack((A[0], A[1], A[2])))
        row = np.hstack((np.tile(np.arange(1,M+1),3)))
        col = np.hstack((np.arange(0,M),np.arange(1,M+1), np.arange(2,M+2)))
            
    return coo_matrix((data,(row,col)),shape=(M+2,M+2))

def empty_square(M):
    return coo_matrix((M,M))
def empty_rec(M,N):
    # coo_matrix((M+2,M))
    return coo_matrix((M,N))
    