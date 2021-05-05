#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:24:49 2021

@author: hanrach
"""

# plot functions
import jax.numpy as np
import matplotlib.pyplot as plt

def plot_solid_conc(cmat_vec, N, M, elec):
        
    xp = elec.l*( np.arange(1,M+1) - 0.5 )/M
    yp = elec.Rp*( np.arange(1,N+1) - 0.5 )/N
    xx, yy = np.meshgrid(xp, yp);

    cmat = np.reshape(cmat_vec, [N+2, M], order="F")
    plt.figure()
    plt.contourf(xx,yy,cmat[1:N+1,:]); plt.colorbar()
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.show()
    
def plot_u(uvec_pe, uvec_sep, uvec_ne):
    plt.figure()
    plt.plot(np.hstack([uvec_pe, uvec_sep, uvec_ne]))
    plt.show()