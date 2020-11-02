#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:57:57 2020

@author: hanrach
"""
import jax
from jax import jacfwd
from jax.config import config
config.update('jax_enable_x64', True)
from p2d_main_fn import p2d_fn
#from res_fn_order2 import fn
from residual import ResidualFunction

#Np = 10
#Nn = 10
#Mp = 10
#Ms = 10
#Mn = 10
#Ma = 5
#Mz = 5
Np = 20
Nn = 20
Mp = 20
Ms = 20
Mn = 20
Ma = 5
Mz = 5

solver = ResidualFunction(Np, Nn, Mp, Mn, Ms, Ma,Mz)
fn = solver.fn
jac_fn = jax.jit(jacfwd(fn))
fn = jax.jit(fn)
U, voltages, temps,time= p2d_fn(Np, Nn, Mp, Mn, Ms, Ma,Mz,fn,jac_fn)