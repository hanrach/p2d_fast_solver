#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:05:04 2020

@author: hanrach
"""



global trans
global F
global R
global gamma
global Tref
global Iapp

Iapp = -30;
trans = 0.364;
F = 96485;
R = 8.314472;
gamma = 2*(1-trans)*R/F;
Tref = 298.15;
delta_t = 10;
h = 1
