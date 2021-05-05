#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:48:57 2020

@author: hanrach
"""
import numpy as onp
def reorder_tot(Mp, Mn, Ms, Ma, Mz):
    up0 =  0
    usep0 =up0 + Mp + 2
    un0 = usep0  + Ms+2
    
    jp0 = un0 + Mn + 2
    jn0 = jp0 + Mp
    
    etap0 = jn0 + Mn 
    etan0 = etap0 + Mp
    
    phisp0 = etan0 + Mn
    phisn0 = phisp0 + Mp + 2
    
    
    phiep0 = phisn0 + Mn +2
    phiesep0 = phiep0 + Mp + 2
    phien0 = phiesep0 + Ms + 2
    
    
    ta0 = phien0 + Mn + 2
    tp0 = ta0 + Ma+2
    tsep0 = tp0 + Mp+2
    tn0 = tsep0+ Ms+2
    tz0 = tn0 + Mn+2
    
    order = onp.arange(0,4*(Mp+2) + 2*Mp + 2*Mn + 4*(Mn+2) + 3*(Ms+2) + Ma+2 + Mz+2)
    
    for i in range(0,Ma+1):
        order[ta0+i]= i
    # u0
    p0 = Ma + 1
    order[0] = p0
     #phisp0
    order[phisp0] = 1 + p0
     #Phie0
    order[phiep0] = 2 + p0
    #Tp0   
    order[tp0] = 3+ p0
    order[ta0 + Ma + 1] = 4+ p0

    # u Mp+1
    order[Mp+1] = 5 + 6*Mp+ p0
    # phis Mp+1    
    order[phisp0 + Mp + 1] = 5 + 6*Mp + 1 + p0
            
    #Phiep Mp+1
    order[phiep0+Mp+1] = 5 + 6*Mp + 2 + p0
            
    # Tp0 + Mp+1    
    order[tp0+Mp+1] = 5 + 6*Mp + 3 + p0
    
    for i in range(1,Mp+1):
        order[i]= 5 + 6*(i-1) + p0
    # j
    for i in range(0, Mp):
        order[i + jp0] = 5 + 6*i + 1+ p0; 
    
    #eta
    for i in range(0, Mp):
        order[i + etap0 ] = 5 + 6*i + 2+ p0;

    
    # phis
    for i in range(1, Mp+1):
        order[i + phisp0] = 5 + 6*(i-1) + 3+ p0;

    
    #phie    
    for i in range(1, Mp+1):
        order[i + phiep0] =5 + 6*(i-1) +4+ p0;
    
    # T    
    for i in range(1, Mp+1):
        order[i + tp0] = 5 + 6*(i-1) + 5+ p0;
        
    #separator
    
    #usep0
    sep0 = 4*(Mp+2) + 2*Mp + 1 + p0
    order[usep0] = sep0
    order[phiesep0] = sep0 + 1
    order[tsep0] = sep0 + 2
    
    
    for i in range(1, Ms+1):
        order[i + usep0] = sep0 + 3*i
        order[i + phiesep0] = sep0 + 3*i + 1
        order[i + tsep0] = sep0 + 3*i + 2
        
    order[usep0+Ms+1] = sep0 + 3*Ms + 3
    order[phiesep0+Ms+1] = sep0 + 3*Ms + 4
    order[tsep0 + Ms + 1] = sep0 + 3*Ms + 5
        
        
    n0 = p0 + 4*(Mp+2) + 2*Mp  + 3*(Ms+2) + 1
        # u0 
    order[un0] = n0 + 1
        #phisp0
    order[phisn0] = n0 + 1  
    #Phie0
    order[phien0] = n0 + 2
        #Tp0    
    order[tn0] = n0 + 3
    
    
    for i in range(1,Mn+1):
        order[un0 + i]= n0 + 4 + 6*(i-1)
 
    # j
    for i in range(0, Mn):
        order[i + jn0] = n0 + 4 + 6*i + 1;

    #eta
    for i in range(0, Mn):
        order[i + etan0 ] = n0 + 4 + 6*i + 2;

    
    # phis
    for i in range(1, Mn+1):
        order[i + phisn0] = n0 + 4 + 6*(i-1) + 3;

    
    #phie    
    for i in range(1, Mn+1):
        order[i + phien0] = n0 + 4 + 6*(i-1) +4;
    
    # T    
    for i in range(1, Mn+1):
        order[i + tn0] = n0 + 4 + 6*(i-1) + 5;
        
        # u Mn+1
    order[un0 + Mn + 1] = n0 + 4 + 6*Mn
    # phis Mp+1    
    order[phisn0 + Mn + 1] = n0 + 4 + 6*Mn + 1  
  #Phiep Mp+1
    order[phien0+Mn+1] = n0 + 4 + 6*Mn + 2      
    # Tp0 + Mp+1    
    order[tn0+Mn+1] = n0 + 4 + 6*Mn + 3
    order[tz0] = n0 + 4 + 6*Mn + 4
    
    for i in range(1,Mz+2):
        order[tz0+i]= n0 + 4 + 6*Mn + 4 + i 
    
#    for i in range(0,Ma+1):
#        order[ta0+i]= n0 + 4 + 6*Mn + 4 + (i+1) + Mz+1
#        
    
    sort_index = onp.argsort(order)
        
    return sort_index