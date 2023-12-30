#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:54:09 2020

@author: hanrach
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 14:41:25 2020

@author: hanrach
"""
from functools import partial
import jax
import jax.numpy as np
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)
#from numpy.linalg import norm
#import matplotlib.pylab as plt
from model.ElectrodeEquation import ElectrodeEquation
from model.SeparatorEquation import SeparatorEquation
from model.CurrentCollectorEquation import CurrentCollectorEquation
from model.settings import p_electrode_constants,p_electrode_grid_param, n_electrode_constants,n_electrode_grid_param, \
sep_constants,sep_grid_param, a_cc_constants, z_cc_constants, cc_grid_param


class ResidualFunctionFast():
    def __init__(self, Mp, Np, Mn, Nn, Ms, Ma, Mz, delta_t, Iapp):
        self.Mp = Mp
        self.Np = Np
        self.Mn = Mn; self.Nn = Nn; self.Ms = Ms; self.Ma=Ma; self.Mz=Mz;

        self.peq = ElectrodeEquation(p_electrode_constants(),p_electrode_grid_param(Mp, Np), "positive", \
                                     sep_constants(), sep_grid_param(Ms), \
                                     a_cc_constants(), cc_grid_param(Ma),\
                                     z_cc_constants(), cc_grid_param(Mz),25751, 51554,delta_t)
        self.neq = ElectrodeEquation(n_electrode_constants(),n_electrode_grid_param(Mn, Nn), "negative", \
                                     sep_constants(), sep_grid_param(Ms), \
                                     a_cc_constants(), cc_grid_param(Ma),\
                                     z_cc_constants(), cc_grid_param(Mz),26128, 30555,delta_t)
        
        self.sepq = SeparatorEquation(sep_constants(),sep_grid_param(Ms), \
                                      p_electrode_constants(), n_electrode_constants(),\
                                      p_electrode_grid_param(Mp,Np), n_electrode_grid_param(Mn, Nn), delta_t)
        
        self.accq = CurrentCollectorEquation(a_cc_constants(),cc_grid_param(Ma),delta_t, Iapp)
        self.zccq = CurrentCollectorEquation(z_cc_constants(),cc_grid_param(Mz),delta_t, Iapp)

        self.Iapp = Iapp
        self.up0 =  0
        self.usep0 = self.up0 + Mp + 2
        self.un0 = self.usep0  + Ms+2
        
        self.jp0 = self.un0 + Mn + 2
        self.jn0 = self.jp0 + Mp
        
        self.etap0 = self.jn0 + Mn 
        self.etan0 = self.etap0 + Mp
        
        self.phisp0 = self.etan0 + Mn
        self.phisn0 = self.phisp0 + Mp + 2
        
        
        self.phiep0 = self.phisn0 + Mn +2
        self.phiesep0 = self.phiep0 + Mp + 2
        self.phien0 = self.phiesep0 + Ms + 2
        
        
        self.ta0 = self.phien0 + Mn + 2
        self.tp0 = self.ta0 + Ma+2
        self.tsep0 = self.tp0 + Mp+2
        self.tn0 = self.tsep0+ Ms+2
        self.tz0 = self.tn0 + Mn+2
        Ntot_pe =  4*(Mp + 2) + 2*(Mp)
        Ntot_ne =  4*(Mn + 2) + 2*(Mn)
        Ntot_sep =  3*(Ms + 2)
        Ntot_acc = Ma + 2
        Ntot_zcc = Mz + 2
        Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
        self.Ntot = Ntot
       
       
        
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_pe(self,val, uvec, Tvec, jvec, uvec_old, uvec_sep,Tvec_sep):
        up0 = self.up0
        Mp = self.Mp
        peq = self.peq
        val = val.at[up0].set(peq.bc_zero_neumann(uvec[0], uvec[1]))
           
        val = val.at[up0 + 1: up0 + Mp + 1].set(vmap(peq.electrolyte_conc)(uvec[0:Mp], uvec[1:Mp+1], uvec[2:Mp+2],Tvec[0:Mp], Tvec[1:Mp+1], Tvec[2:Mp+2], jvec[0:Mp],uvec_old[1:Mp+1]))
           
        val = val.at[up0 + Mp + 1].set(peq.bc_u_sep_p(uvec[Mp],uvec[Mp+1],Tvec[Mp],Tvec[Mp+1],\
         uvec_sep[0],uvec_sep[1],Tvec_sep[0],Tvec_sep[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_sep(self,val, uvec, Tvec, uvec_old, uvec_pe, Tvec_pe, uvec_ne, Tvec_ne):
        usep0 = self.usep0
        Ms = self.Ms
        peq = self.peq
        Mp = self.Mp
        sepq = self.sepq
        val = val.at[usep0].set(peq.bc_inter_cont(uvec[0], uvec[1], uvec_pe[Mp], uvec_pe[Mp + 1]) )
           
        val = val.at[usep0+ 1: usep0 + 1 + Ms].set(vmap(sepq.electrolyte_conc)(uvec[0:Ms], uvec[1:Ms+1], uvec[2:Ms+2], Tvec[0:Ms], Tvec[1:Ms+1], Tvec[2:Ms+2], uvec_old[1:Ms+1]))
           
        val = val.at[usep0+ Ms + 1].set(peq.bc_inter_cont(uvec_ne[0], uvec_ne[1], uvec[Ms], uvec[Ms+1]))
        return val
    
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_ne(self,val, uvec, Tvec, jvec, uvec_old, uvec_sep, Tvec_sep):
        un0= self.un0
        Mn = self.Mn
        neq = self.neq
        Ms = self.Ms
        val = val.at[un0].set(neq.bc_u_sep_n(uvec[0], uvec[1], Tvec[0], Tvec[1], uvec_sep[Ms], uvec_sep[Ms+1], Tvec_sep[Ms], Tvec_sep[Ms+1]))
        
        val = val.at[un0+ 1: un0+ 1 + Mn].set(vmap(neq.electrolyte_conc)(uvec[0:Mn], uvec[1:Mn+1], uvec[2:Mn+2], Tvec[0:Mn], Tvec[1:Mn+1], Tvec[2:Mn+2], jvec[0:Mn],uvec_old[1:Mn+1]))
        
        val = val.at[un0 + 1 + Mn].set(neq.bc_zero_neumann(uvec[Mn],uvec[Mn+1]))    
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))        
    def res_T_acc(self,val, Tvec, Tvec_old, Tvec_pe):
        ta0 = self.ta0
        Ma = self.Ma
        accq = self.accq
        peq = self.peq
        val = val.at[ta0].set(accq.bc_temp_a(Tvec[0], Tvec[1]))
        val = val.at[ta0 + 1: ta0 + Ma + 1].set(vmap(accq.temperature)(Tvec[0:Ma], Tvec[1:Ma+1], Tvec[2:Ma+2], Tvec_old[1:Ma+1]))
        val = val.at[ta0 + Ma + 1].set(peq.bc_inter_cont(Tvec[Ma], Tvec[Ma+1], Tvec_pe[0], Tvec_pe[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_sep(self,val, Tvec, uvec, phievec, Tvec_old, Tvec_pe, Tvec_ne):
        tsep0 = self.tsep0
        peq = self.peq
        sepq = self.sepq
        Mp = self.Mp
        Ms = self.Ms
        val = val.at[tsep0].set(peq.bc_inter_cont(Tvec_pe[Mp], Tvec_pe[Mp+1], Tvec[0], Tvec[1]))
        
        val = val.at[tsep0 + 1: tsep0 + 1 + Ms].set(vmap(sepq.temperature)(uvec[0:Ms], uvec[1:Ms+1], uvec[2:Ms+2], phievec[0:Ms], phievec[2:Ms+2], Tvec[0:Ms], Tvec[1:Ms+1], Tvec[2:Ms+2], Tvec_old[1:Ms+1]))
        
        val = val.at[ tsep0 + 1 + Ms].set(peq.bc_inter_cont(Tvec[Ms], Tvec[Ms+1], Tvec_ne[0], Tvec_ne[1]))
        return val

    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_zcc(self,val, Tvec, Tvec_old, Tvec_ne):
        tz0 = self.tz0
        Mn = self.Mn
        Mz = self.Mz
        zccq = self.zccq
        neq = self.neq
        val = val.at[tz0].set(neq.bc_inter_cont(Tvec_ne[Mn], Tvec_ne[Mn+1], Tvec[0], Tvec[1]))
        val = val.at[tz0+1:tz0 + 1 + Mz].set(vmap(zccq.temperature)(Tvec[0:Mz], Tvec[1:Mz+1], Tvec[2:Mz+2], Tvec_old[1:Mz+1]))
        val = val.at[tz0+Mz+1].set(zccq.bc_temp_z(Tvec[Mz], Tvec[Mz+1]))
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_pe(self,val, uvec, phievec, Tvec, jvec, uvec_sep, phievec_sep, Tvec_sep):
        phiep0 = self.phiep0
        peq = self.peq
        Mp = self.Mp
        val = val.at[phiep0].set(peq.bc_zero_neumann(phievec[0], phievec[1]))
        
        val = val.at[phiep0 + 1: phiep0 + Mp + 1].set(vmap(peq.electrolyte_poten)(uvec[0:Mp], uvec[1:Mp+1], uvec[2:Mp+2],
        phievec[0:Mp],phievec[1:Mp+1], phievec[2:Mp+2], Tvec[0:Mp], Tvec[1:Mp+1], Tvec[2:Mp+2], jvec[0:Mp]))
        
           
        val = val.at[phiep0 + Mp+1].set(peq.bc_phie_p(phievec[Mp], phievec[Mp+1],  phievec_sep[0], phievec_sep[1], \
                           uvec[Mp], uvec[Mp+1], uvec_sep[0], uvec_sep[1],\
                           Tvec[Mp], Tvec[Mp+1], Tvec_sep[0], Tvec_sep[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_sep(self,val, uvec, phievec, Tvec, phievec_pe, phievec_ne):
        phiesep0 = self.phiesep0
        peq = self.peq
        sepq= self.sepq
        Ms = self.Ms
        neq = self.neq
        Mp = self.Mp
        val = val.at[phiesep0].set(peq.bc_inter_cont(phievec_pe[Mp], phievec_pe[Mp+1], phievec[0], phievec[1]))
        
        val = val.at[phiesep0 + 1: phiesep0 + Ms + 1].set(vmap(sepq.electrolyte_poten)(uvec[0:Ms], uvec[1:Ms+1], uvec[2:Ms+2], phievec[0:Ms], phievec[1:Ms+1], phievec[2:Ms+2], Tvec[0:Ms], Tvec[1:Ms+1], Tvec[2:Ms+2]))
        
        val = val.at[phiesep0 + Ms+1].set(neq.bc_inter_cont(phievec_ne[0], phievec_ne[1], phievec[Ms], phievec[Ms+1]))
        return val
        
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_ne(self,val, uvec, phievec, Tvec, jvec, uvec_sep, phievec_sep, Tvec_sep):
        phien0 = self.phien0
        neq = self.neq
        Mn = self.Mn
        Ms = self.Ms
        
        val = val.at[phien0].set(neq.bc_phie_n(phievec[0], phievec[1], phievec_sep[Ms], phievec_sep[Ms+1],\
                           uvec[0], uvec[1], uvec_sep[Ms], uvec_sep[Ms+1], \
                           Tvec[0], Tvec[1], Tvec_sep[Ms], Tvec_sep[Ms+1]))
        
        val = val.at[phien0 + 1: phien0 + Mn + 1].set(vmap(neq.electrolyte_poten)(uvec[0:Mn], uvec[1:Mn+1], uvec[2:Mn+2],
        phievec[0:Mn],phievec[1:Mn+1], phievec[2:Mn+2], Tvec[0:Mn], Tvec[1:Mn+1], Tvec[2:Mn+2], jvec[0:Mn]))
        
        val = val.at[phien0 + Mn+1].set(neq.bc_zero_dirichlet(phievec[Mn], phievec[Mn+1]))
        return val
        
    
    
    
    #    @jax.jit  
    @partial(jax.jit, static_argnums=(0,))
    def res_phis(self,val, phis_pe, jvec_pe, phis_ne, jvec_ne):
        phisp0 = self.phisp0
        peq= self.peq
        Mp = self.Mp
        phisn0  = self.phisn0
        neq = self.neq
        Mn = self.Mn
        val = val.at[phisp0].set(peq.bc_phis(phis_pe[0], phis_pe[1], self.Iapp))
        #    val = val.at[phisp0].set(peq.bc_zero_dirichlet(phis_pe[0], phis_pe[1]))
        val = val.at[phisp0 + 1: phisp0 + Mp+1].set(vmap(peq.solid_poten)(phis_pe[0:Mp], phis_pe[1:Mp+1], phis_pe[2:Mp+2], jvec_pe[0:Mp]))
        val = val.at[phisp0 + Mp+1].set(peq.bc_phis(phis_pe[Mp], phis_pe[Mp+1], 0) )
        #    val = val.at[phisp0 + Mp+1].set(peq.bc_zero_dirichlet(phis_pe[Mp], phis_pe[Mp+1]) )
        
        val = val.at[phisn0].set(neq.bc_phis(phis_ne[0], phis_ne[1], 0))
        val = val.at[phisn0 + 1: phisn0 + Mn +1].set(vmap(neq.solid_poten)(phis_ne[0:Mn], phis_ne[1:Mn+1], phis_ne[2:Mn+2], jvec_ne[0:Mn]))
        val = val.at[phisn0 + Mn+1].set(neq.bc_phis(phis_ne[Mn], phis_ne[Mn+1], self.Iapp))
        #    val = val.at[phisn0 + Mn+1].set(neq.bc_zero_dirichlet(phis_ne[Mn], phis_ne[Mn+1]))
        return val
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_pe_fast(self,val, Tvec, uvec, phievec, phisvec, jvec, etavec, cs_pe1, gamma_p, Tvec_old, Tvec_acc, Tvec_sep):
        tp0 = self.tp0
        Ma = self.Ma
        peq = self.peq
        Mp = self.Mp
        val = val.at[tp0].set(peq.bc_temp_ap(Tvec_acc[Ma], Tvec_acc[Ma+1], Tvec[0], Tvec[1]))
        
        val = val.at[tp0 + 1: tp0 + Mp + 1].set(vmap(peq.temperature_fast)(uvec[0:Mp], uvec[1:Mp+1], uvec[2:Mp+2],\
                           phievec[0:Mp], phievec[2:Mp+2], phisvec[0:Mp], phisvec[2:Mp+2], \
                           Tvec[0:Mp], Tvec[1:Mp+1], Tvec[2:Mp+2], \
                           jvec[0:Mp], etavec[0:Mp], cs_pe1, gamma_p, peq.cmax*np.ones(Mp), Tvec_old[1:Mp+1]))
        
        val = val.at[tp0 + Mp + 1].set(peq.bc_temp_ps(Tvec[Mp], Tvec[Mp+1], Tvec_sep[0], Tvec_sep[1]))
        return val
    
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))    
    def res_T_ne_fast(self,val, Tvec, uvec, phievec, phisvec, jvec, etavec, cs_ne1, gamma_n, Tvec_old, Tvec_zcc, Tvec_sep):
        tn0= self.tn0
        Ms = self.Ms
        neq = self.neq
        Mn = self.Mn
        val = val.at[ tn0].set(neq.bc_temp_sn(Tvec_sep[Ms], Tvec_sep[Ms+1], Tvec[0], Tvec[1]))
        
        val = val.at[tn0 + 1: tn0+ 1 + Mn].set(vmap(neq.temperature_fast)(uvec[0:Mn], uvec[1:Mn+1], uvec[2:Mn+2],\
                           phievec[0:Mn], phievec[2:Mn+2], phisvec[0:Mn], phisvec[2:Mn+2], \
                           Tvec[0:Mn], Tvec[1:Mn+1], Tvec[2:Mn+2], \
                           jvec[0:Mn], etavec[0:Mn], cs_ne1, gamma_n, neq.cmax*np.ones(Mn), Tvec_old[1:Mn+1]))
        
        val = val.at[tn0+ 1 + Mn].set(neq.bc_temp_n(Tvec[Mn], Tvec[Mn+1], Tvec_zcc[0], Tvec_zcc[1]))
        return val
        
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_j_fast(self,val, jvec_pe, uvec_pe, Tvec_pe, eta_pe, cs_pe1, gamma_p, jvec_ne, uvec_ne, Tvec_ne, eta_ne, cs_ne1, gamma_n):
        jp0 = self.jp0
        peq= self.peq
        Mp = self.Mp
        jn0  = self.jn0
        neq = self.neq
        Mn = self.Mn
        val = val.at[jp0:jp0 + Mp].set(vmap(peq.ionic_flux_fast)(jvec_pe, uvec_pe[1:Mp+1], Tvec_pe[1:Mp+1], eta_pe, cs_pe1, gamma_p, peq.cmax*np.ones(Mp)))
        val = val.at[jn0: jn0 + Mn].set(vmap(neq.ionic_flux_fast)(jvec_ne, uvec_ne[1:Mn+1], Tvec_ne[1:Mn+1], eta_ne,cs_ne1, gamma_n, neq.cmax*np.ones(Mn)))
        return val
    
    #    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_eta_fast(self,val, eta_pe, phis_pe, phie_pe, Tvec_pe, jvec_pe, cs_pe1, gamma_p, eta_ne, phis_ne, phie_ne, Tvec_ne, jvec_ne, cs_ne1, gamma_n):
        etap0 = self.etap0
        etan0 = self.etan0
        Mp = self.Mp; Mn = self.Mn; peq=self.peq; neq=self.neq; 
        val = val.at[etap0:etap0 + Mp].set(vmap(peq.over_poten_fast)(eta_pe, phis_pe[1:Mp+1], phie_pe[1:Mp+1], Tvec_pe[1:Mp+1], jvec_pe, cs_pe1, gamma_p, peq.cmax*np.ones(Mp)))
        val = val.at[etan0: etan0 + Mn].set(vmap(neq.over_poten_fast)(eta_ne, phis_ne[1:Mn+1], phie_ne[1:Mn+1], Tvec_ne[1:Mn+1],jvec_ne, cs_ne1, gamma_n, neq.cmax*np.ones(Mn)))
        return val



#    
