#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 01:58:27 2020

@author: hanrach
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:50:59 2020

@author: hanrach
"""

from functools import partial
import jax
import jax.numpy as np
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)
from settings import Iapp
from ElectrodeEquation import ElectrodeEquation, p_electrode_constants,p_electrode_grid_param, n_electrode_constants,n_electrode_grid_param
from SeparatorEquation import SeparatorEquation, sep_constants,sep_grid_param
from CurrentCollectorEquation import CurrentCollectorEquation, a_cc_constants, z_cc_constants, cc_grid_param
from unpack import unpack
image_folder = 'images'
video_name = 'video.avi'
from jax import lax

class ResidualFunction:
    def __init__(self, Mp, Np, Mn, Nn, Ms, Ma, Mz):
        self.Mp = Mp
        self.Np = Np
        self.Mn = Mn; self.Nn = Nn; self.Ms = Ms; self.Ma=Ma; self.Mz=Mz;
        Ntot_pe = Mp*(Np+2) +  4*(Mp + 2) + 2*(Mp)
        Ntot_ne= Mn*(Nn+2)+  4*(Mn + 2) + 2*(Mn)
        Ntot_sep =  3*(Ms + 2)
        Ntot_acc = Ma + 2
        Ntot_zcc = Mz + 2
        self.Ntot = Ntot_pe+ Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
        self.peq = ElectrodeEquation(p_electrode_constants(),p_electrode_grid_param(Mp, Np), "positive",25751, 51554)
        self.neq = ElectrodeEquation(n_electrode_constants(),n_electrode_grid_param(Mn, Nn), "negative", 26128, 30555)
        self.sepq = SeparatorEquation(sep_constants(),sep_grid_param(Ms))
        self.accq = CurrentCollectorEquation(a_cc_constants(),cc_grid_param(Ma))
        self.zccq = CurrentCollectorEquation(z_cc_constants(),cc_grid_param(Mz))
        
        

        self.up0 =  Mp*(Np+2) + Mn*(Nn+2)
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
                            
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_c_fn(self, val, cmat_pe, jvec_pe, Tvec_pe, cmat_old_pe, cmat_ne, jvec_ne, Tvec_ne, cmat_old_ne):
  
        Mp = self.Mp
        Np = self.Np
        peq = self.peq
        Mn = self.Mn
        Nn = self.Nn
        neq = self.neq
        for i in range(0,Mp):
            val = jax.ops.index_update(val, jax.ops.index[i*(Np+2)], peq.bc_zero_neumann(cmat_pe[0,i], cmat_pe[1,i]) )
            val = jax.ops.index_update(val, jax.ops.index[i*(Np+2) + Np+1], peq.bc_neumann_c(cmat_pe[Np,i], cmat_pe[Np+1,i],jvec_pe[i], Tvec_pe[i+1] )) 
            res_c = peq.solid_conc_2(cmat_pe[0:Np,i], cmat_pe[1:Np+1,i], cmat_pe[2:Np+2,i], cmat_old_pe[1:Np+1, i])
            val = jax.ops.index_update(val, jax.ops.index[i*(Np+2)+1 : Np+1 + i*(Np+2) ], res_c)
        
        """ negative """    
   
        for i in range(0,Mn):
            val = jax.ops.index_update(val, jax.ops.index[Mp*(Np+2) + i*(Nn+2)], neq.bc_zero_neumann(cmat_ne[0,i], cmat_ne[1,i]) )
            val = jax.ops.index_update(val, jax.ops.index[Mp*(Np+2) + i*(Nn+2) + Nn+1], neq.bc_neumann_c(cmat_ne[Nn,i], cmat_ne[Nn+1,i], jvec_ne[i], Tvec_ne[i+1])) 
            res_cn = neq.solid_conc_2(cmat_ne[0:Nn,i], cmat_ne[1:Nn+1,i], cmat_ne[2:Nn+2,i], cmat_old_ne[1:Nn+1, i])
            val = jax.ops.index_update(val, jax.ops.index[Mp*(Np+2) + i*(Nn+2)+1 : Mp*(Np+2) + Nn+1 + i*(Nn+2) ], res_cn)
        
        return val
    
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_pe(self,val, uvec, Tvec, jvec, uvec_old, uvec_sep,Tvec_sep):
        up0 = self.up0
        Mp = self.Mp
        peq = self.peq
        val = jax.ops.index_update(val, jax.ops.index[up0], peq.bc_zero_neumann(uvec[0], uvec[1]))
       
        val = jax.ops.index_update(val, jax.ops.index[up0 + 1: up0 + Mp + 1],
                                   vmap(peq.electrolyte_conc)(uvec[0:Mp], uvec[1:Mp+1], uvec[2:Mp+2],
                                       Tvec[0:Mp], Tvec[1:Mp+1], Tvec[2:Mp+2], jvec[0:Mp],uvec_old[1:Mp+1]))
       
        val = jax.ops.index_update(val, jax.ops.index[up0 + Mp + 1], peq.bc_u_sep_p(uvec[Mp],uvec[Mp+1],Tvec[Mp],Tvec[Mp+1],\
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
        val = jax.ops.index_update(val, jax.ops.index[usep0], peq.bc_inter_cont(uvec[0], uvec[1], uvec_pe[Mp], uvec_pe[Mp + 1]) )
       
        val = jax.ops.index_update(val, jax.ops.index[usep0+ 1: usep0 + 1 + Ms], vmap(sepq.electrolyte_conc)(uvec[0:Ms], uvec[1:Ms+1], uvec[2:Ms+2], Tvec[0:Ms], Tvec[1:Ms+1], Tvec[2:Ms+2], uvec_old[1:Ms+1]))
       
        val = jax.ops.index_update(val, jax.ops.index[usep0+ Ms + 1],
        peq.bc_inter_cont(uvec_ne[0], uvec_ne[1], uvec[Ms], uvec[Ms+1]))
        return val
    
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_u_ne(self,val, uvec, Tvec, jvec, uvec_old, uvec_sep, Tvec_sep):
        un0= self.un0
        Mn = self.Mn
        neq = self.neq
        Ms = self.Ms
        val = jax.ops.index_update(val, jax.ops.index[un0], neq.bc_u_sep_n(uvec[0], uvec[1], Tvec[0], Tvec[1], uvec_sep[Ms], uvec_sep[Ms+1], Tvec_sep[Ms], Tvec_sep[Ms+1]))
        
        val = jax.ops.index_update(val, jax.ops.index[un0+ 1: un0+ 1 + Mn],
                                   vmap(neq.electrolyte_conc)(uvec[0:Mn], uvec[1:Mn+1], uvec[2:Mn+2],
                                       Tvec[0:Mn], Tvec[1:Mn+1], Tvec[2:Mn+2], jvec[0:Mn],uvec_old[1:Mn+1]))
        
        val = jax.ops.index_update(val, jax.ops.index[un0 + 1 + Mn], neq.bc_zero_neumann(uvec[Mn],uvec[Mn+1]))    
        return val
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))        
    def res_T_acc(self,val, Tvec, Tvec_old, Tvec_pe):
        ta0 = self.ta0
        Ma = self.Ma
        accq = self.accq
        peq = self.peq
        val = jax.ops.index_update(val, jax.ops.index[ta0], accq.bc_temp_a(Tvec[0], Tvec[1]))
        val = jax.ops.index_update(val, jax.ops.index[ta0 + 1: ta0 + Ma + 1], vmap(accq.temperature)(Tvec[0:Ma], Tvec[1:Ma+1], Tvec[2:Ma+2], Tvec_old[1:Ma+1]))
        val = jax.ops.index_update(val, jax.ops.index[ta0 + Ma + 1], peq.bc_inter_cont(Tvec[Ma], Tvec[Ma+1], Tvec_pe[0], Tvec_pe[1]))
        return val

#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))    
    def res_T_pe(self,val, Tvec, uvec, phievec, phisvec, jvec, etavec, cmat, Tvec_old, Tvec_acc, Tvec_sep):
        tp0 = self.tp0
        Ma = self.Ma
        peq = self.peq
        Mp = self.Mp
        Np = self.Np
        val = jax.ops.index_update(val, jax.ops.index[tp0], peq.bc_temp_ap(Tvec_acc[Ma], Tvec_acc[Ma+1], Tvec[0], Tvec[1]))
    
        val = jax.ops.index_update(val, jax.ops.index[tp0 + 1: tp0 + Mp + 1], vmap(peq.temperature)(uvec[0:Mp], uvec[1:Mp+1], uvec[2:Mp+2],\
                                   phievec[0:Mp], phievec[2:Mp+2], phisvec[0:Mp], phisvec[2:Mp+2], \
                                   Tvec[0:Mp], Tvec[1:Mp+1], Tvec[2:Mp+2], \
                                   jvec[0:Mp], etavec[0:Mp], cmat[Np, :], cmat[Np+1, :], peq.cmax*np.ones(Mp), Tvec_old[1:Mp+1]))
    
  
        val = jax.ops.index_update(val, jax.ops.index[tp0 + Mp + 1], peq.bc_temp_ps(Tvec[Mp], Tvec[Mp+1], Tvec_sep[0], Tvec_sep[1]))
        return val
    
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_sep(self,val, Tvec, uvec, phievec, Tvec_old, Tvec_pe, Tvec_ne):
        tsep0 = self.tsep0
        peq = self.peq
        sepq = self.sepq
        Mp = self.Mp
        Ms = self.Ms
        val = jax.ops.index_update(val, jax.ops.index[tsep0], peq.bc_inter_cont(Tvec_pe[Mp], Tvec_pe[Mp+1], Tvec[0], Tvec[1]))
    
        val = jax.ops.index_update(val, jax.ops.index[tsep0 + 1: tsep0 + 1 + Ms],
            vmap(sepq.temperature)(uvec[0:Ms], uvec[1:Ms+1], uvec[2:Ms+2], phievec[0:Ms], phievec[2:Ms+2],
                Tvec[0:Ms], Tvec[1:Ms+1], Tvec[2:Ms+2], Tvec_old[1:Ms+1]))
        
        val = jax.ops.index_update(val, jax.ops.index[ tsep0 + 1 + Ms], peq.bc_inter_cont(Tvec[Ms], Tvec[Ms+1], Tvec_ne[0], Tvec_ne[1]))
        return val
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_ne(self,val, Tvec, uvec, phievec, phisvec, jvec, etavec, cmat, Tvec_old, Tvec_zcc, Tvec_sep):
        tn0= self.tn0
        Ms = self.Ms
        neq = self.neq
        Mn = self.Mn
        Nn = self.Nn
        val = jax.ops.index_update(val, jax.ops.index[ tn0], neq.bc_temp_sn(Tvec_sep[Ms], Tvec_sep[Ms+1], Tvec[0], Tvec[1]))
        
        val = jax.ops.index_update(val, jax.ops.index[tn0 + 1: tn0+ 1 + Mn], vmap(neq.temperature)(uvec[0:Mn], uvec[1:Mn+1], uvec[2:Mn+2],\
                                   phievec[0:Mn], phievec[2:Mn+2], phisvec[0:Mn], phisvec[2:Mn+2], \
                                   Tvec[0:Mn], Tvec[1:Mn+1], Tvec[2:Mn+2], \
                                   jvec[0:Mn], etavec[0:Mn], cmat[Nn, :], cmat[Nn+1, :], neq.cmax*np.ones(Mn), Tvec_old[1:Mn+1]))
    

    
        val = jax.ops.index_update(val, jax.ops.index[tn0+ 1 + Mn], neq.bc_temp_n(Tvec[Mn], Tvec[Mn+1], Tvec_zcc[0], Tvec_zcc[1]))
        return val
    
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_T_zcc(self,val, Tvec, Tvec_old, Tvec_ne):
        tz0 = self.tz0
        Mn = self.Mn
        Mz = self.Mz
        zccq = self.zccq
        neq = self.neq
        val = jax.ops.index_update(val, jax.ops.index[tz0], neq.bc_inter_cont(Tvec_ne[Mn], Tvec_ne[Mn+1], Tvec[0], Tvec[1]))
        val = jax.ops.index_update(val, jax.ops.index[tz0+1:tz0 + 1 + Mz], vmap(zccq.temperature)(Tvec[0:Mz], Tvec[1:Mz+1], Tvec[2:Mz+2], Tvec_old[1:Mz+1]))
        val = jax.ops.index_update(val, jax.ops.index[tz0+Mz+1], zccq.bc_temp_z(Tvec[Mz], Tvec[Mz+1]))
        return val
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_pe(self,val, uvec, phievec, Tvec, jvec, uvec_sep, phievec_sep, Tvec_sep):
        phiep0 = self.phiep0
        peq = self.peq
        Mp = self.Mp
        val = jax.ops.index_update(val, jax.ops.index[phiep0], peq.bc_zero_neumann(phievec[0], phievec[1]))
        
        val = jax.ops.index_update(val, jax.ops.index[phiep0 + 1: phiep0 + Mp + 1], vmap(peq.electrolyte_poten)(uvec[0:Mp], uvec[1:Mp+1], uvec[2:Mp+2],
            phievec[0:Mp],phievec[1:Mp+1], phievec[2:Mp+2], Tvec[0:Mp], Tvec[1:Mp+1], Tvec[2:Mp+2], jvec[0:Mp]))
        
       
        val = jax.ops.index_update(val, jax.ops.index[phiep0 + Mp+1], peq.bc_phie_p(phievec[Mp], phievec[Mp+1],  phievec_sep[0], phievec_sep[1], \
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
        val = jax.ops.index_update(val, jax.ops.index[phiesep0], peq.bc_inter_cont(phievec_pe[Mp], phievec_pe[Mp+1], phievec[0], phievec[1]))
        
        val = jax.ops.index_update(val, jax.ops.index[phiesep0 + 1: phiesep0 + Ms + 1], vmap(sepq.electrolyte_poten)(uvec[0:Ms], uvec[1:Ms+1], uvec[2:Ms+2], phievec[0:Ms], phievec[1:Ms+1], phievec[2:Ms+2], Tvec[0:Ms], Tvec[1:Ms+1], Tvec[2:Ms+2]))
        
        val = jax.ops.index_update(val, jax.ops.index[phiesep0 + Ms+1], neq.bc_inter_cont(phievec_ne[0], phievec_ne[1], phievec[Ms], phievec[Ms+1]))
        return val
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_phie_ne(self,val, uvec, phievec, Tvec, jvec, uvec_sep, phievec_sep, Tvec_sep):
        phien0 = self.phien0
        neq = self.neq
        Mn = self.Mn
        Ms = self.Ms
    
        val = jax.ops.index_update(val, jax.ops.index[phien0], neq.bc_phie_n(phievec[0], phievec[1], phievec_sep[Ms], phievec_sep[Ms+1],\
                                   uvec[0], uvec[1], uvec_sep[Ms], uvec_sep[Ms+1], \
                                   Tvec[0], Tvec[1], Tvec_sep[Ms], Tvec_sep[Ms+1]))
        
        val = jax.ops.index_update(val, jax.ops.index[phien0 + 1: phien0 + Mn + 1], vmap(neq.electrolyte_poten)(uvec[0:Mn], uvec[1:Mn+1], uvec[2:Mn+2],
            phievec[0:Mn],phievec[1:Mn+1], phievec[2:Mn+2], Tvec[0:Mn], Tvec[1:Mn+1], Tvec[2:Mn+2], jvec[0:Mn]))
        
        val = jax.ops.index_update(val, jax.ops.index[phien0 + Mn+1], neq.bc_zero_dirichlet(phievec[Mn], phievec[Mn+1]))
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
        val = jax.ops.index_update(val, jax.ops.index[phisp0], peq.bc_phis(phis_pe[0], phis_pe[1], Iapp))
    #    val = jax.ops.index_update(val, jax.ops.index[phisp0], peq.bc_zero_dirichlet(phis_pe[0], phis_pe[1]))
        val = jax.ops.index_update(val, jax.ops.index[phisp0 + 1: phisp0 + Mp+1], vmap(peq.solid_poten)(phis_pe[0:Mp], phis_pe[1:Mp+1], phis_pe[2:Mp+2], jvec_pe[0:Mp]))
        val = jax.ops.index_update(val, jax.ops.index[phisp0 + Mp+1], peq.bc_phis(phis_pe[Mp], phis_pe[Mp+1], 0) )
    #    val = jax.ops.index_update(val, jax.ops.index[phisp0 + Mp+1], peq.bc_zero_dirichlet(phis_pe[Mp], phis_pe[Mp+1]) )
    
        val = jax.ops.index_update(val, jax.ops.index[phisn0], neq.bc_phis(phis_ne[0], phis_ne[1], 0))
        val = jax.ops.index_update(val, jax.ops.index[phisn0 + 1: phisn0 + Mn +1], vmap(neq.solid_poten)(phis_ne[0:Mn], phis_ne[1:Mn+1], phis_ne[2:Mn+2], jvec_ne[0:Mn]))
        val = jax.ops.index_update(val, jax.ops.index[phisn0 + Mn+1], neq.bc_phis(phis_ne[Mn], phis_ne[Mn+1], Iapp))
    #    val = jax.ops.index_update(val, jax.ops.index[phisn0 + Mn+1], neq.bc_zero_dirichlet(phis_ne[Mn], phis_ne[Mn+1]))
        return val
    
    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_j(self,val, jvec_pe, uvec_pe, Tvec_pe, eta_pe, cmat_pe, jvec_ne, uvec_ne, Tvec_ne, eta_ne, cmat_ne):
        jp0 = self.jp0
        peq= self.peq
        Mp = self.Mp
        jn0  = self.jn0
        neq = self.neq
        Mn = self.Mn
        Np = self.Np; Nn = self.Nn
        val = jax.ops.index_update(val, jax.ops.index[jp0:jp0 + Mp], vmap(peq.ionic_flux)(jvec_pe, uvec_pe[1:Mp+1], Tvec_pe[1:Mp+1], eta_pe, cmat_pe[Np, :], cmat_pe[Np+1,:], peq.cmax*np.ones(Mp)))
        val = jax.ops.index_update(val, jax.ops.index[jn0: jn0 + Mn], vmap(neq.ionic_flux)(jvec_ne, uvec_ne[1:Mn+1], Tvec_ne[1:Mn+1], eta_ne, cmat_ne[Nn, :], cmat_ne[Nn+1,:], neq.cmax*np.ones(Mn)))
        return val
    
    

    
#    @jax.jit
    @partial(jax.jit, static_argnums=(0,))
    def res_eta(self,val, eta_pe, phis_pe, phie_pe, Tvec_pe, cmat_pe, eta_ne, phis_ne, phie_ne, Tvec_ne, cmat_ne):
        etap0 = self.etap0
        etan0 = self.etan0
        Mp = self.Mp; Mn = self.Mn; peq=self.peq; neq=self.neq; Np = self.Np; Nn = self.Nn
        val = jax.ops.index_update(val, jax.ops.index[etap0:etap0 + Mp], vmap(peq.over_poten)(eta_pe, phis_pe[1:Mp+1], phie_pe[1:Mp+1], Tvec_pe[1:Mp+1], cmat_pe[Np,:], cmat_pe[Np+1,:], peq.cmax*np.ones(Mp)))
        val = jax.ops.index_update(val, jax.ops.index[etan0: etan0 + Mn], vmap(neq.over_poten)(eta_ne, phis_ne[1:Mn+1], phie_ne[1:Mn+1], Tvec_ne[1:Mn+1], cmat_ne[Nn,:], cmat_ne[Nn+1,:], neq.cmax*np.ones(Mn)))
        return val


    #    val = np.zeros(Ntot)
    @partial(jax.jit, static_argnums=(0,))
    def fn(self,U,Uold):
        
        val= np.zeros(self.Ntot)
        Mp = self.Mp; Np = self.Np
        Mn = self.Mn; Nn = self.Nn; Ms = self.Ms; Ma = self.Ma; Mz=self.Mz
        cmat_pe, cmat_ne,\
        uvec_pe, uvec_sep, uvec_ne, \
        Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
        phie_pe, phie_sep, phie_ne, \
        phis_pe, phis_ne, jvec_pe,jvec_ne,eta_pe,eta_ne = unpack(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
        
        cmat_old_pe, cmat_old_ne,\
        uvec_old_pe, uvec_old_sep, uvec_old_ne,\
        Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc,\
        _, _, \
        _, _,\
        _,_,\
        _,_,_= unpack(Uold,Mp, Np, Mn, Nn, Ms, Ma, Mz)
        
        val = self.res_c_fn(val, cmat_pe, jvec_pe, Tvec_pe, cmat_old_pe, cmat_ne, jvec_ne, Tvec_ne, cmat_old_ne )
           
        val = self.res_u_pe(val, uvec_pe, Tvec_pe, jvec_pe, uvec_old_pe, uvec_sep, Tvec_sep)
        val = self.res_u_sep(val, uvec_sep, Tvec_sep, uvec_old_sep, uvec_pe, Tvec_pe, uvec_ne, Tvec_ne)
        val = self.res_u_ne(val, uvec_ne, Tvec_ne, jvec_ne, uvec_old_ne, uvec_sep, Tvec_sep)
        
        val = self.res_T_acc(val, Tvec_acc, Tvec_old_acc, Tvec_pe)
        val = self.res_T_pe(val, Tvec_pe, uvec_pe, phie_pe, phis_pe, jvec_pe, eta_pe, cmat_pe, Tvec_old_pe, Tvec_acc, Tvec_sep)
        val = self.res_T_sep(val, Tvec_sep, uvec_sep, phie_sep, Tvec_old_sep, Tvec_pe, Tvec_ne )
        val = self.res_T_ne(val, Tvec_ne, uvec_ne, phie_ne, phis_ne, jvec_ne, eta_ne, cmat_ne, Tvec_old_ne, Tvec_zcc, Tvec_sep)
        val = self.res_T_zcc(val, Tvec_zcc, Tvec_old_zcc, Tvec_ne)
        
        val = self.res_phie_pe(val, uvec_pe, phie_pe, Tvec_pe, jvec_pe, uvec_sep,phie_sep, Tvec_sep)
        #    val = res_phie_pe_phi(val, uvec_pe, phie_pe, Tvec_pe, phis_pe, uvec_sep,phie_sep, Tvec_sep)
        val = self.res_phie_sep(val, uvec_sep, phie_sep, Tvec_sep, phie_pe, phie_ne)
        val = self.res_phie_ne(val, uvec_ne, phie_ne, Tvec_ne, jvec_ne, uvec_sep, phie_sep, Tvec_sep)
        #    val = res_phie_ne_phi(val, uvec_ne, phie_ne, Tvec_ne, phis_ne, uvec_sep, phie_sep, Tvec_sep)
        
        val = self.res_phis(val, phis_pe, jvec_pe, phis_ne, jvec_ne)
        
        val = self.res_j(val, jvec_pe, uvec_pe, Tvec_pe, eta_pe, cmat_pe, jvec_ne, uvec_ne, Tvec_ne, eta_ne, cmat_ne)
        #    val = res_j_phi(val, jvec_pe, uvec_pe, Tvec_pe, phis_pe, phie_pe, cmat_pe, jvec_ne, uvec_ne, Tvec_ne, phis_ne, phie_ne, cmat_ne)
        val = self.res_eta(val, eta_pe, phis_pe, phie_pe, Tvec_pe, cmat_pe, eta_ne, phis_ne, phie_ne, Tvec_ne, cmat_ne)
        return val
        
