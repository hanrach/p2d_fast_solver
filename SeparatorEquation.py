#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:19:58 2020

@author: hanrach
"""
from dataclasses import dataclass
import coeffs
import jax.numpy as np
from jax import grad
from batterySection import Electrode, Separator, CurrentCollector, p_electrode_constants, p_electrode_grid_param,\
n_electrode_constants,n_electrode_grid_param,sep_constants,sep_grid_param,a_cc_constants,z_cc_constants,cc_grid_param
from settings import F, R, gamma, trans, Tref, delta_t
from jax.config import config
config.update("jax_enable_x64", True)

pe = Electrode(p_electrode_constants(),p_electrode_grid_param(), 25751, 51554)
ne = Electrode(n_electrode_constants(),n_electrode_grid_param(), 26128, 30555)
sep = Separator(sep_constants(), sep_grid_param())
acc = CurrentCollector(a_cc_constants(),cc_grid_param())
zcc = CurrentCollector(z_cc_constants(),cc_grid_param())

class SeparatorEquation:
    def __init__(self, constants, gridparam):
        self.rho = constants.rho;
        self.Cp = constants.Cp;
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.l = constants.l
        
        self.M = gridparam.M
        M = self.M;
        self.hx = self.l/M; 
#        self.hx = 1/M; self.hy=1/N 
        
    def Qohm(self, phien, phiep, un, up, uc, T):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        kapeff = coeffs.electrolyteConductCoeff(eps,brugg,uc,T)
        ans = kapeff*( (phiep - phien)/(2*hx) )**2 + \
        (2*kapeff*R*T/F)*(1-trans)*( (np.log(up) - np.log(un))/(2*hx) )*( (phiep - phien)/(2*hx) )
        return ans
        
    def electrolyte_conc(self,un, uc, up, Tn, Tc, Tp, uold):
        eps = self.eps; brugg = self.brugg; hx = self.hx

        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        Deff_r = coeffs.electrolyteDiffCoeff(eps,brugg,umid_r,Tmid_r);
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,umid_l,Tmid_l);
 
        ans = (uc-uold) -  (delta_t/eps)*( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx 
        return ans.reshape()
    
    def Du_Dun(self,un,uc,Tn,Tc):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        umid_l = (un+uc)/2;
        Tmid_l = (Tn+Tc)/2;
    
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,umid_l,Tmid_l);
        Deff_l_Du = grad(coeffs.electrolyteDiffCoeff,(2))(eps,brugg,umid_l,Tmid_l)
        ans = (delta_t/(eps*hx**2))*(Deff_l_Du*(uc-un) + Deff_l)
        return ans

    
    # boundary condition for positive electrode
    def bc_u_sep_p(self,u0_pe,u1_pe,T0_pe,T1_pe,\
             u0_sep,u1_sep,T0_sep,T1_sep):
        eps_p = pe.eps; eps_s = self.eps;
        brugg_p = pe.brugg; brugg_s = self.brugg;
        Deff_pe = coeffs.electrolyteDiffCoeff(eps_p,brugg_p,(u0_pe + u1_pe)/2,(T0_pe + T1_pe)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        bc = -Deff_pe*(u1_pe - u0_pe)/pe.hx + Deff_sep*(u1_sep - u0_sep)/sep.hx
        return bc.reshape()

    # boundary condition for negative electrode
    def bc_u_sep_n(self,u0_ne,u1_ne,T0_ne,T1_ne,\
                 u0_sep,u1_sep,T0_sep,T1_sep):
        eps_n = ne.eps; eps_s = self.eps;
        brugg_n = ne.brugg; brugg_s = self.brugg;
        
        Deff_ne = coeffs.electrolyteDiffCoeff(eps_n,brugg_n,(u0_ne + u1_ne)/2,(T0_ne + T1_ne)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        
        bc = -Deff_sep*(u1_sep - u0_sep)/sep.hx + Deff_ne*(u1_ne - u0_ne)/ne.hx
        return bc.reshape()
    
    def electrolyte_poten(self,un, uc, up, phien, phiec, phiep, Tn, Tc, Tp):
    
        eps = self.eps; brugg = self.brugg;
        hx = self.hx; 
        
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        
        kapeff_r = coeffs.electrolyteConductCoeff(eps,brugg,umid_r,Tmid_r);
        kapeff_l = coeffs.electrolyteConductCoeff(eps,brugg,umid_l,Tmid_l);
        
        ans = - ( kapeff_r*(phiep - phiec)/hx - kapeff_l*(phiec - phien)/hx )/hx + gamma*( kapeff_r*Tmid_r*(np.log(up) - np.log(uc))/hx  \
            - kapeff_l*Tmid_l*(np.log(uc) - np.log(un))/hx )/hx
        return ans.reshape()
    
    def bc_phie_ps(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s, T0_p, T1_p, T0_s, T1_s):
        kapeff_p = coeffs.electrolyteConductCoeff(pe.eps,pe.brugg,(u0_p + u1_p)/2,(T0_p + T1_p)/2);
        kapeff_s = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2);
        bc = -kapeff_p*(phie1_p - phie0_p)/pe.hx + kapeff_s*(phie1_s - phie0_s)/sep.hx
        return bc.reshape()
    
    def bc_phie_sn(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s, T0_n, T1_n, T0_s, T1_s):
        
        kapeff_n = coeffs.electrolyteConductCoeff(ne.eps,ne.brugg,(u0_n + u1_n)/2,(T0_n + T1_n)/2);
        
        kapeff_s = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2);
        bc = -kapeff_s*(phie1_s - phie0_s)/sep.hx + kapeff_n*(phie1_n - phie0_n)/ne.hx
        return bc.reshape()

    def temperature(self, un, uc, up, phien, phiep, Tn, Tc, Tp, Told):
        hx = self.hx
#        ans = (Tc - Told) -  (delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
#        self.Qohm( phien, phiep, un, up, uc, Tc) )
        ans = (Tc - Told) -  (delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
        self.Qohm( phien, phiep, un, up, uc, Tc) )
        return ans.reshape()
    
    def bc_temp_ps(self,T0_p, T1_p, T0_s, T1_s):
        bc = -pe.lam*(T1_p - T0_p)/pe.hx + sep.lam*(T1_s - T0_s)/sep.hx
        return bc.reshape()
    
    def bc_temp_sn(self,T0_s, T1_s, T0_n, T1_n):
        bc = -sep.lam*(T1_s - T0_s)/sep.hx + ne.lam*(T1_n - T0_n)/ne.hx
        return bc.reshape()
@dataclass
class grid_param_pack:
    M: int;
    
@dataclass
class separator_constants:    
    rho: float; Cp:float;
    eps: float; lam: float; 
    brugg: float; 
    l:float; 
    
def sep_constants():
    rho = 1100;
    Cp = 700;
    eps = 0.724
    lam = 0.16;
    brugg = 4;
    l = 2.5*1e-5;
#    l = 8*1e-5;
    return separator_constants(rho, Cp, eps, lam, brugg, l)

def sep_grid_param(M):
    
    return grid_param_pack(M)

#sepq = SeparatorEquation(sep_constants(),sep_grid_param())