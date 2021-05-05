#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:19:58 2020

@author: hanrach
"""

import coeffs
import jax.numpy as np
from jax import grad
from settings import F, R, gamma, trans
from jax.config import config
config.update("jax_enable_x64", True)

class SeparatorEquation:
    def __init__(self, constants, gridparam, p_constants, n_constants, p_grid, n_grid, delta_t):
        self.rho = constants.rho;
        self.Cp = constants.Cp;
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.l = constants.l
        self.delta_t = delta_t
        self.M = gridparam.M
        M = self.M;
        self.hx = self.l/M; 
#        self.hx = 1/M; self.hy=1/N 
        self.pe = p_constants
        self.ne = n_constants
        self.pe_hx = self.pe.l/p_grid.M
        self.ne_hx = self.ne.l/n_grid.M
        
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
 
        ans = (uc-uold) -  (self.delta_t/eps)*( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx 
        return ans.reshape()
    
    def Du_Dun(self,un,uc,Tn,Tc):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        umid_l = (un+uc)/2;
        Tmid_l = (Tn+Tc)/2;
    
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,umid_l,Tmid_l);
        Deff_l_Du = grad(coeffs.electrolyteDiffCoeff,(2))(eps,brugg,umid_l,Tmid_l)
        ans = (self.delta_t/(eps*hx**2))*(Deff_l_Du*(uc-un) + Deff_l)
        return ans

    
    # boundary condition for positive electrode
    def bc_u_sep_p(self,u0_pe,u1_pe,T0_pe,T1_pe,\
             u0_sep,u1_sep,T0_sep,T1_sep):
        eps_p = self.pe.eps; eps_s = self.eps;
        brugg_p = self.pe.brugg; brugg_s = self.brugg;
        Deff_pe = coeffs.electrolyteDiffCoeff(eps_p,brugg_p,(u0_pe + u1_pe)/2,(T0_pe + T1_pe)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        bc = -Deff_pe*(u1_pe - u0_pe)/self.pe_hx + Deff_sep*(u1_sep - u0_sep)/self.hx
        return bc.reshape()

    # boundary condition for negative electrode
    def bc_u_sep_n(self,u0_ne,u1_ne,T0_ne,T1_ne,\
                 u0_sep,u1_sep,T0_sep,T1_sep):
        eps_n = self.ne.eps; eps_s = self.eps;
        brugg_n = self.ne.brugg; brugg_s = self.ne.brugg;
        
        Deff_ne = coeffs.electrolyteDiffCoeff(eps_n,brugg_n,(u0_ne + u1_ne)/2,(T0_ne + T1_ne)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        
        bc = -Deff_sep*(u1_sep - u0_sep)/self.hx + Deff_ne*(u1_ne - u0_ne)/self.ne_hx
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
        kapeff_p = coeffs.electrolyteConductCoeff(self.pe.eps,self.pe.brugg,(u0_p + u1_p)/2,(T0_p + T1_p)/2);
        kapeff_s = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2);
        bc = -kapeff_p*(phie1_p - phie0_p)/self.pe_hx + kapeff_s*(phie1_s - phie0_s)/self.hx
        return bc.reshape()
    
    def bc_phie_sn(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s, T0_n, T1_n, T0_s, T1_s):
        
        kapeff_n = coeffs.electrolyteConductCoeff(self.ne.eps,self.ne.brugg,(u0_n + u1_n)/2,(T0_n + T1_n)/2);
        
        kapeff_s = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2);
        bc = -kapeff_s*(phie1_s - phie0_s)/self.hx + kapeff_n*(phie1_n - phie0_n)/self.ne_hx
        return bc.reshape()

    def temperature(self, un, uc, up, phien, phiep, Tn, Tc, Tp, Told):
        hx = self.hx
#        ans = (Tc - Told) -  (delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
#        self.Qohm( phien, phiep, un, up, uc, Tc) )
        ans = (Tc - Told) -  (self.delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
        self.Qohm( phien, phiep, un, up, uc, Tc) )
        return ans.reshape()
    
    def bc_temp_ps(self,T0_p, T1_p, T0_s, T1_s):
        bc = -self.pe.lam*(T1_p - T0_p)/self.pe_hx + self.lam*(T1_s - T0_s)/self.hx
        return bc.reshape()
    
    def bc_temp_sn(self,T0_s, T1_s, T0_n, T1_n):
        bc = -self.lam*(T1_s - T0_s)/self.hx + self.ne.lam*(T1_n - T0_n)/self.ne_hx
        return bc.reshape()


#sepq = SeparatorEquation(sep_constants(),sep_grid_param())