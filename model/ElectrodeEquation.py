#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:50:15 2020

@author: hanrach
"""

import model.coeffs as coeffs
import jax.numpy as np
from model.settings import F, R, gamma, trans, Tref
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial
import jax

class ElectrodeEquation:   
    
    def __init__(self, constants, gridparam, electrode_type, s_constants, s_grid, a_constants, acc_grid, z_constants, zcc_grid, cavg, cmax, delta_t):
        self.cavg = cavg
        self.cmax = cmax
        self.electrode_type = electrode_type
        self.rho = constants.rho;
        self.Cp = constants.Cp;
        self.sigma = constants.sigma
        self.epsf = constants.epsf
        self.eps = constants.eps
        self.lam = constants.lam
        self.brugg = constants.brugg
        self.a = constants.a
        self.Rp = constants.Rp
        self.k = constants.k
        self.l = constants.l
        self.Ds = constants.Ds;
        self.sigma = constants.sigma
        self.Ek = constants.Ek
        self.ED = constants.ED
        self.Ds = constants.Ds
        self.delta_t=delta_t
        self.sep = s_constants
        self.sep_hx = self.sep.l/s_grid.M
        
        self.acc = a_constants
        self.acc_hx = self.acc.l/acc_grid.M
        
        self.zcc=z_constants
        self.zcc_hx = self.zcc.l/zcc_grid.M
        
        self.N = gridparam.N; self.M = gridparam.M
        N = self.N; M = self.M;
        self.hx = self.l/M; self.hy = self.Rp/N


        self.sigeff  = self.sigma*(1-self.eps - self.epsf)
        
        self.rpts_end = np.arange(0,N)*self.Rp/N
        self.rpts_mid = self.Rp*(np.arange(0,N)-0.5)/N
        
        self.rpts_end = self.Rp*np.linspace(0,N, N+1)/N
        self.rpts_mid = self.Rp*(np.linspace(1,N, N)-0.5)/N
        R = self.Rp*(np.linspace(1,N,N)-(1/2))/N;
#        R = np.arange(0,self.Rp + self.hy, self.hy) + self.hy/2 ;
#        R = R[0:-1]
        r = self.Rp*(np.linspace(0,N, N+1))/N
#        r = np.arange(0,self.Rp + self.hy, self.hy) + self.hy; 
#        self.rpts_end = self.Rp*np.linspace(0,N, N+1)/N
#        self.rpts_mid = self.Rp*(np.linspace(1,N, N)-0.5)/N
#        lambda1 = k*r[0:N]**2/R**2/hy**2;
#        lambda2 = k*r[1:N+1]**2/R**2/hy**2;
        self.lambda1 = delta_t*r[0:N]**2/(R**2*self.hy**2);
        self.lambda2 = delta_t*r[1:N+1]**2/(R**2*self.hy**2);
        
    """ Heat source terms """
    
    def Qohm(self,phisn, phisp, phien, phiep, un, up, uc, T):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        sigeff = self.sigma*(1-self.eps-self.epsf)
        kapeff = coeffs.electrolyteConductCoeff(eps,brugg,uc,T)
        
        ans = sigeff*( (phisp - phisn)/(2*hx) )**2 + kapeff*( (phiep - phien)/(2*hx) )**2 + \
        (2*kapeff*R*T/F)*(1-trans)*( (np.log(up) - np.log(un))/(2*hx) )*( (phiep - phien)/(2*hx) )
        
        return ans

    def Qrxn(self,j,eta):
        ans = F*self.a*j*eta
        return ans
    
    def Qrxn_phi(self,j,phis,phie,T,cM,cMp,cmax):
        ans = F*self.a*j*self.over_poten_exp(phis,phie,T,cM,cMp,cmax)
        return ans
    
    def Qrev(self,j,T,cM, cMp,cmax):
        ans = F*self.a*j*T*self.entropy_change(cM, cMp,T,cmax)
        return ans
    
    def Qrev_fast(self,j,T,cs,cmax):
        ans = F*self.a*j*T*self.entropy_change_fast(cs,T,cmax)
        return ans
    
    """ Equations for c"""
    def solid_conc(self,cn,cc,cp, cold, lambda1, lambda2):
        Ds = self.Ds
        ans = (cc-cold)  + Ds*( cc*(lambda2 + lambda1) - lambda2*cp - lambda1*cn)
        return ans
    


    def solid_conc_2(self,cn,cc,cp, cold):
        hy = self.hy
        Ds = self.Ds
        
        lambda1  = self.delta_t*self.rpts_end[0:self.N]**2/self.rpts_mid**2/hy**2;
        lambda2  = self.delta_t*self.rpts_end[1:self.N+1]**2/self.rpts_mid**2/hy**2;
        
        k = self.delta_t;
        N = self.N
        R = self.Rp*(np.linspace(1,N,N)-(1/2))/N;
#        R = np.arange(0,self.Rp + self.hy, self.hy) + self.hy/2 ;
#        R = R[0:-1]
        r = self.Rp*(np.linspace(0,N, N+1))/N
#        r = np.arange(0,self.Rp + self.hy, self.hy) + self.hy; 
#        self.rpts_end = self.Rp*np.linspace(0,N, N+1)/N
#        self.rpts_mid = self.Rp*(np.linspace(1,N, N)-0.5)/N
#        lambda1 = k*r[0:N]**2/R**2/hy**2;
#        lambda2 = k*r[1:N+1]**2/R**2/hy**2;
        lambda1 = k*r[0:N]**2/(R**2*hy**2);
        lambda2 = k*r[1:N+1]**2/(R**2*hy**2);
#(c1 - cold) + c1*(lambda2 + lambda1) - lambda2*c2 - lambda1*c0
        ans = (cc-cold)  + Ds*( cc*(lambda2 + lambda1) - lambda2*cp - lambda1*cn)
        return ans
    
    def bc_neumann_c(self,c0,c1,jvec,T):
        hy = self.hy
        Deff = coeffs.solidDiffCoeff(self.Ds,self.ED,T)
        bc = (c1 - c0)/hy + jvec/Deff
#        bc = (c1-c0) + jvec*hy/Deff
        return bc.reshape()
    
    """ Equations for u """
    @partial(jax.jit, static_argnums=(0,))
    def electrolyte_conc(self,un, uc, up, Tn, Tc, Tp, j,uold):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        a = self.a
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        Deff_r = coeffs.electrolyteDiffCoeff(eps,brugg,umid_r,Tmid_r);
        Deff_l = coeffs.electrolyteDiffCoeff(eps,brugg,umid_l,Tmid_l);
        
        
        ans = (uc-uold) - (self.delta_t/eps)*( ( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx + a*(1-trans)*j ) 
    
        return ans.reshape()
    
    def bc_zero_neumann(self,u0, u1):
        bc =  u1 - u0
        return bc.reshape()

    def bc_const_dirichlet(self,u0, u1, constant):
        bc =  (u1 + u0)/2 - constant
        return bc.reshape()
    
    # boundary condition for positive electrode
    def bc_u_sep_p(self,u0_pe,u1_pe,T0_pe,T1_pe,\
             u0_sep,u1_sep,T0_sep,T1_sep):
        
        eps_p = self.eps; eps_s = self.sep.eps;
        brugg_p = self.brugg; brugg_s = self.sep.brugg;
        
#        Deff_pe = coeffs.electrolyteDiffCoeff(eps_p,brugg_p,(u0_pe + u1_pe)/2,(T0_pe + T1_pe)/2)
#        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        
        Deff_pe = (coeffs.electrolyteDiffCoeff(eps_p,brugg_p,u0_pe,T0_pe) + coeffs.electrolyteDiffCoeff(eps_p,brugg_p,u1_pe,T1_pe))/2
        Deff_sep =( coeffs.electrolyteDiffCoeff(eps_s,brugg_s,u0_sep,T0_sep) + coeffs.electrolyteDiffCoeff(eps_s,brugg_s,u1_sep,T1_sep))/2          
        
        bc = -Deff_pe*(u1_pe - u0_pe)/self.hx + Deff_sep*(u1_sep - u0_sep)/self.sep_hx
#        bc = -Deff_pe*(u1_pe - u0_pe)*sep.hx + Deff_sep*(u1_sep - u0_sep)*pe.hx
        return bc.reshape()
    
    def bc_inter_cont(self, u0_pe, u1_pe, u0_sep, u1_sep):
        ans = (u0_pe+u1_pe)/2 - (u0_sep+u1_sep)/2
        return ans.reshape()

    # boundary condition for negative electrode
    def bc_u_sep_n(self,u0_ne,u1_ne,T0_ne,T1_ne,\
                 u0_sep,u1_sep,T0_sep,T1_sep):
        
        eps_n = self.eps; eps_s = self.sep.eps;
        brugg_n = self.brugg; brugg_s = self.sep.brugg;
        
        Deff_ne = coeffs.electrolyteDiffCoeff(eps_n,brugg_n,(u0_ne + u1_ne)/2,(T0_ne + T1_ne)/2)
        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        bc = -Deff_sep*(u1_sep - u0_sep)/self.sep_hx + Deff_ne*(u1_ne - u0_ne)/self.hx
        return bc.reshape()
    
    """ Electrolyte potential equations: phie """
    
    def electrolyte_poten(self,un, uc, up, phien, phiec, phiep, Tn, Tc, Tp,j):
    
        eps = self.eps; brugg = self.brugg;
        hx = self.hx; 
        a = self.a;
        
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        
        kapeff_r = coeffs.electrolyteConductCoeff(eps,brugg,umid_r,Tmid_r);
        kapeff_l = coeffs.electrolyteConductCoeff(eps,brugg,umid_l,Tmid_l);
        
        ans = a*F*j + (kapeff_r*(phiep - phiec)/hx \
                    - kapeff_l*(phiec - phien)/hx)/hx \
        - gamma*(kapeff_r*Tmid_r*(np.log(up) - \
                                               np.log(uc))/hx  \
            - kapeff_l*Tmid_l*(np.log(uc) - \
                                               np.log(un))/hx )/hx
        return ans.reshape()
    
    def electrolyte_poten_phi(self,un, uc, up, phien, phiec, phiep, Tn, Tc, Tp,phisn, phisc, phisp):

        eps = self.eps; brugg = self.brugg;
        hx = self.hx; 
#        a = self.a;
        sigeff = self.sigma*(1-self.eps-self.epsf)
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        
        kapeff_r = coeffs.electrolyteConductCoeff(eps,brugg,umid_r,Tmid_r);
        kapeff_l = coeffs.electrolyteConductCoeff(eps,brugg,umid_l,Tmid_l);
        
        ans = sigeff*( phisn - 2*phisc + phisp)/hx**2 + (kapeff_r*(phiep - phiec)/hx \
                    - kapeff_l*(phiec - phien)/hx)/hx \
        - gamma*(kapeff_r*Tmid_r*(np.log(up) - \
                                               np.log(uc))/hx  \
            - kapeff_l*Tmid_l*(np.log(uc) - \
                                               np.log(un))/hx )/hx
        return ans.reshape()

    def bc_zero_dirichlet(self,phie0, phie1):
        ans= (phie0 + phie1)/2
        return ans.reshape()

    def bc_phie_p(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s, T0_p, T1_p, T0_s, T1_s):
        
        kapeff_p = coeffs.electrolyteConductCoeff(self.eps,self.brugg,(u0_p + u1_p)/2,(T0_p + T1_p)/2);
        kapeff_s = coeffs.electrolyteConductCoeff(self.sep.eps,self.sep.brugg,(u0_s + u1_s)/2,(T0_s + T1_s)/2);
        
        bc = -kapeff_p*(phie1_p - phie0_p)/self.hx + kapeff_s*(phie1_s - phie0_s)/self.sep_hx
        return bc.reshape()
    
    def bc_phie_n(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s, T0_n, T1_n, T0_s, T1_s):
        kapeff_n = coeffs.electrolyteConductCoeff(self.eps, self.brugg, (u0_n + u1_n)/2,(T0_n + T1_n)/2);
        kapeff_s = coeffs.electrolyteConductCoeff(self.sep.eps, self.sep.brugg, (u0_s + u1_s)/2,(T0_s + T1_s)/2);
        
        bc = -kapeff_s*(phie1_s - phie0_s)/self.sep_hx + kapeff_n*(phie1_n - phie0_n)/self.hx
        return bc.reshape()
    
    """ Equations for solid potential phis"""
    def solid_poten(self,phisn, phisc, phisp, j):
        hx = self.hx; a = self.a
        sigeff = self.sigma*(1-self.eps-self.epsf)
        ans = ( phisn - 2*phisc + phisp) - (a*F*j*hx**2)/sigeff
        return ans.reshape()
    
    def bc_phis(self,phis0, phis1, source):
        sigeff = self.sigma*(1-self.eps-self.epsf)
        bc = ( phis1 - phis0 ) + self.hx*(source)/sigeff
        return bc.reshape()
    
    """ Equations for Temperature T"""
    def temperature(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cM, cMp, cmax, Told):
        hx = self.hx
        ans = (Tc - Told) -  (self.delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
        self.Qohm(phisn, phisp, phien, phiep, un, up, uc, Tc) + self.Qrxn(j,eta) + self.Qrev(j,Tc,cM, cMp,cmax) )
#        ans = (Tc - Told)*hx**2 -  (delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)+ \
#        (self.Qohm(phisn, phisp, phien, phiep, un, up, uc, Tc) + self.Qrxn(j,eta) + self.Qrev(j,Tc,cM, cMp,cmax))*hx**2 )
        return ans.reshape()
    
    def temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
        hx = self.hx
        cs = cs_1 - gamma_c*j/coeffs.solidDiffCoeff(self.Ds, self.ED, Tc)
        ans = (Tc - Told) -  (self.delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
        self.Qohm(phisn, phisp, phien, phiep, un, up, uc, Tc) + self.Qrxn(j,eta) + self.Qrev_fast(j,Tc,cs,cmax) )
        return ans.reshape()
    
    def temperature_phi(self,un, uc, up, phien, phiec, phiep, phisn, phisc, phisp, Tn, Tc, Tp,j, cM, cMp, cmax, Told):
        hx = self.hx
        ans = (Tc - Told) -  (self.delta_t/(self.rho*self.Cp))*( self.lam*( Tn - 2*Tc + Tp)/hx**2 + \
        self.Qohm(phisn, phisp, phien, phiep, un, up, uc, Tc) + self.Qrxn_phi(j,phisc,phiec,Tc,cMp,cMp,cmax) + self.Qrev(j,Tc,cM, cMp,cmax) )
        return ans.reshape()
    
    # boundary conditions
    def bc_temp_ap(self,T0_acc, T1_acc, T0_pe, T1_pe): 
        bc = -self.acc.lam*(T1_acc - T0_acc)/self.acc_hx + self.lam*(T1_pe - T0_pe)/self.hx
        return bc.reshape()
    
    def bc_temp_ps(self,T0_p, T1_p, T0_s, T1_s):
        bc = -self.lam*(T1_p - T0_p)/self.hx+ self.sep.lam*(T1_s - T0_s)/self.sep_hx
        return bc.reshape()
    
    def bc_temp_sn(self,T0_s, T1_s, T0_n, T1_n):
        bc = -self.sep.lam*(T1_s - T0_s)/self.sep_hx + self.lam*(T1_n - T0_n)/self.hx
        return bc.reshape()
    
    def bc_temp_n(self,T0_ne, T1_ne, T0_zcc, T1_zcc):
        bc = -self.lam*(T1_ne - T0_ne)/self.hx+ self.zcc.lam*(T1_zcc - T0_zcc)/self.zcc_hx
        return bc.reshape()
    
    """ ionic flux: j"""
    
    def ionic_flux(self,j,u,T,eta,cM, cMp,cmax):
        cs = (cM+cMp)/2
#        cs = self.cstar(cM,cMp,j,T)
        keff = self.k*np.exp( (-self.Ek/R)*((1/T) - (1/Tref)))
        var = ((0.5*F)/(R*T))*eta
        term2 = (np.exp(var)-np.exp(-var))/2
        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*term2
#        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*np.sinh( (0.5*F/(R*T))*(eta) )
        return ans.reshape()
    
    def ionic_flux_fast(self,j,u,T,eta,cs1, gamma_c,cmax):
        cs = cs1 - gamma_c*j/coeffs.solidDiffCoeff(self.Ds, self.ED,T )
        keff = self.k*np.exp( (-self.Ek/R)*((1/T) - (1/Tref)))
        var = ((0.5*F)/(R*T))*eta
        term2 = (np.exp(var)-np.exp(-var))/2
        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*term2
    #        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*np.sinh( (0.5*F/(R*T))*(eta) )
        return ans.reshape()
    
    def ionic_flux_phi(self,j,u,T,phis,phie,cM, cMp,cmax):
        cs = (cM+cMp)/2
#        cs = self.cstar(cM,cMp,j,T)
        keff = self.k*np.exp( (-self.Ek/R)*((1/T) - (1/Tref)))
        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*np.sinh( (0.5*F/(R*T))*(self.over_poten_exp(phis,phie, T,  cM, cMp, cmax)) )
        return ans.reshape()
    
    
    """ over potential : eta """
    
    def open_circ_poten_ref(self,cM, cMp,T, cmax):
        cs = (cM+cMp)/2
#        cs = self.cstar(cM,cMp,j,T)
        theta = cs/cmax;
        if (self.electrode_type == "positive"):
            ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
            (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
        else:
            ans = 0.7222 + 0.1387*theta + 0.029*theta**(0.5) - 0.0172/theta + 0.0019/(theta**1.5) + 0.2808*np.exp(0.9 - 15*theta) - 0.7984*np.exp(0.4465*theta - 0.4108)
        return ans
    
    def open_circ_poten_ref_fast(self,cs,T, cmax):
    #        cs = self.cstar(cM,cMp,j,T)
        theta = cs/cmax;
        if (self.electrode_type == "positive"):
            ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
            (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
        else:
            ans = 0.7222 + 0.1387*theta + 0.029*theta**(0.5) - 0.0172/theta + 0.0019/(theta**1.5) + 0.2808*np.exp(0.9 - 15*theta) - 0.7984*np.exp(0.4465*theta - 0.4108)
        return ans
    
    
    def open_circuit_poten(self,cM, cMp,T,cmax):
        Uref = self.open_circ_poten_ref(cM, cMp,T,cmax)
        ans = Uref + (T - Tref)*self.entropy_change(cM, cMp,T,cmax)
        return ans
    
    def open_circuit_poten_fast(self,cs,T,cmax):
        Uref = self.open_circ_poten_ref_fast(cs,T,cmax)
        ans = Uref + (T - Tref)*self.entropy_change_fast(cs,T,cmax)
        return ans
    

    def entropy_change(self,cM,cMp,T,cmax):
        cs = (cM+cMp)/2
#        cs = self.cstar(cM,cMp,T)
        theta = cs/cmax
        if (self.electrode_type == "positive"):
            ans = -0.001*( (0.199521039 - 0.92837822*theta + 1.364550689000003*theta**2 - 0.6115448939999998*theta**3)/\
            (1 - 5.661479886999997*theta + 11.47636191*theta**2 - 9.82431213599998*theta**3 + \
             3.046755063*theta**4))
        else:
            # typo for + 38379.18127*theta**7 
            ans = 0.001*(0.005269056 + 3.299265709*theta - 91.79325798*theta**2 + \
                         1004.911008*theta**3 - 5812.278127*theta**4 + \
                         19329.7549*theta**5 - 37147.8947*theta**6 + 38379.18127*theta**7 - \
                         16515.05308*theta**8)/(1 - 48.09287227*theta + 1017.234804*theta**2 - 10481.80419*theta**3 +\
                                             59431.3*theta**4 - 195881.6488*theta**5 + 374577.3152*theta**6 -\
                                             385821.1607*theta**7 + 165705.8597*theta**8)
            
        return ans
    
    def entropy_change_fast(self,cs,T,cmax):
        theta = cs/cmax
        if (self.electrode_type == "positive"):
            ans = -0.001*( (0.199521039 - 0.92837822*theta + 1.364550689000003*theta**2 - 0.6115448939999998*theta**3)/\
            (1 - 5.661479886999997*theta + 11.47636191*theta**2 - 9.82431213599998*theta**3 + \
             3.046755063*theta**4))
        else:
        # typo for + 38379.18127*theta**7 
            ans = 0.001*(0.005269056 + 3.299265709*theta - 91.79325798*theta**2 + \
             1004.911008*theta**3 - 5812.278127*theta**4 + \
             19329.7549*theta**5 - 37147.8947*theta**6 + 38379.18127*theta**7 - \
             16515.05308*theta**8)/(1 - 48.09287227*theta + 1017.234804*theta**2 - 10481.80419*theta**3 +\
             59431.3*theta**4 - 195881.6488*theta**5 + 374577.3152*theta**6 -\
             385821.1607*theta**7 + 165705.8597*theta**8)
        
        return ans
   
    def cstar(self, cM, cMp, j, T):
        cs = -self.hy*j/(coeffs.solidDiffCoeff(self.Ds,self.ED,T)) + (2*cM - cMp)/2
        return cs
        
    def over_poten(self,eta, phis,phie, T, cM, cMp, cmax):
        ans = eta - phis + phie + self.open_circuit_poten(cM, cMp,T,cmax);
        return ans.reshape()
    
    def over_poten_fast(self,eta, phis,phie, T, j,cs1, gamma_c, cmax):
        cs = cs1 - gamma_c*j/coeffs.solidDiffCoeff(self.Ds, self.ED, T)
        ans = eta - phis + phie + self.open_circuit_poten_fast(cs,T,cmax);
        return ans.reshape()
    
    def over_poten_exp(self,phis,phie, T,  cM, cMp, cmax):
        ans = phis - phie - self.open_circuit_poten(cM,cMp,T,cmax);
        return ans.reshape()
    
    

