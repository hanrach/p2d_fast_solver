# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:25:53 2020

@author: hanrach
"""

import numpy as onp
from jax.config import config
config.update('jax_enable_x64', True)
from decoupled.init import p2d_init_fast
from decoupled.p2d_newton_fast import newton_fast_sparse
from utils.build_der import *
from scipy.linalg import solve_banded
from model.p2d_param import get_battery_sections
from utils.precompute_c import precompute
import matplotlib.pylab as plt
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csc_matrix
import jax.numpy as np
from model.settings import Tref
from utils.reorder import reorder_tot
from utils.banded_matrix import diagonal_form
from utils.unpack import unpack_fast
from utils.derivatives import compute_jac, partials
Np = 20
Nn = 20
Mp = 20
Ms = 5
Mn = 20
Ma = 5
Mz = 5
Iapp = -30
# i=0
peq, neq, sepq, accq, zccq = get_battery_sections(Np, Nn, Mp, Mn, Ms, Ma, Mz, 10, Iapp)
Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq, neq)
gamma_p_vec = gamma_p * np.ones(Mp)
gamma_n_vec = gamma_n * np.ones(Mn)
lu_p = splu(csc_matrix(Ap))
lu_n = splu(csc_matrix(An))

U_fast = np.hstack(
    [

        1000 + np.zeros(Mp + 2),
        1000 + np.zeros(Ms + 2),
        1000 + np.zeros(Mn + 2),

        np.zeros(Mp),
        np.zeros(Mn),
        np.zeros(Mp),
        np.zeros(Mn),

        np.zeros(Mp + 2) + peq.open_circuit_poten(peq.cavg, peq.cavg, Tref, peq.cmax),
        np.zeros(Mn + 2) + neq.open_circuit_poten(neq.cavg, neq.cavg, Tref, neq.cmax),

        np.zeros(Mp + 2) + 0,
        np.zeros(Ms + 2) + 0,
        np.zeros(Mn + 2) + 0,

        Tref + np.zeros(Ma + 2),
        Tref + np.zeros(Mp + 2),
        Tref + np.zeros(Ms + 2),
        Tref + np.zeros(Mn + 2),
        Tref + np.zeros(Mz + 2)

    ])

c_pe = peq.cavg * np.ones(Mp * (Np + 2))
c_ne = neq.cavg * np.ones(Mn * (Nn + 2))

lu = {"pe": lu_p, "ne": lu_n}


def cmat_format(cmat, M, N):
    val = cmat
    for i in range(0, M):
        val = jax.ops.index_update(val, jax.ops.index[i * (N + 2)], 0)
        val = jax.ops.index_update(val, jax.ops.index[i * (N + 2) + N + 1], 0)
    return val


cmat_rhs_pe = cmat_format(c_pe, Mp, Np)
cmat_rhs_ne = cmat_format(c_ne, Mn, Nn)
lu_pe = lu["pe"];
lu_ne = lu["ne"]
cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
cI_pe_reshape = np.reshape(cI_pe_vec, [Np + 2, Mp], order="F");
cs_pe1 = np.asarray((cI_pe_reshape[Np, :] + cI_pe_reshape[Np + 1, :]) / 2)
cI_ne_reshape = np.reshape(cI_ne_vec, [Nn + 2, Mn], order="F");
cs_ne1 = np.asarray((cI_ne_reshape[Nn, :] + cI_ne_reshape[Nn + 1, :]) / 2)

fn_fast, jac_fn_fast = p2d_init_fast(Np, Nn, Mp, Mn, Ms, Ma, Mz, 10, Iapp)

U_fast_new, info = newton_fast_sparse(fn_fast, jac_fn_fast, U_fast, cs_pe1, cs_ne1, gamma_p_vec,gamma_n_vec)

Jfast1 = jac_fn_fast(U_fast_new, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec)
yfast1 = fn_fast(U_fast_new, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec)

idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
re_idx = np.argsort(idx_tot)
Jreorder = np.zeros([len(U_fast), len(U_fast)])

# from p2d_reorder_newton import newton
d = partials(accq, peq, sepq, neq, zccq)
grid_num = (Mp, Np, Mn, Nn, Ms, Ma, Mz)
vars = (peq, neq, Iapp)
jacfn = compute_jac(gamma_p_vec, gamma_n_vec, d, grid_num, vars)
# U_newton, info = newton(fn_fast, jacfn, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec, idx_tot, re_idx)


Jreorder = Jfast1[:, idx_tot]
Jreorder = Jreorder[idx_tot, :]

Jreorder_onp = onp.asarray(Jreorder)
Jab = diagonal_form((11, 11), Jreorder_onp)
idx_original = np.argsort(idx_tot)
yfast_reorder = yfast1[idx_tot]
yab = solve_banded((11, 11), Jab, yfast_reorder)


uvec_pe, uvec_sep, uvec_ne, \
Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
phie_pe, phie_sep, phie_ne, \
phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U_fast_new, Mp, Np, Mn, Nn, Ms, Ma, Mz)
uvec_old_pe, uvec_old_sep, uvec_old_ne, \
Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc, \
phie_old_pe, phie_old_sep, phie_old_ne, \
phis_old_pe, phis_old_ne, jvec_old_pe, jvec_old_ne, eta_old_pe, eta_old_ne = unpack_fast(U_fast, Mp, Np, Mn, Nn, Ms, Ma,
                                                                                         Mz)


def find_indices_j(dgrad):
    dup_idx = {}
    for (i, partials) in enumerate(dgrad):
        idx = []
        for (derivative) in (partials):
            temp = np.where(np.isclose(derivative, Jab, atol=1e-12))
            idx.append((temp[0], temp[1]))
        dup_idx[i] = idx
    return dup_idx


def find_indices(dgrad):
    dup_idx = {}
    for (i, partials) in enumerate(dgrad):
        idx = []
        for (derivative) in (partials):
            temp = np.where(np.isclose(derivative, Jab, atol=1e-16))
            idx.append((temp[0], temp[1]))
        dup_idx[i] = idx
    return dup_idx


def find_bc_idx(dgrad):
    idx_bc = [];
    for deriv in dgrad:
        temp = np.where(np.isclose(deriv, Jab, atol=1e-16))
        idx_bc.append((temp[0], temp[1]))
    return idx_bc


arg_up = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2], Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2],
          jvec_pe[0:Mp], uvec_old_pe[1:Mp + 1]]
dup = jax.vmap(jax.grad(peq.electrolyte_conc, argnums=range(0, 8)))(*arg_up)
bc_u0p = peq.bc_zero_neumann(uvec_pe[0], uvec_pe[1])
dup_bc0 = jax.grad(peq.bc_zero_neumann, argnums=(0, 1))(uvec_pe[0], uvec_pe[1])
dup_bcM = jax.grad(peq.bc_u_sep_p, argnums=range(0, 8))(uvec_pe[Mp], uvec_pe[Mp + 1], Tvec_pe[Mp], Tvec_pe[Mp + 1], \
                                                        uvec_sep[0], uvec_sep[1], Tvec_sep[0], Tvec_sep[1])

dup_bcM_idx = find_bc_idx(dup_bcM)

dup_idx = find_indices(dup)

# ionic_flux_fast(self,j,u,T,eta,cs1, gamma_c,cmax)


arg_jp = [jvec_pe[0:Mp], uvec_pe[1:Mp + 1], Tvec_pe[1:Mp + 1], eta_pe[0:Mp], cs_pe1, gamma_p_vec,
          peq.cmax * np.ones([Mp, 1])]
djp = jax.vmap(jax.grad(peq.ionic_flux_fast, argnums=range(0, 4)))(*arg_jp)
djp_idx = find_indices(djp)

arg_etap = [eta_pe[0:Mp], phis_pe[1:Mp + 1], phie_pe[1:Mp + 1], Tvec_pe[1:Mp + 1], jvec_pe[0:Mp], cs_pe1, gamma_p_vec,
            peq.cmax * np.ones([Mp, 1])]
detap = jax.vmap(jax.grad(peq.over_poten_fast, argnums=range(0, 5)))(*arg_etap)
detap_idx = find_indices(detap)

arg_phisp = [phis_pe[0:Mp], phis_pe[1:Mp + 1], phis_pe[2:Mp + 2], jvec_pe[0:Mp]]
dphisp = jax.vmap(jax.grad(peq.solid_poten, argnums=range(0, len(arg_phisp))))(*arg_phisp)
dphisp_idx = find_indices(dphisp)
dphisp_bc0 = jax.grad(peq.bc_phis, argnums=(0, 1))(phis_pe[0], phis_pe[1], Iapp)
dphisp_bcM = jax.grad(peq.bc_phis, argnums=(0, 1))(phis_pe[Mp], phis_pe[Mp + 1], 0)

dphisp_bc0_idx = find_bc_idx(dphisp_bc0)
dphisp_bcM_idx = find_bc_idx(dphisp_bcM)

arg_phiep = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2], phie_pe[0:Mp], phie_pe[1:Mp + 1], phie_pe[2:Mp + 2],
             Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2], jvec_pe[0:Mp]]
dphiep = jax.vmap(jax.grad(peq.electrolyte_poten, argnums=range(0, len(arg_phiep))))(*arg_phiep)
dphiep_idx = find_indices(dphiep)

dphiep_bc0 = jax.grad(peq.bc_zero_neumann, argnums=(0, 1))(phie_pe[0], phie_pe[1])
dphiep_bcM = jax.grad(peq.bc_phie_p, argnums=range(0, 12))(phie_pe[Mp], phie_pe[Mp + 1], phie_sep[0], phie_sep[1], \
                                                           uvec_pe[Mp], uvec_pe[Mp + 1], uvec_sep[0], uvec_sep[1], \
                                                           Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])

dphiep_bc0_idx = find_bc_idx(dphiep_bc0)
dphiep_bcM_idx = find_bc_idx(dphiep_bcM)

# temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
arg_Tp = arg_Tp = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2],
                   phie_pe[0:Mp], phie_pe[2:Mp + 2], \
                   phis_pe[0:Mp], phis_pe[2:Mp + 2],
                   Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2],
                   jvec_pe[0:Mp], \
                   eta_pe[0:Mp],
                   cs_pe1, gamma_p_vec, peq.cmax * np.ones([Mp, 1]),
                   Tvec_old_pe[1:Mp + 1]]

dTp = jax.vmap(jax.grad(peq.temperature_fast, argnums=range(0, 12)))(*arg_Tp)
dTp_idx = find_indices(dTp)

dTp_bc0 = jax.grad(peq.bc_temp_ap, argnums=range(0, 4))(Tvec_acc[Ma], Tvec_acc[Ma + 1], Tvec_pe[0], Tvec_pe[1])
dTp_bcM = jax.grad(peq.bc_temp_ps, argnums=range(0, 4))(Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])
dTp_bc0_idx = find_bc_idx(dTp_bc0)
dTp_bcM_idx = find_bc_idx(dTp_bcM)

# Separator
# u
dus = jax.vmap(jax.grad(sepq.electrolyte_conc, argnums=range(0, 6)))(uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                                                                     uvec_sep[2:Ms + 2], Tvec_sep[0:Ms],
                                                                     Tvec_sep[1:Ms + 1], Tvec_sep[2:Ms + 2],
                                                                     uvec_old_sep[1:Ms + 1])
dus_bc0 = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(uvec_sep[0], uvec_sep[1], uvec_pe[Mp], uvec_pe[Mp + 1])
dus_bcM = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(uvec_ne[0], uvec_ne[1], uvec_sep[Ms], uvec_sep[Ms + 1])

dus_idx = find_indices(dus)
dus_bc0_idx = find_bc_idx(dus_bc0)

# phie
dphies = jax.vmap(jax.grad(sepq.electrolyte_poten, argnums=range(0, 9)))(uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                                                                         uvec_sep[2:Ms + 2],
                                                                         phie_sep[0:Ms], phie_sep[1:Ms + 1],
                                                                         phie_sep[2:Ms + 2],
                                                                         Tvec_sep[0:Ms], Tvec_sep[1:Ms + 1],
                                                                         Tvec_sep[2:Ms + 2])
dphies_bc0 = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(phie_pe[Mp], phie_pe[Mp + 1], phie_sep[0], phie_sep[1], )
dphies_bcM = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(phie_ne[0], phie_ne[1], phie_sep[Ms], phie_sep[Ms + 1])

dphies_bc0_idx = find_bc_idx(dphies_bc0)
dphies_bcM_idx = find_bc_idx(dphies_bcM)
dphies_idx = find_indices(dphies)
# T
dTs = jax.vmap(jax.grad(sepq.temperature, argnums=range(0, 8)))(uvec_sep[0:Ms], uvec_sep[1:Ms + 1], uvec_sep[2:Ms + 2],
                                                                phie_sep[0:Ms], phie_sep[2:Ms + 2],
                                                                Tvec_sep[0:Ms], Tvec_sep[1:Ms + 1], Tvec_sep[2:Ms + 2],
                                                                Tvec_old_sep[1:Ms + 1])
dTs_bc0 = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])
dTs_bcM = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(Tvec_ne[0], Tvec_ne[1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
dTs_idx = find_indices(dTs)
dTs_bcM_idx = find_bc_idx(dTs_bcM)


def find_partial(d_idx, col_idx):
    N = len(d_idx)
    pairs = []
    for i in range(0, N):
        row_idx = np.where(d_idx[i][1] == col_idx[i])[0]
        pairs.append((d_idx[i][0][row_idx], col_idx[i]))
    return pairs


p0 = Ma + 1
sep0 = 4 * (Mp + 2) + 2 * Mp + 1 + p0
n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1

dus_bc0_pairs = find_partial(dus_bc0_idx, [sep0, sep0 + 3, 5 + 6 * (Mp - 1) + p0, 5 + 6 * Mp + p0])
dus_bcM_pairs = find_partial(dus_bc0_idx, [n0, n0 + 4, sep0 + 3 * Ms, sep0 + 3 * (Ms + 1)])

dphies_bc0_pairs = find_partial(dphies_bc0_idx,
                                [5 + 6 * (Mp - 1) + 4 + p0, 5 + 6 * (Mp) + 2 + p0, sep0 + 1, sep0 + 1 + 3])
dphies_bcM_pairs = find_partial(dphies_bcM_idx, [n0 + 2, n0 + 4 + 4, sep0 + 3 * Ms + 1, sep0 + 3 * Ms + 4])
dT_bc0_pairs = find_partial(dphies_bcM_idx, [5 + 6 * (Mp - 1) + 5 + p0, 5 + 6 * Mp + 3 + p0, sep0 + 2, sep0 + 5])
dT_bcM_pairs = find_partial(dphies_bcM_idx, [sep0 + 3 * Ms + 2, sep0 + 3 * Ms + 5, n0 + 3, n0 + 4 + 5])

bc_p = dict([('u', np.concatenate((np.array(dup_bc0), np.array(dup_bcM)))),
             ('phis', np.concatenate((np.array(dphisp_bc0), np.array(dphisp_bcM)))),
             ('phie', np.concatenate((np.array(dphiep_bc0), np.array(dphiep_bcM)))),
             ('T', np.concatenate((np.array(dTp_bc0), np.array(dTp_bcM))))])

# Negative electrode
dun_bc0 = jax.grad(neq.bc_u_sep_n, argnums=range(0, 8))(uvec_ne[0], uvec_ne[1], Tvec_ne[0], Tvec_ne[1],
                                                   uvec_sep[Ms], uvec_sep[Ms + 1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
dun = jax.vmap(jax.grad(neq.electrolyte_conc, argnums=range(0, 7)))(uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2],
                                                                    Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2],
                                                                    jvec_ne[0:Mn], uvec_old_ne[1:Mn + 1])
dun_bcM = jax.grad(neq.bc_zero_neumann, argnums=(0, 1))(uvec_ne[Mn], uvec_ne[Mn + 1])
dun_bc0_idx = find_bc_idx(dun_bc0)
dun_bcM_idx = find_bc_idx(dun_bcM)
dun_idx = find_indices(dun)

dun_bcM_pairs = find_partial(dun_bcM_idx, [n0 + 4 + 6 * (Mn - 1), n0 + 4 + 6 * Mn])

# negative j
arg_jn = [jvec_ne[0:Mn], uvec_ne[1:Mn + 1], Tvec_ne[1:Mn + 1], eta_ne[0:Mp], cs_ne1, gamma_n_vec,
          neq.cmax * np.ones([Mn, 1])]
djn = jax.vmap(jax.grad(neq.ionic_flux_fast, argnums=range(0, 4)))(*arg_jn)
djn_idx = find_indices(djn)

arg_etan = [eta_ne[0:Mn], phis_pe[1:Mn + 1], phie_ne[1:Mn + 1], Tvec_ne[1:Mn + 1], jvec_ne[0:Mn], cs_ne1, gamma_n_vec,
            neq.cmax * np.ones([Mn, 1])]
detan = jax.vmap(jax.grad(neq.over_poten_fast, argnums=range(0, 5)))(*arg_etan)
detan_idx = find_indices(detan)

# Phis
arg_phisn = [phis_ne[0:Mn], phis_ne[1:Mn + 1], phis_ne[2:Mn + 2], jvec_ne[0:Mn]]
dphisn = jax.vmap(jax.grad(neq.solid_poten, argnums=range(0, len(arg_phisn))))(*arg_phisn)
dphisn_idx = find_indices(dphisn)
dphisn_bc0 = jax.grad(neq.bc_phis, argnums=(0, 1))(phis_ne[0], phis_ne[1], 0)
dphisn_bcM = jax.grad(neq.bc_phis, argnums=(0, 1))(phis_ne[Mn], phis_ne[Mn + 1], Iapp)

dphisn_bc0_idx = find_bc_idx(dphisn_bc0)
dphisn_bcM_idx = find_bc_idx(dphisn_bcM)

dphisn_bc0_pairs = find_partial(dphisn_bc0_idx, [n0 + 1, n0 + 4 + 3])
dphisn_bcM_pairs = find_partial(dphisn_bcM_idx, [n0 + 4 + 6 * (Mn - 1) + 3, n0 + 4 + 6 * Mn + 1])

arg_phien = [uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2], phie_ne[0:Mn], phie_ne[1:Mn + 1], phie_ne[2:Mn + 2],
             Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2], jvec_ne[0:Mn]]
dphien = jax.vmap(jax.grad(neq.electrolyte_poten, argnums=range(0, len(arg_phien))))(*arg_phien)
dphien_idx = find_indices(dphien)

dphien_bc0 = jax.grad(neq.bc_phie_n, argnums=range(0, 12))(phie_ne[0], phie_ne[1], phie_sep[Ms], phie_sep[Ms + 1], \
                                                           uvec_ne[0], uvec_ne[1], uvec_sep[Ms], uvec_sep[Ms + 1], \
                                                           Tvec_ne[0], Tvec_ne[1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
dphien_bcM = jax.grad(neq.bc_zero_dirichlet, argnums=(0, 1))(phie_ne[Mn], phie_ne[Mn + 1])

dphien_bc0_idx = find_bc_idx(dphien_bc0)
dphien_bcM_idx = find_bc_idx(dphien_bcM)
dphien_bcM_pairs = find_partial(dphien_bcM_idx, [n0 + 4 + 6 * (Mn - 1) + 4, n0 + 4 + 6 * Mn + 3])

# temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
arg_Tn = arg_Tn = [uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2],
                   phie_ne[0:Mn], phie_ne[2:Mn + 2], \
                   phis_ne[0:Mn], phis_ne[2:Mn + 2],
                   Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2],
                   jvec_ne[0:Mn], \
                   eta_ne[0:Mn],
                   cs_ne1, gamma_n_vec, neq.cmax * np.ones([Mn, 1]),
                   Tvec_old_ne[1:Mn + 1]]

dTn = jax.vmap(jax.grad(neq.temperature_fast, argnums=range(0, 12)))(*arg_Tn)
dTn_idx = find_indices(dTn)

dTn_bc0 = jax.grad(neq.bc_temp_sn, argnums=range(0, 4))(Tvec_sep[Ms], Tvec_sep[Ms + 1], Tvec_ne[0], Tvec_ne[1])
dTn_bcM = jax.grad(neq.bc_temp_n, argnums=range(0, 4))(Tvec_ne[Mn], Tvec_ne[Mn + 1], Tvec_zcc[0], Tvec_zcc[1])
dTn_bc0_idx = find_bc_idx(dTn_bc0)
dTn_bcM_idx = find_bc_idx(dTn_bcM)

# current collector

dTa_bc0 = jax.grad(accq.bc_temp_a, argnums=(0, 1))(Tvec_acc[0], Tvec_acc[1])
dTa = jax.vmap(jax.grad(accq.temperature, argnums=range(0, 3)))(Tvec_acc[0:Ma], Tvec_acc[1:Ma + 1], Tvec_acc[2:Ma + 2],
                                                                Tvec_old_acc[1:Ma + 1])
dTa_bcM = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(Tvec_acc[Ma], Tvec_acc[Ma + 1], Tvec_pe[0], Tvec_pe[1])

dTz_bc0 = jax.grad(neq.bc_inter_cont, argnums=range(0, 4))(Tvec_ne[Mn], Tvec_ne[Mn + 1], Tvec_zcc[0], Tvec_zcc[1])
dTz = jax.vmap(jax.grad(zccq.temperature, argnums=range(0, 3)))(Tvec_zcc[0:Mz], Tvec_zcc[1:Mz + 1], Tvec_zcc[2:Mz + 2],
                                                                Tvec_old_zcc[1:Mz + 1])
dTz_bcM = jax.grad(zccq.bc_temp_z, argnums=(0, 1))(Tvec_zcc[Mz], Tvec_zcc[Mz + 1])

dTa_bc0_idx = find_bc_idx(dTa_bc0)
dTa_bcM_idx = find_bc_idx(dTa_bcM)
dTa_idx = find_indices(dTa)

dTz_bc0_idx = find_bc_idx(dTz_bc0)
dTz_idx = find_indices(dTz)
dTz_bcM_idx = find_bc_idx(dTz_bcM)


d =partials(accq, peq, sepq, neq, zccq)

from functools import partial


@partial(jax.jit, static_argnums=(4,5,6))
def compute_der_new(U, Uold, gamma_p_vec, gamma_n_vec, d, grid_num, vars):
    peq, neq, Iapp = vars
    Mp, Np, Mn, Nn, Ms, Ma, Mz = grid_num
    Jnew = np.zeros([23, len(U)])

    uvec_pe, uvec_sep, uvec_ne, \
    Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
    phie_pe, phie_sep, phie_ne, \
    phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    uvec_old_pe, uvec_old_sep, uvec_old_ne, \
    Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc, \
    phie_old_pe, phie_old_sep, phie_old_ne, \
    phis_old_pe, phis_old_ne, jvec_old_pe, jvec_old_ne, eta_old_pe, eta_old_ne = unpack_fast(Uold, Mp, Np, Mn, Nn, Ms,
                                                                                             Ma, Mz)

    arg_up = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2], Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2],
              jvec_pe[0:Mp], uvec_old_pe[1:Mp + 1]]

    Jnew = build_dup(Jnew, d['dup'](*arg_up), Ma, Mp)

    arg_jp = [jvec_pe[0:Mp], uvec_pe[1:Mp + 1], Tvec_pe[1:Mp + 1], eta_pe[0:Mp], cs_pe1, gamma_p_vec,
              peq.cmax * np.ones([Mp, 1])]
    Jnew = build_djp(Jnew, d['djp'](*arg_jp), Ma, Mp)

    arg_etap = [eta_pe[0:Mp], phis_pe[1:Mp + 1], phie_pe[1:Mp + 1], Tvec_pe[1:Mp + 1], jvec_pe[0:Mp], cs_pe1,
                gamma_p_vec, peq.cmax * np.ones([Mp, 1])]

    Jnew = build_detap(Jnew, d['detap'](*arg_etap), Ma, Mp)
    arg_phisp = [phis_pe[0:Mp], phis_pe[1:Mp + 1], phis_pe[2:Mp + 2], jvec_pe[0:Mp]]

    Jnew = build_dphisp(Jnew, d['dphisp'](*arg_phisp), Ma, Mp)

    arg_phiep = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2], phie_pe[0:Mp], phie_pe[1:Mp + 1],
                 phie_pe[2:Mp + 2],
                 Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2], jvec_pe[0:Mp]]

    Jnew = build_dphiep(Jnew, d['dphiep'](*arg_phiep), Ma, Mp)

    # temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
    arg_Tp = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2],
              phie_pe[0:Mp], phie_pe[2:Mp + 2],
              phis_pe[0:Mp], phis_pe[2:Mp + 2],
              Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2],
              jvec_pe[0:Mp],
              eta_pe[0:Mp],
              cs_pe1, gamma_p_vec, peq.cmax * np.ones([Mp, 1]),
              Tvec_old_pe[1:Mp + 1]]

    Jnew = build_dTp(Jnew, d['dTp'](*arg_Tp), Ma, Mp)

    dup_bc0 = d['dup_bc0'](uvec_pe[0], uvec_pe[1])
    dup_bcM = d['dup_bcM'](uvec_pe[Mp], uvec_pe[Mp + 1], Tvec_pe[Mp], Tvec_pe[Mp + 1],
                           uvec_sep[0], uvec_sep[1], Tvec_sep[0], Tvec_sep[1])

    dphisp_bc0 = d['dphisp_bc0'](phis_pe[0], phis_pe[1], Iapp)
    dphisp_bcM = d['dphisp_bcM'](phis_pe[Mp], phis_pe[Mp + 1], 0)

    dphiep_bc0 = d['dphiep_bc0'](phie_pe[0], phie_pe[1])
    dphiep_bcM = d['dphiep_bcM'](phie_pe[Mp], phie_pe[Mp + 1], phie_sep[0], phie_sep[1],
                                 uvec_pe[Mp], uvec_pe[Mp + 1], uvec_sep[0], uvec_sep[1],
                                 Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])

    dTp_bc0 = d['dTp_bc0'](Tvec_acc[Ma], Tvec_acc[Ma + 1], Tvec_pe[0], Tvec_pe[1])
    dTp_bcM = d['dTp_bcM'](Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])

    bc_p = dict([('u', np.concatenate((np.array(dup_bc0), np.array(dup_bcM)))),
                 ('phis', np.concatenate((np.array(dphisp_bc0), np.array(dphisp_bcM)))),
                 ('phie', np.concatenate((np.array(dphiep_bc0), np.array(dphiep_bcM)))),
                 ('T', np.concatenate((np.array(dTp_bc0), np.array(dTp_bcM))))])

    Jnew = build_bc_p(Jnew, bc_p, Ma, Mp)

    bc_s = d['bc_s'](uvec_sep[0], uvec_sep[1], uvec_pe[Mp], uvec_pe[Mp + 1])
    Jnew = build_bc_s(Jnew, np.concatenate((np.array(bc_s), np.array(bc_s))), Ma, Mp, Ms)
    dus = d['dus'](uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                   uvec_sep[2:Ms + 2], Tvec_sep[0:Ms],
                   Tvec_sep[1:Ms + 1], Tvec_sep[2:Ms + 2],
                   uvec_old_sep[1:Ms + 1])

    Jnew = build_dus(Jnew, dus, Ma, Mp, Ms)
    dphies = d['dphies'](uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                         uvec_sep[2:Ms + 2],
                         phie_sep[0:Ms], phie_sep[1:Ms + 1],
                         phie_sep[2:Ms + 2],
                         Tvec_sep[0:Ms], Tvec_sep[1:Ms + 1],
                         Tvec_sep[2:Ms + 2])
    Jnew = build_dphies(Jnew, dphies, Ma, Mp, Ms)

    dTs = d['dTs'](uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                   uvec_sep[2:Ms + 2],
                   phie_sep[0:Ms], phie_sep[2:Ms + 2],
                   Tvec_sep[0:Ms], Tvec_sep[1:Ms + 1],
                   Tvec_sep[2:Ms + 2],
                   Tvec_old_sep[1:Ms + 1])

    Jnew = build_dTs(Jnew, dTs, Ma, Mp, Ms)

    dun_bc0 = d['dun_bc0'](uvec_ne[0], uvec_ne[1], Tvec_ne[0], Tvec_ne[1],
                           uvec_sep[Ms], uvec_sep[Ms + 1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
    dun = d['dun'](uvec_ne[0:Mn], uvec_ne[1:Mn + 1],
                   uvec_ne[2:Mn + 2],
                   Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1],
                   Tvec_ne[2:Mn + 2],
                   jvec_ne[0:Mn], uvec_old_ne[1:Mn + 1])

    dun_bcM = d['dun_bcM'](uvec_ne[Mn], uvec_ne[Mn + 1])
    Jnew = build_dun(Jnew, dun, Ma, Mp, Ms, Mn)

    arg_jn = [jvec_ne[0:Mn], uvec_ne[1:Mn + 1], Tvec_ne[1:Mn + 1], eta_ne[0:Mp], cs_ne1, gamma_n_vec,
              neq.cmax * np.ones([Mn, 1])]

    Jnew = build_djn(Jnew, d['djn'](*arg_jn), Ma, Mp, Ms, Mn)

    arg_etan = [eta_ne[0:Mn], phis_pe[1:Mn + 1], phie_ne[1:Mn + 1], Tvec_ne[1:Mn + 1], jvec_ne[0:Mn], cs_ne1,
                gamma_n_vec,
                neq.cmax * np.ones([Mn, 1])]

    Jnew = build_detan(Jnew, d['detan'](*arg_etan), Ma, Mp, Ms, Mn)

    arg_phisn = [phis_ne[0:Mn], phis_ne[1:Mn + 1], phis_ne[2:Mn + 2], jvec_ne[0:Mn]]

    dphisn_bc0 = d['dphisn_bc0'](phis_ne[0], phis_ne[1], 0)
    dphisn_bcM = d['dphisn_bcM'](phis_ne[Mn], phis_ne[Mn + 1], Iapp)

    Jnew = build_dphisn(Jnew, d['dphisn'](*arg_phisn), Ma, Mp, Ms, Mn)
    arg_phien = [uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2], phie_ne[0:Mn], phie_ne[1:Mn + 1],
                 phie_ne[2:Mn + 2],
                 Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2], jvec_ne[0:Mn]]
    dphien = jax.vmap(jax.grad(neq.electrolyte_poten, argnums=range(0, len(arg_phien))))(*arg_phien)

    Jnew = build_dphien(Jnew, d['dphien'](*arg_phien), Ma, Mp, Ms, Mn)
    dphien_bc0 = d['dphien_bc0'](phie_ne[0], phie_ne[1], phie_sep[Ms], phie_sep[Ms + 1], \
                                                               uvec_ne[0], uvec_ne[1], uvec_sep[Ms], uvec_sep[Ms + 1], \
                                                               Tvec_ne[0], Tvec_ne[1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
    dphien_bcM = d['dphien_bcM'](phie_ne[Mn], phie_ne[Mn + 1])

    arg_Tn = [uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2],
              phie_ne[0:Mn], phie_ne[2:Mn + 2], \
              phis_ne[0:Mn], phis_ne[2:Mn + 2],
              Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2],
              jvec_ne[0:Mn], \
              eta_ne[0:Mn],
              cs_ne1, gamma_n_vec, neq.cmax * np.ones([Mn, 1]),
              Tvec_old_ne[1:Mn + 1]]

    Jnew = build_dTn(Jnew, d['dTn'](*arg_Tn), Ma, Mp, Ms, Mn)
    dTn_bc0 = d['dTn_bc0'](Tvec_sep[Ms], Tvec_sep[Ms + 1], Tvec_ne[0], Tvec_ne[1])
    dTn_bcM = d['dTn_bcM'](Tvec_ne[Mn], Tvec_ne[Mn + 1], Tvec_zcc[0], Tvec_zcc[1])
    bc_n = dict([('u', np.concatenate((np.array(dun_bc0), np.array(dun_bcM)))),
                 ('phis', np.concatenate((np.array(dphisn_bc0), np.array(dphisn_bcM)))),
                 ('phie', np.concatenate((np.array(dphien_bc0), np.array(dphien_bcM)))),
                 ('T', np.concatenate((np.array(dTn_bc0), np.array(dTn_bcM))))
                 ])
    Jnew = build_bc_n(Jnew, bc_n, Ma, Mp, Ms, Mn)

    dTa_bc0 = d['dTa_bc0'](Tvec_acc[0], Tvec_acc[1])
    dTa = d['dTa'](Tvec_acc[0:Ma], Tvec_acc[1:Ma + 1],
                                                                    Tvec_acc[2:Ma + 2],
                                                                    Tvec_old_acc[1:Ma + 1])
    dTa_bcM = d['dTa_bcM'](Tvec_acc[Ma], Tvec_acc[Ma + 1], Tvec_pe[0], Tvec_pe[1])

    dTz_bc0 = d['dTz_bc0'](Tvec_ne[Mn], Tvec_ne[Mn + 1], Tvec_zcc[0], Tvec_zcc[1])
    dTz = d['dTz'](Tvec_zcc[0:Mz], Tvec_zcc[1:Mz + 1],
                                                                    Tvec_zcc[2:Mz + 2],
                                                                    Tvec_old_zcc[1:Mz + 1])
    dTz_bcM = d['dTz_bcM'](Tvec_zcc[Mz], Tvec_zcc[Mz + 1])

    Jnew = build_dTa(Jnew, dTa, Ma)
    Jnew = build_dTz(Jnew, dTz, Ma, Mp, Ms, Mn, Mz)
    bc_acc = dict([
        ('acc', np.concatenate((np.array(dTa_bc0), np.array(dTa_bcM)))),
        ('zcc', np.concatenate((np.array(dTz_bc0), np.array(dTz_bcM))))
    ])
    Jnew = build_bc_cc(Jnew, bc_acc, Ma, Mp, Ms, Mn, Mz)

    return Jnew


@jax.jit
def compute_der(U, Uold, cs_pe1, cs_ne1):
    Jnew = np.zeros([23, len(U_fast)])

    uvec_pe, uvec_sep, uvec_ne, \
    Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
    phie_pe, phie_sep, phie_ne, \
    phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack_fast(U, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    uvec_old_pe, uvec_old_sep, uvec_old_ne, \
    Tvec_old_acc, Tvec_old_pe, Tvec_old_sep, Tvec_old_ne, Tvec_old_zcc, \
    phie_old_pe, phie_old_sep, phie_old_ne, \
    phis_old_pe, phis_old_ne, jvec_old_pe, jvec_old_ne, eta_old_pe, eta_old_ne = unpack_fast(Uold, Mp, Np, Mn, Nn, Ms,
                                                                                             Ma, Mz)

    arg_up = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2], Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2],
              jvec_pe[0:Mp], uvec_old_pe[1:Mp + 1]]
    dup = jax.vmap(jax.grad(peq.electrolyte_conc, argnums=range(0, 8)))(*arg_up)

    Jnew = build_dup(Jnew, dup, Ma, Mp)

    arg_jp = [jvec_pe[0:Mp], uvec_pe[1:Mp + 1], Tvec_pe[1:Mp + 1], eta_pe[0:Mp], cs_pe1, gamma_p_vec,
              peq.cmax * np.ones([Mp, 1])]
    djp = jax.vmap(jax.grad(peq.ionic_flux_fast, argnums=range(0, 4)))(*arg_jp)

    Jnew = build_djp(Jnew, djp, Ma, Mp)

    arg_etap = [eta_pe[0:Mp], phis_pe[1:Mp + 1], phie_pe[1:Mp + 1], Tvec_pe[1:Mp + 1], jvec_pe[0:Mp], cs_pe1,
                gamma_p_vec, peq.cmax * np.ones([Mp, 1])]
    detap = jax.vmap(jax.grad(peq.over_poten_fast, argnums=range(0, 5)))(*arg_etap)

    Jnew = build_detap(Jnew, detap, Ma, Mp)
    arg_phisp = [phis_pe[0:Mp], phis_pe[1:Mp + 1], phis_pe[2:Mp + 2], jvec_pe[0:Mp]]
    dphisp = jax.vmap(jax.grad(peq.solid_poten, argnums=range(0, len(arg_phisp))))(*arg_phisp)

    Jnew = build_dphisp(Jnew, dphisp, Ma, Mp)

    arg_phiep = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2], phie_pe[0:Mp], phie_pe[1:Mp + 1],
                 phie_pe[2:Mp + 2],
                 Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2], jvec_pe[0:Mp]]
    dphiep = jax.vmap(jax.grad(peq.electrolyte_poten, argnums=range(0, len(arg_phiep))))(*arg_phiep)
    Jnew = build_dphiep(Jnew, dphiep, Ma, Mp)

    # temperature_fast(self,un, uc, up, phien, phiep, phisn, phisp, Tn, Tc, Tp,j,eta, cs_1, gamma_c, cmax, Told):
    arg_Tp = [uvec_pe[0:Mp], uvec_pe[1:Mp + 1], uvec_pe[2:Mp + 2],
              phie_pe[0:Mp], phie_pe[2:Mp + 2],
              phis_pe[0:Mp], phis_pe[2:Mp + 2],
              Tvec_pe[0:Mp], Tvec_pe[1:Mp + 1], Tvec_pe[2:Mp + 2],
              jvec_pe[0:Mp],
              eta_pe[0:Mp],
              cs_pe1, gamma_p_vec, peq.cmax * np.ones([Mp, 1]),
              Tvec_old_pe[1:Mp + 1]]

    dTp = jax.vmap(jax.grad(peq.temperature_fast, argnums=range(0, 12)))(*arg_Tp)
    Jnew = build_dTp(Jnew, dTp, Ma, Mp)

    dup_bc0 = jax.grad(peq.bc_zero_neumann, argnums=(0, 1))(uvec_pe[0], uvec_pe[1])
    dup_bcM = jax.grad(peq.bc_u_sep_p, argnums=range(0, 8))(uvec_pe[Mp], uvec_pe[Mp + 1], Tvec_pe[Mp], Tvec_pe[Mp + 1],
                                                            uvec_sep[0], uvec_sep[1], Tvec_sep[0], Tvec_sep[1])

    dphisp_bc0 = jax.grad(peq.bc_phis, argnums=(0, 1))(phis_pe[0], phis_pe[1], Iapp)
    dphisp_bcM = jax.grad(peq.bc_phis, argnums=(0, 1))(phis_pe[Mp], phis_pe[Mp + 1], 0)

    dphiep_bc0 = jax.grad(peq.bc_zero_neumann, argnums=(0, 1))(phie_pe[0], phie_pe[1])
    dphiep_bcM = jax.grad(peq.bc_phie_p, argnums=range(0, 12))(phie_pe[Mp], phie_pe[Mp + 1], phie_sep[0], phie_sep[1],
                                                               uvec_pe[Mp], uvec_pe[Mp + 1], uvec_sep[0], uvec_sep[1],
                                                               Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])

    dTp_bc0 = jax.grad(peq.bc_temp_ap, argnums=range(0, 4))(Tvec_acc[Ma], Tvec_acc[Ma + 1], Tvec_pe[0], Tvec_pe[1])
    dTp_bcM = jax.grad(peq.bc_temp_ps, argnums=range(0, 4))(Tvec_pe[Mp], Tvec_pe[Mp + 1], Tvec_sep[0], Tvec_sep[1])

    bc_p = dict([('u', np.concatenate((np.array(dup_bc0), np.array(dup_bcM)))),
                 ('phis', np.concatenate((np.array(dphisp_bc0), np.array(dphisp_bcM)))),
                 ('phie', np.concatenate((np.array(dphiep_bc0), np.array(dphiep_bcM)))),
                 ('T', np.concatenate((np.array(dTp_bc0), np.array(dTp_bcM))))])

    Jnew = build_bc_p(Jnew, bc_p, Ma, Mp)

    bc_s = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(uvec_sep[0], uvec_sep[1], uvec_pe[Mp], uvec_pe[Mp + 1])
    Jnew = build_bc_s(Jnew, np.concatenate((np.array(bc_s), np.array(bc_s))), Ma, Mp, Ms)
    dus = jax.vmap(jax.grad(sepq.electrolyte_conc, argnums=range(0, 6)))(uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                                                                         uvec_sep[2:Ms + 2], Tvec_sep[0:Ms],
                                                                         Tvec_sep[1:Ms + 1], Tvec_sep[2:Ms + 2],
                                                                         uvec_old_sep[1:Ms + 1])

    Jnew = build_dus(Jnew, dus, Ma, Mp, Ms)
    dphies = jax.vmap(jax.grad(sepq.electrolyte_poten, argnums=range(0, 9)))(uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                                                                             uvec_sep[2:Ms + 2],
                                                                             phie_sep[0:Ms], phie_sep[1:Ms + 1],
                                                                             phie_sep[2:Ms + 2],
                                                                             Tvec_sep[0:Ms], Tvec_sep[1:Ms + 1],
                                                                             Tvec_sep[2:Ms + 2])
    Jnew = build_dphies(Jnew, dphies, Ma, Mp, Ms)

    dTs = jax.vmap(jax.grad(sepq.temperature, argnums=range(0, 8)))(uvec_sep[0:Ms], uvec_sep[1:Ms + 1],
                                                                    uvec_sep[2:Ms + 2],
                                                                    phie_sep[0:Ms], phie_sep[2:Ms + 2],
                                                                    Tvec_sep[0:Ms], Tvec_sep[1:Ms + 1],
                                                                    Tvec_sep[2:Ms + 2],
                                                                    Tvec_old_sep[1:Ms + 1])

    Jnew = build_dTs(Jnew, dTs, Ma, Mp, Ms)

    dun_bc0 = jax.grad(neq.bc_u_sep_n, argnums=range(0, 8))(uvec_ne[0], uvec_ne[1], Tvec_ne[0], Tvec_ne[1],
                                                       uvec_sep[Ms], uvec_sep[Ms + 1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
    dun = jax.vmap(jax.grad(neq.electrolyte_conc, argnums=range(0, 8)))(uvec_ne[0:Mn], uvec_ne[1:Mn + 1],
                                                                        uvec_ne[2:Mn + 2],
                                                                        Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1],
                                                                        Tvec_ne[2:Mn + 2],
                                                                        jvec_ne[0:Mn], uvec_old_ne[1:Mn + 1])
    dun_bcM = jax.grad(neq.bc_zero_neumann, argnums=(0, 1))(uvec_ne[Mn], uvec_ne[Mn + 1])
    Jnew = build_dun(Jnew, dun, Ma, Mp, Ms, Mn)

    arg_jn = [jvec_ne[0:Mn], uvec_ne[1:Mn + 1], Tvec_ne[1:Mn + 1], eta_ne[0:Mp], cs_ne1, gamma_n_vec,
              neq.cmax * np.ones([Mn, 1])]
    djn = jax.vmap(jax.grad(neq.ionic_flux_fast, argnums=range(0, 4)))(*arg_jn)

    Jnew = build_djn(Jnew, djn, Ma, Mp, Ms, Mn)

    arg_etan = [eta_ne[0:Mn], phis_pe[1:Mn + 1], phie_ne[1:Mn + 1], Tvec_ne[1:Mn + 1], jvec_ne[0:Mn], cs_ne1,
                gamma_n_vec,
                neq.cmax * np.ones([Mn, 1])]
    detan = jax.vmap(jax.grad(neq.over_poten_fast, argnums=range(0, 5)))(*arg_etan)
    Jnew = build_detan(Jnew, detan, Ma, Mp, Ms, Mn)

    arg_phisn = [phis_ne[0:Mn], phis_ne[1:Mn + 1], phis_ne[2:Mn + 2], jvec_ne[0:Mn]]
    dphisn = jax.vmap(jax.grad(neq.solid_poten, argnums=range(0, len(arg_phisn))))(*arg_phisn)
    dphisn_bc0 = jax.grad(neq.bc_phis, argnums=(0, 1))(phis_ne[0], phis_ne[1], 0)
    dphisn_bcM = jax.grad(neq.bc_phis, argnums=(0, 1))(phis_ne[Mn], phis_ne[Mn + 1], Iapp)

    Jnew = build_dphisn(Jnew, dphisn, Ma, Mp, Ms, Mn)
    arg_phien = [uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2], phie_ne[0:Mn], phie_ne[1:Mn + 1],
                 phie_ne[2:Mn + 2],
                 Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2], jvec_ne[0:Mn]]
    dphien = jax.vmap(jax.grad(neq.electrolyte_poten, argnums=range(0, len(arg_phien))))(*arg_phien)

    Jnew = build_dphien(Jnew, dphien, Ma, Mp, Ms, Mn)
    dphien_bc0 = jax.grad(neq.bc_phie_n, argnums=range(0, 12))(phie_ne[0], phie_ne[1], phie_sep[Ms], phie_sep[Ms + 1], \
                                                               uvec_ne[0], uvec_ne[1], uvec_sep[Ms], uvec_sep[Ms + 1], \
                                                               Tvec_ne[0], Tvec_ne[1], Tvec_sep[Ms], Tvec_sep[Ms + 1])
    dphien_bcM = jax.grad(neq.bc_zero_dirichlet, argnums=(0, 1))(phie_ne[Mn], phie_ne[Mn + 1])

    arg_Tn = [uvec_ne[0:Mn], uvec_ne[1:Mn + 1], uvec_ne[2:Mn + 2],
              phie_ne[0:Mn], phie_ne[2:Mn + 2], \
              phis_ne[0:Mn], phis_ne[2:Mn + 2],
              Tvec_ne[0:Mn], Tvec_ne[1:Mn + 1], Tvec_ne[2:Mn + 2],
              jvec_ne[0:Mn], \
              eta_ne[0:Mn],
              cs_ne1, gamma_n_vec, neq.cmax * np.ones([Mn, 1]),
              Tvec_old_ne[1:Mn + 1]]

    dTn = jax.vmap(jax.grad(neq.temperature_fast, argnums=range(0, 12)))(*arg_Tn)

    Jnew = build_dTn(Jnew, dTn, Ma, Mp, Ms, Mn)
    dTn_bc0 = jax.grad(neq.bc_temp_sn, argnums=range(0, 4))(Tvec_sep[Ms], Tvec_sep[Ms + 1], Tvec_ne[0], Tvec_ne[1])
    dTn_bcM = jax.grad(neq.bc_temp_n, argnums=range(0, 4))(Tvec_ne[Mn], Tvec_ne[Mn + 1], Tvec_zcc[0], Tvec_zcc[1])
    bc_n = dict([('u', np.concatenate((np.array(dun_bc0), np.array(dun_bcM)))),
                 ('phis', np.concatenate((np.array(dphisn_bc0), np.array(dphisn_bcM)))),
                 ('phie', np.concatenate((np.array(dphien_bc0), np.array(dphien_bcM)))),
                 ('T', np.concatenate((np.array(dTn_bc0), np.array(dTn_bcM))))
                 ])
    Jnew = build_bc_n(Jnew, bc_n, Ma, Mp, Ms, Mn)

    dTa_bc0 = jax.grad(accq.bc_temp_a, argnums=(0, 1))(Tvec_acc[0], Tvec_acc[1])
    dTa = jax.vmap(jax.grad(accq.temperature, argnums=range(0, 3)))(Tvec_acc[0:Ma], Tvec_acc[1:Ma + 1],
                                                                    Tvec_acc[2:Ma + 2],
                                                                    Tvec_old_acc[1:Ma + 1])
    dTa_bcM = jax.grad(peq.bc_inter_cont, argnums=range(0, 4))(Tvec_acc[Ma], Tvec_acc[Ma + 1], Tvec_pe[0], Tvec_pe[1])

    dTz_bc0 = jax.grad(neq.bc_inter_cont, argnums=range(0, 4))(Tvec_ne[Mn], Tvec_ne[Mn + 1], Tvec_zcc[0], Tvec_zcc[1])
    dTz = jax.vmap(jax.grad(zccq.temperature, argnums=range(0, 3)))(Tvec_zcc[0:Mz], Tvec_zcc[1:Mz + 1],
                                                                    Tvec_zcc[2:Mz + 2],
                                                                    Tvec_old_zcc[1:Mz + 1])
    dTz_bcM = jax.grad(zccq.bc_temp_z, argnums=(0, 1))(Tvec_zcc[Mz], Tvec_zcc[Mz + 1])

    Jnew = build_dTa(Jnew, dTa, Ma)
    Jnew = build_dTz(Jnew, dTz, Ma, Mp, Ms, Mn, Mz)
    bc_acc = dict([
        ('acc', np.concatenate((np.array(dTa_bc0), np.array(dTa_bcM)))),
        ('zcc', np.concatenate((np.array(dTz_bc0), np.array(dTz_bcM))))
    ])
    Jnew = build_bc_cc(Jnew, bc_acc, Ma, Mp, Ms, Mn, Mz)

    return Jnew


Joutput = compute_der(U_fast_new, U_fast, cs_pe1, cs_ne1).block_until_ready()

diff0 = abs(Joutput - Jab)
diff0_matrix = np.zeros_like(Joutput)
diff0_matrix[Jab.nonzero()] = diff0[Jab.nonzero()]/abs(Jab[Jab.nonzero()])
plt.figure()
plt.imshow(diff0_matrix);
plt.colorbar()
plt.show()

plt.figure()
diff = abs(Joutput[:, n0:] - Jab[:,n0:])
plt.imshow(diff);
plt.colorbar()
plt.show()
np.where(diff > 1e-9)

plt.figure()
diff2 = abs(Joutput[:, 5 + Ma + 1:40] - Jab[:, 5 + Ma + 1:40])
plt.imshow(diff2);
plt.colorbar()
plt.show()

Ntot_pe =  4 * (Mp + 2) + 2 * (Mp)
Ntot_ne = 4 * (Mp + 2) + 2 * (Mp)

Ntot_sep = 3 * (Ms + 2)
Ntot_acc = Ms + 2
Ntot_zcc = Ms + 2
Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
