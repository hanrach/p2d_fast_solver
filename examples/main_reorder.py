from jax.config import config

config.update('jax_enable_x64', True)
from utils.unpack import unpack_fast
from decoupled.init import p2d_init_fast
from utils.derivatives import partials, compute_jac
from model.p2d_param import get_battery_sections
import jax.numpy as np
from utils.precompute_c import precompute
from scipy.sparse import csc_matrix
from scikits.umfpack import splu
from reordered.p2d_reorder_fn import p2d_reorder_fn

Np = 5
Nn = 5
Mp = 5
Ms = 5
Mn = 5
Ma = 5
Mz = 5
delta_t = 10
Iapp = -30
peq, neq, sepq, accq, zccq = get_battery_sections(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, Iapp)
fn, _ = p2d_init_fast(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, Iapp)

# Precompute c
Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq, neq)
gamma_p_vec = gamma_p * np.ones(Mp)
gamma_n_vec = gamma_n * np.ones(Mn)
lu_p = splu(csc_matrix(Ap))
lu_n = splu(csc_matrix(An))


partial_fns = partials(accq, peq, sepq, neq, zccq)
jac_fn = compute_jac(gamma_p_vec, gamma_n_vec, partial_fns, (Np, Nn, Mp, Mn,Ms, Ma, Mz), (peq, neq, Iapp))
U_fast, cmat_pe, cmat_ne, voltages, temps, time = p2d_reorder_fn(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t,
                                                                 lu_p, lu_n, temp_p, temp_n,
                                                                 gamma_p_vec, gamma_n_vec,
                                                                 fn, jac_fn, Iapp, 3520)

uvec_pe, Tvec_pe, phie_pe, phis_pe, \
uvec_ne, Tvec_ne, phie_ne, phis_ne, \
uvec_sep, Tvec_sep, phie_sep, Tvec_acc, Tvec_zcc, \
j_pe, eta_pe, j_ne, eta_ne = unpack_fast(U_fast, Mp, Np, Mn, Nn, Ms, Ma, Mz)
