from jax.config import config
config.update('jax_enable_x64', True)
from utils.unpack import unpack_fast, unpack
# import unpack_fast, unpack, partials, compute_jac
from decoupled.init import p2d_init_fast
from decoupled.p2d_main_fast_fn import p2d_fast_fn_short

from naive.init import p2d_init_slow
from naive.p2d_main_fn import p2d_fn_short

from utils.derivatives import partials, compute_jac
from model.p2d_param import get_battery_sections
import jax.numpy as np
from utils.precompute_c import precompute
from scipy.sparse import csc_matrix
from scikits.umfpack import splu
from reordered.p2d_reorder_fn import p2d_reorder_fn

Np = 10
Nn = 10
Mp = 10
Ms = 10
Mn = 10
Ma = 12
Mz = 12
delta_t = 10
Iapp = -30

def run_naive(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp):
    fn, jac_fn = p2d_init_slow(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, Iapp)
    U, voltages, temps, time = p2d_fn_short(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, fn, jac_fn, Iapp, 3000)

    cmat_pe, cmat_ne,uvec_pe, uvec_sep, uvec_ne, \
            Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
            phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack(U,Mp, Np, Mn, Nn, Ms, Ma, Mz)
    return cmat_pe, cmat_ne,uvec_pe, uvec_sep, uvec_ne, \
            Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
            phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne

def run_decoupled(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp):

    fn, jac_fn = p2d_init_fast(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, Iapp)
    U_fast, cmat_pe, cmat_ne, voltages, temps, time = p2d_fast_fn_short(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, fn, jac_fn,
                                                                        Iapp, 3000)

    uvec_pe, uvec_sep, uvec_ne, \
    Tvec_acc, Tvec_pe, Tvec_sep, \
    Tvec_ne, Tvec_zcc, phie_pe, phie_sep, phie_ne, \
    phis_pe, phis_ne, j_pe,j_ne,eta_pe,eta_ne = unpack_fast(U_fast, Mp, Np, Mn, Nn, Ms, Ma, Mz)

    return cmat_pe, cmat_ne, uvec_pe, uvec_sep, uvec_ne, \
           Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
           phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, j_pe, j_ne, eta_pe, eta_ne

def run_reordered(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp):
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
                                                                     fn, jac_fn, Iapp, 3000)

    uvec_pe, uvec_sep, uvec_ne, \
    Tvec_acc, Tvec_pe, Tvec_sep, \
    Tvec_ne, Tvec_zcc, phie_pe, phie_sep, phie_ne, \
    phis_pe, phis_ne, j_pe, j_ne, eta_pe, eta_ne = unpack_fast(U_fast, Mp, Np, Mn, Nn, Ms, Ma, Mz)
    return cmat_pe, cmat_ne, uvec_pe, uvec_sep, uvec_ne, \
           Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
           phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, j_pe, j_ne, eta_pe, eta_ne


cmat_pe_slow, cmat_ne_slow, uvec_pe_slow, uvec_sep_slow, uvec_ne_slow, \
    Tvec_acc_slow, Tvec_pe_slow, Tvec_sep_slow, Tvec_ne_slow, Tvec_zcc_slow, \
    phie_pe_slow, phie_sep_slow, phie_ne_slow, phis_pe_slow, phis_ne_slow, jvec_pe_slow, jvec_ne_slow, eta_pe_slow, eta_ne_slow = run_naive(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp)

cmat_pe_dec, cmat_ne_dec, uvec_pe_dec, uvec_sep_dec, uvec_ne_dec, \
Tvec_acc_dec, Tvec_pe_dec, Tvec_sep_dec, Tvec_ne_dec, Tvec_zcc_dec, \
phie_pe_dec, phie_sep_dec, phie_ne_dec, phis_pe_dec, phis_ne_dec, j_pe_dec, j_ne_dec, eta_pe_dec, eta_ne_dec = run_decoupled(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp)

cmat_pe_r, cmat_ne_r, uvec_pe_r, uvec_sep_r, uvec_ne_r, \
           Tvec_acc_r, Tvec_pe_r, Tvec_sep_r, Tvec_ne_r, Tvec_zcc_r, \
           phie_pe_r, phie_sep_r, phie_ne_r, phis_pe_r, phis_ne_r, j_pe_r, j_ne_r, eta_pe_r, eta_ne_r = run_reordered(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp)

def test_c():

    assert np.allclose(cmat_pe_slow, np.reshape(cmat_pe_dec, [Np+2, Mp], order="F"), rtol=1e-10)
    assert np.allclose(cmat_ne_slow, np.reshape(cmat_ne_dec, [Nn + 2, Mn], order="F"), rtol=1e-10)

def test_u():
    assert np.allclose(uvec_pe_slow, uvec_pe_dec, rtol=1e-9)
    assert np.allclose(uvec_ne_slow, uvec_ne_dec, rtol=1e-9)
    assert np.allclose(uvec_sep_slow, uvec_sep_dec, rtol=1e-9)

def test_j():
    assert np.allclose(jvec_pe_slow, j_pe_dec, rtol=1e-9)
    assert np.allclose(jvec_ne_slow, j_ne_dec, rtol=1e-9)

def test_eta():
    assert np.allclose(eta_pe_slow, eta_pe_dec, rtol=1e-9)
    assert np.allclose(eta_ne_slow, eta_ne_dec, rtol=1e-9)

def test_phie():
    assert np.allclose(phie_pe_slow, phie_pe_dec, rtol=1e-9)
    assert np.allclose(phie_ne_slow, phie_ne_dec, rtol=1e-9)
    assert np.allclose(phie_sep_slow, phie_sep_dec, rtol=1e-9)

def test_T():
    assert np.allclose(Tvec_acc_slow, Tvec_acc_dec, rtol=1e-9)
    assert np.allclose(Tvec_pe_slow, Tvec_pe_dec, rtol=1e-9)
    assert np.allclose(Tvec_ne_slow, Tvec_ne_dec, rtol=1e-9)
    assert np.allclose(Tvec_sep_slow, Tvec_sep_dec, rtol=1e-9)
    assert np.allclose(Tvec_zcc_slow, Tvec_zcc_dec, rtol=1e-9)

def test_phis():
    assert np.allclose(phis_pe_slow, phis_pe_dec, rtol=1e-9)
    assert np.allclose(phis_ne_slow, phis_ne_dec, rtol=1e-9)



def reorder_test_c():

    assert np.allclose(cmat_pe_r, np.reshape(cmat_pe_dec, [Np+2, Mp], order="F"), rtol=1e-10)
    assert np.allclose(cmat_ne_r, np.reshape(cmat_ne_dec, [Nn + 2, Mn], order="F"), rtol=1e-10)

def reorder_test_u():
    assert np.allclose(uvec_pe_r, uvec_pe_dec, rtol=1e-9)
    assert np.allclose(uvec_ne_r, uvec_ne_dec, rtol=1e-9)
    assert np.allclose(uvec_sep_r, uvec_sep_dec, rtol=1e-9)

def reorder_test_j():
    assert np.allclose(j_pe_r, j_pe_dec, rtol=1e-9)
    assert np.allclose(j_ne_r, j_ne_dec, rtol=1e-9)

def reorder_test_eta():
    assert np.allclose(eta_pe_r, eta_pe_dec, rtol=1e-9)
    assert np.allclose(eta_ne_r, eta_ne_dec, rtol=1e-9)

def reorder_test_phie():
    assert np.allclose(phie_pe_r, phie_pe_dec, rtol=1e-9)
    assert np.allclose(phie_ne_r, phie_ne_dec, rtol=1e-9)
    assert np.allclose(phie_sep_r, phie_sep_dec, rtol=1e-9)

def reorder_test_T():
    assert np.allclose(Tvec_acc_r, Tvec_acc_dec, rtol=1e-9)
    assert np.allclose(Tvec_pe_r, Tvec_pe_dec, rtol=1e-9)
    assert np.allclose(Tvec_ne_r, Tvec_ne_dec, rtol=1e-9)
    assert np.allclose(Tvec_sep_r, Tvec_sep_dec, rtol=1e-9)
    assert np.allclose(Tvec_zcc_r, Tvec_zcc_dec, rtol=1e-9)

def reorder_test_phis():
    assert np.allclose(phis_pe_r, phis_pe_dec, rtol=1e-9)
    assert np.allclose(phis_ne_r, phis_ne_dec, rtol=1e-9)

