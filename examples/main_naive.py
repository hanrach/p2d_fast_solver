
from jax.config import config

config.update('jax_enable_x64', True)
from utils.unpack import unpack
from naive.init import p2d_init_slow
from naive.p2d_main_fn import p2d_fn_short

Np = 10
Nn = 10
Mp = 10
Ms = 10
Mn = 10
Ma = 12
Mz = 12
delta_t = 10
Iapp = -30
fn, jac_fn = p2d_init_slow(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, Iapp)
U, voltages, temps, time = p2d_fn_short(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, fn, jac_fn, Iapp, 100)

cmat_pe, cmat_ne,uvec_pe, uvec_sep, uvec_ne, \
        Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, \
        phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, jvec_pe, jvec_ne, eta_pe, eta_ne = unpack(U,Mp, Np, Mn, Nn, Ms, Ma, Mz)
