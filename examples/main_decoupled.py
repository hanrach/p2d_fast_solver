from jax.config import config
config.update('jax_enable_x64', True)
from utils.unpack import unpack_fast
from decoupled.init import p2d_init_fast
from decoupled.p2d_main_fast_fn import p2d_fast_fn_short
Np = 10
Nn = 10
Mp = 10
Ms = 10
Mn = 10
Ma = 12
Mz = 12
delta_t = 10
Iapp = -30
fn, jac_fn = p2d_init_fast(Np, Nn, Mp, Mn, Ms, Ma,Mz,delta_t, Iapp)
U_fast, cmat_pe, cmat_ne, voltages, temps,time = p2d_fast_fn_short(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, fn, jac_fn, Iapp, 100)

uvec_pe, Tvec_pe, phie_pe, phis_pe, \
uvec_ne, Tvec_ne, phie_ne, phis_ne,\
uvec_sep, Tvec_sep, phie_sep, Tvec_acc, Tvec_zcc,\
 j_pe, eta_pe, j_ne, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)
 