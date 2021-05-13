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

from decoupled.init import p2d_init_fast
from decoupled.p2d_main_fast_fn import p2d_fast_fn_short

from utils.unpack import unpack
from naive.init import p2d_init_slow
from naive.p2d_main_fn import p2d_fn_short

import numpy as onp
Np = [ 10, 20, 30, 40, 50]
Nn = [ 10, 20, 30, 40, 50]
Mp = [ 10, 20, 30, 40, 50]
Ms = [5,5,5,5, 5]
Mn = [10, 20, 30, 40, 50]
Ma = [5,5,5,5,5]
Mz = [5,5,5,5,5]

delta_t = 10
Iapp = -30
repnum = 1

linsolve_slow_list = onp.zeros([repnum, len(Np)])
linsolve_fast_list = onp.zeros([repnum, len(Np)])
linsolve_reorder_list = onp.zeros([repnum, len(Np)])

eval_slow_list = onp.zeros([repnum, len(Np)])
eval_fast_list = onp.zeros([repnum, len(Np)])
eval_reorder_list = onp.zeros([repnum, len(Np)])

overhead_slow_list = onp.zeros([repnum, len(Np)])
overhead_fast_list = onp.zeros([repnum, len(Np)])
overhead_reorder_list = onp.zeros([repnum, len(Np)])

tot_slow_list = onp.zeros([repnum, len(Np)])
tot_fast_list = onp.zeros([repnum, len(Np)])
tot_reorder_list = onp.zeros([repnum, len(Np)])

init_slow_list = onp.zeros([repnum, len(Np)])
init_fast_list = onp.zeros([repnum, len(Np)])
init_reorder_list = onp.zeros([repnum, len(Np)])


for j in range(0, repnum):
    for i in range(0, len(Mp)):
        peq, neq, sepq, accq, zccq = get_battery_sections(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, Iapp)

        fn, jac_fn = p2d_init_slow(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, Iapp)
        U_naive, _, _, time_naive = p2d_fn_short(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, fn, jac_fn, Iapp, 3000)

        fn_fast, jac_fn_fast = p2d_init_fast(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, Iapp)
        U_fast, _, _, _, _, time_decoupled = p2d_fast_fn_short(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], fn_fast, jac_fn_fast, Iapp, 3000)

        fn, _ = p2d_init_fast(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, Iapp)

        # Precompute c
        Ap, An, gamma_n, gamma_p, temp_p, temp_n = precompute(peq, neq)
        gamma_p_vec = gamma_p * np.ones(Mp[i])
        gamma_n_vec = gamma_n * np.ones(Mn[i])
        lu_p = splu(csc_matrix(Ap))
        lu_n = splu(csc_matrix(An))

        partial_fns = partials(accq, peq, sepq, neq, zccq)
        jac_fn = compute_jac(gamma_p_vec, gamma_n_vec, partial_fns, (Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i]), (peq, neq, Iapp))
        U_reorder, _, _, _, _, time_reorder = p2d_reorder_fn(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t,
                                                                         lu_p, lu_n, temp_p, temp_n,
                                                                         gamma_p_vec, gamma_n_vec,
                                                                         fn, jac_fn, Iapp,3000)

        tot_slow_list[j, i] = time_naive[0]
        linsolve_slow_list[j, i] = time_naive[1]
        eval_slow_list[j, i] = time_naive[2]
        overhead_slow_list[j, i] = time_naive[3]
        init_slow_list[j,i] = time_naive[4]

        tot_fast_list[j, i] = time_decoupled[0]
        linsolve_fast_list[j, i] = time_decoupled[1]
        eval_fast_list[j, i] = time_decoupled[2]
        overhead_fast_list[j, i] = time_decoupled[3]
        init_fast_list[j, i] = time_decoupled[4]

        tot_reorder_list[j, i] = time_reorder[0]
        linsolve_reorder_list[j, i] = time_reorder[1]
        eval_reorder_list[j, i] = time_reorder[2]
        overhead_reorder_list[j, i] = time_reorder[3]
        init_reorder_list[j, i] = time_reorder[4]

linsolve_slow_avg = np.mean(linsolve_slow_list, 0)
linsolve_fast_avg = np.mean(linsolve_fast_list, 0)
linsolve_reorder_avg = np.mean(linsolve_reorder_list, 0)

tot_slow_avg = np.mean(tot_slow_list, 0)
tot_fast_avg = np.mean(tot_fast_list, 0)
tot_reorder_avg = np.mean(tot_reorder_list, 0)

eval_slow_avg = np.mean(eval_slow_list, 0)
eval_fast_avg = np.mean(eval_fast_list, 0)
eval_reorder_avg = np.mean(eval_reorder_list, 0)

init_slow_avg = np.mean(init_slow_list, 0)
init_fast_avg = np.mean(init_fast_list, 0)
init_reorder_avg = np.mean(init_reorder_list, 0)

overhead_slow_avg = np.mean(overhead_slow_list, 0)
overhead_fast_avg = np.mean(overhead_fast_list, 0)
overhead_reorder_avg = np.mean(overhead_reorder_list, 0)

