from jax.config import config

config.update('jax_enable_x64', True)
# from utils.derivatives import partials, compute_jac
from __init__ import partials, compute_jac
from model.p2d_param import get_battery_sections
import jax.numpy as np
from utils.precompute_c import precompute
from scipy.sparse import csc_matrix
from scikits.umfpack import splu
from reordered.p2d_reorder_fn import p2d_reorder_fn
import pickle
from decoupled.init import p2d_init_fast
from decoupled.p2d_main_fast_fn import p2d_fast_fn_short


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
repnum = 5


linsolve_fast_list = onp.zeros([repnum, len(Np)])
linsolve_reorder_list = onp.zeros([repnum, len(Np)])


eval_fast_list = onp.zeros([repnum, len(Np)])
eval_reorder_list = onp.zeros([repnum, len(Np)])


overhead_fast_list = onp.zeros([repnum, len(Np)])
overhead_reorder_list = onp.zeros([repnum, len(Np)])


tot_fast_list = onp.zeros([repnum, len(Np)])
tot_reorder_list = onp.zeros([repnum, len(Np)])


init_fast_list = onp.zeros([repnum, len(Np)])
init_reorder_list = onp.zeros([repnum, len(Np)])


for j in range(0, repnum):
    for i in range(0, len(Mp)):
        peq, neq, sepq, accq, zccq = get_battery_sections(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, Iapp)

        fn_fast, jac_fn_fast = p2d_init_fast(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], delta_t, Iapp)
        U_fast, _, _, _, _, time_decoupled = p2d_fast_fn_short(Np[i], Nn[i], Mp[i], Mn[i], Ms[i], Ma[i], Mz[i], fn_fast, jac_fn_fast, Iapp, 3520)

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
                                                                         fn, jac_fn, Iapp,3520)


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


linsolve_fast_avg = np.mean(linsolve_fast_list, 0)
linsolve_reorder_avg = np.mean(linsolve_reorder_list, 0)


tot_fast_avg = np.mean(tot_fast_list, 0)
tot_reorder_avg = np.mean(tot_reorder_list, 0)


eval_fast_avg = np.mean(eval_fast_list, 0)
eval_reorder_avg = np.mean(eval_reorder_list, 0)


init_fast_avg = np.mean(init_fast_list, 0)
init_reorder_avg = np.mean(init_reorder_list, 0)


overhead_fast_avg = np.mean(overhead_fast_list, 0)
overhead_reorder_avg = np.mean(overhead_reorder_list, 0)

pickle.dump( [linsolve_fast_avg, linsolve_reorder_avg,
              tot_fast_avg, tot_reorder_avg,
              eval_fast_avg, eval_reorder_avg,
              init_fast_avg, init_reorder_avg,
              overhead_fast_avg, overhead_reorder_avg], open( "decoupled_reordered.p", "wb" ) )

import matplotlib.pyplot as plt
def save_plot_results():
    Ntot_list=[]
    for i in range(0, len(Mp)):
        Ntot_pe = (Mp[i] + 2) * (Mp[i]) + 4 * (Mp[i] + 2) + 2 * (Mp[i])
        Ntot_ne = (Mp[i] + 2) * (Mp[i]) + 4 * (Mp[i] + 2) + 2 * (Mp[i])

        Ntot_sep = 3 * (Ms[i] + 2)
        Ntot_acc = Ms[i] + 2
        Ntot_zcc = Ms[i] + 2
        Ntot = Ntot_pe + Ntot_ne + Ntot_sep + Ntot_acc + Ntot_zcc
        Ntot_list.append(Ntot)

    plt.figure()
    ax = plt.gca()
    plt.rcParams.update({'font.size': 16})
    plt.semilogy(Ntot_list,(tot_fast_avg),'r+-',label="decoupled");
    plt.semilogy(Ntot_list,(tot_reorder_avg),'b+-',label="reordered");
    ax.grid(which='major', axis='both')
    plt.xlabel("Degree of freedom")
    plt.ylabel("Total simulation time")
    plt.legend(loc='best');
    plt.savefig('decoupled_reordered_tot.pdf', bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    plt.rcParams.update({'font.size': 16})
    plt.semilogy(Ntot_list,(linsolve_fast_avg),'r+-',label="decoupled");
    plt.semilogy(Ntot_list,(linsolve_reorder_avg),'b+-',label="reordered");
    ax.grid(which='major', axis='both')
    plt.xlabel("Degree of freedom")
    plt.ylabel("Linear solve time")
    plt.legend(loc='best');
    plt.savefig('decoupled_reordered_linsolve.pdf', bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    plt.rcParams.update({'font.size': 16})
    plt.semilogy(Ntot_list,(eval_fast_avg),'r+-',label="decoupled");
    plt.semilogy(Ntot_list,(eval_reorder_avg),'b+-',label="reordered");
    ax.grid(which='major', axis='both')
    plt.xlabel("Degree of freedom")
    plt.ylabel("Function evaluation time")
    plt.legend(loc='best');
    plt.savefig('decoupled_reordered_eval.pdf', bbox_inches='tight')


save_plot_results()