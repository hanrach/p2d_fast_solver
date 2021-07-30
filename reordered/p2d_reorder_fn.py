import timeit
import jax
import model.coeffs as coeffs
from model.p2d_param import get_battery_sections
from jax import vmap
import jax.numpy as np
import numpy as onp
from model.settings import Tref
from reordered.p2d_reorder_newton import newton, process_y, reorder_vec
from utils.reorder import reorder_tot
from utils.unpack import unpack_vars



def p2d_reorder_fn(Np, Nn, Mp, Mn, Ms, Ma, Mz, delta_t, lu_pe, lu_ne, temp_p, temp_n, gamma_p_vec, gamma_n_vec, fn_fast, jac_fn,  Iapp, Tf, tol=1e-7):
    start0 = timeit.default_timer()
    peq, neq, sepq, accq, zccq = get_battery_sections(Np, Nn, Mp, Ms, Mn, Ma, Mz, delta_t, Iapp)

    @jax.jit
    def cmat_format_p(cmat):
        val = jax.ops.index_update(cmat, jax.ops.index[0:Mp * (Np + 2):Np + 2], 0)
        val = jax.ops.index_update(val, jax.ops.index[Np + 1:Mp * (Np + 2):Np + 2], 0)
        #        for i in range(0,M):
        #            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
        #            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)
        return val

    @jax.jit
    def cmat_format_n(cmat):
        val = jax.ops.index_update(cmat, jax.ops.index[0:Mn * (Nn + 2):Nn + 2], 0)
        val = jax.ops.index_update(val, jax.ops.index[Nn + 1:Mn * (Nn + 2):Nn + 2], 0)
        #        for i in range(0,M):
        #            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)], 0)
        #            val = jax.ops.index_update(val, jax.ops.index[i*(N+2)+N+1],0)
        return val

    @jax.jit
    def form_c2_p_jit(temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(peq.Ds * np.ones(Mp), peq.ED * np.ones(Mp), T[1:Mp + 1])
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        #        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val = vmap(fn, (0, None, 0), 1)(j, temp, Deff_vec)

        return val

    @jax.jit
    def form_c2_n_jit(temp, j, T):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(neq.Ds * np.ones(Mn), neq.ED * np.ones(Mn), T[1:Mn + 1])
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        #        val=vmap(fn,(0,1,0),1)(j,temp,Deff_vec)
        val = vmap(fn, (0, None, 0), 1)(j, temp, Deff_vec)
        return val

    @jax.partial(jax.jit, static_argnums=(2, 3,))
    def combine_c(cII, cI_vec, M,N):
        return np.reshape(cII, [M * (N + 2)], order="F") + cI_vec

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

    cmat_pe = peq.cavg * np.ones(Mp * (Np + 2))
    cmat_ne = neq.cavg * np.ones(Mn * (Nn + 2))

    # lu = {"pe": lu_p, "ne": lu_n}

    idx_tot = reorder_tot(Mp, Mn, Ms, Ma, Mz)
    re_idx = np.argsort(idx_tot)


    steps = Tf / delta_t;
    voltages = []
    temps = []
    end0 = timeit.default_timer()

    print("setup time", end0 - start0)
    #    res_list=[]
    solve_time_tot = 0
    jf_tot_time = 0
    overhead_time = 0
    cmat_rhs_pe = cmat_format_p(cmat_pe)
    cmat_rhs_ne = cmat_format_n(cmat_ne)
    # lu_pe = lu["pe"];
    # lu_ne = lu["ne"]

    init_csolve=timeit.default_timer();
    cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
    cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))

    cs_pe1 = (cI_pe_vec[Np:Mp * (Np + 2):Np + 2] + cI_pe_vec[Np + 1:Mp * (Np + 2):Np + 2]) / 2
    cs_ne1 = (cI_ne_vec[Nn:Mn * (Nn + 2):Nn + 2] + cI_ne_vec[Nn + 1:Mn * (Nn + 2):Nn + 2]) / 2
    end_init_csolve=timeit.default_timer();
    print("initial csolve time:", end_init_csolve-init_csolve)

    start_init = timeit.default_timer()
    Jinit = jac_fn(U_fast, U_fast, cs_pe1, cs_ne1).block_until_ready()
    y = fn_fast(U_fast, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec).block_until_ready()
    y = process_y(y, idx_tot).block_until_ready()
    vec = reorder_vec(y, re_idx).block_until_ready()
    cmat_rhs_pe = cmat_format_p(cmat_pe).block_until_ready()
    cmat_rhs_ne = cmat_format_n(cmat_ne).block_until_ready()
    cII_p = form_c2_p_jit(temp_p, np.zeros(Mp), Tref + np.zeros(Mp + 2)).block_until_ready()
    cII_n = form_c2_n_jit(temp_n, np.zeros(Mn), Tref + np.zeros(Mn + 2),).block_until_ready()
    cmat_pe_init = np.reshape(cII_p, [Mp * (Np + 2)], order="F").block_until_ready() + cI_pe_vec
    cmat_ne_init = np.reshape(cII_n, [Mn * (Nn + 2)], order="F").block_until_ready() + cI_ne_vec
    U_fast_init, info_init = newton(fn_fast, jac_fn, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec, idx_tot, re_idx, tol)
    end_init = timeit.default_timer()

    init_time = end_init - start_init
    start1 = timeit.default_timer()


    extra=0
    tot_newton=0
    for i in range(0, int(steps)):
        prep = timeit.default_timer()
        cmat_rhs_pe = cmat_format_p(cmat_pe).block_until_ready()
        cmat_rhs_ne = cmat_format_n(cmat_ne).block_until_ready()
        prep_end=timeit.default_timer()
        # lu_pe = lu["pe"];
        # lu_ne = lu["ne"]

        start = timeit.default_timer()
        cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
        cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
        end = timeit.default_timer()
        c_lintime = end - start

        prep1=timeit.default_timer()
        cs_pe1 = (cI_pe_vec[Np:Mp * (Np + 2):Np + 2] + cI_pe_vec[Np + 1:Mp * (Np + 2):Np + 2]) / 2
        cs_ne1 = (cI_ne_vec[Nn:Mn * (Nn + 2):Nn + 2] + cI_ne_vec[Nn + 1:Mn * (Nn + 2):Nn + 2]) / 2
        prep1_end=timeit.default_timer()

        newton_time=timeit.default_timer()
        U_fast, info = newton(fn_fast, jac_fn, U_fast, cs_pe1, cs_ne1, gamma_p_vec, gamma_n_vec, idx_tot,re_idx, tol)
        newton_time_end=timeit.default_timer()
        tot_newton += (newton_time_end-newton_time)

        extra_time = timeit.default_timer()
        (fail, jf_time, overhead, solve_time) = info
        solve_time_tot += solve_time + c_lintime
        jf_tot_time += jf_time
        overhead_time += overhead

        # start_format = timeit.default_timer()

        Tvec_pe, Tvec_ne, phis_pe, phis_ne, jvec_pe, jvec_ne = unpack_vars(U_fast, Mp, Mn, Ms, Ma)

        cII_p = form_c2_p_jit(temp_p, jvec_pe, Tvec_pe).block_until_ready()
        cII_n = form_c2_n_jit(temp_n, jvec_ne, Tvec_ne).block_until_ready()
        cmat_pe = np.reshape(cII_p, [Mp * (Np + 2)], order="F").block_until_ready() + cI_pe_vec
        cmat_ne = np.reshape(cII_n, [Mn * (Nn + 2)], order="F").block_until_ready() + cI_ne_vec
        # cmat_pe = combine_c(cII_p, cI_pe_vec, Mp, Np)
        # cmat_ne = combine_c(cII_n, cI_ne_vec, Mn, Nn)

        voltages.append(phis_pe[1] - phis_ne[Mn])
        temps.append(np.mean(Tvec_pe[1:Mp + 1]))

        end_extra_time=timeit.default_timer()
        extra = extra + (end_extra_time-extra_time) + (prep_end - prep) + (prep1_end-prep1)

        # overhead_time += (end_format - start_format)
        if (fail == 0):
            pass
        #            print("timestep:", i)
        else:
            print('Premature end of run\n')
            print("timestep:", i)
            break

    end1 = timeit.default_timer();
    tot_time = (end1 - start1)
    time = (tot_time, solve_time_tot, jf_tot_time, overhead_time, init_time, extra)
    print("extra time:", extra)
    print("total newton time:", tot_newton)
    print("Done reordered simulation\n")
    return U_fast, cmat_pe, cmat_ne, voltages, temps, time



