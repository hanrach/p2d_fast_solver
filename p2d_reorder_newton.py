
from functools import partial
from jax.scipy.linalg import solve
import jax.numpy as np
from scipy.linalg import solve_banded
from jax.numpy.linalg import norm
import jax
import numpy as onp
import timeit

@jax.jit
def process_y(y,idx):
    y1=y[idx]
    return y1

@jax.jit
def reorder_vec(b, idx):
    return b[idx]
def newton(fn_fast, jac_fn_fast, U, cs_pe1, cs_ne1, gamma_p, gamma_n, idx, re_idx):
    maxit = 10
    tol = 1e-7

    res = 100
    fail = 0
    Uold = U

    start = timeit.default_timer();
    J = jac_fn_fast(U, Uold, cs_pe1, cs_ne1).block_until_ready()
    y = fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
    end = timeit.default_timer();
    jf_time0 = end - start
    # print("Initial J and f evaluation:", end-start);

    start = timeit.default_timer();
    y = process_y(y, idx).block_until_ready()
    end = timeit.default_timer();
    overhead_r = end - start
    # print("Reordering overhead:", end-start)

    start = timeit.default_timer();
    delta = solve_banded((11, 11), J, y)
    end = timeit.default_timer();
    solve_time0 = end - start

    delta_reordered = reorder_vec(delta, re_idx)
    U = U - delta_reordered
    #    res0 = norm(y/norm(U,np.inf),np.inf)
    count = 1

    jf_time = 0;
    overhead = overhead_r;
    solve_time = solve_time0;
    while (count < maxit and res > tol):
        start = timeit.default_timer();
        J = jac_fn_fast(U, Uold, cs_pe1, cs_ne1).block_until_ready()
        y = fn_fast(U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n).block_until_ready()
        end = timeit.default_timer();
        jf_time += end - start;

        start = timeit.default_timer();
        y = process_y(y, idx).block_until_ready();
        end = timeit.default_timer();
        overhead += end - start

        res = norm(y / norm(U, np.inf), np.inf)

        start = timeit.default_timer();
        delta = solve_banded((11, 11), J, y)
        end = timeit.default_timer();
        solve_time += end - start

        start = timeit.default_timer()
        delta_reordered = reorder_vec(delta, re_idx).block_until_ready()
        end = timeit.default_timer()
        overhead += end - start

        U = U - delta_reordered
        count = count + 1
        print(count, res)

    #    print("Total to evaluate Jacobian:",jf_time)
    if fail == 0 and np.any(np.isnan(delta)):
        fail = 1

        print("nan solution")

    if fail == 0 and max(abs(np.imag(delta))) > 0:
        fail = 1
        print("solution complex")

    if fail == 0 and res > tol:
        fail = 1;
        print('Newton fail: no convergence')
    else:
        fail == 0

    info = (fail, jf_time, overhead, solve_time)
    return U, info