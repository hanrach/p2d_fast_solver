from jax import lax
import jax
import jax.numpy as np


@jax.jit
def array_update(state, update_element):
    element, ind = update_element
    J, start_index, row = state
    return (jax.ops.index_update(J, jax.ops.index[row, start_index + 6 * ind], element), start_index, row), ind

@jax.jit
def array_update_sep(state, update_element):
    element, ind = update_element
    J, start_index, row = state
    return (jax.ops.index_update(J, jax.ops.index[row, start_index + 3 * ind], element), start_index, row), ind



@jax.partial(jax.jit, static_argnums=(2, 3,))
def build_dup(J, partial, Ma, Mp):
    ranger_p = np.arange(1, Mp)
    ranger_m = np.arange(0, Mp - 1)
    # dudTp
    J, _, _ = lax.scan(array_update, (J, 5  + 5+ Ma + 1, 0), (partial[5][0:-1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[2, Ma + 1 + 5 + 6 * Mp + 3], partial[5][-1])
    # dudum

    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 17), (partial[0][1:Mp], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[16, Ma + 1], partial[0][0])
    # duduc
    ranger_c = np.arange(0, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 11), (partial[1][0:Mp], ranger_c))[0]
    # dudup
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 6, 5), (partial[2][0:Mp], ranger_c))[0]
    # dudTm
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 12), (partial[3][1:Mp], ranger_m))[0]
    # TODO
    J = jax.ops.index_update(J, jax.ops.index[13, Ma + 1 + 3], partial[3][0])
    # dup[3][0] not assigned

    # dudTc
    # TODO : error
    J, _, _ = lax.scan(array_update, (J, 5 + 5 + Ma + 1, 6), (partial[4][0:Mp], ranger_c))[0]

    # dudj
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 1, 10), (partial[6][0:Mp], ranger_c))[0]
    return J


@jax.partial(jax.jit, static_argnums=(2, 3,))
def build_djp(J, partial, Ma, Mp):
    ranger_c = np.arange(0, Mp)
    # djdj
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 1, 11), (partial[0][0:Mp], ranger_c))[0]
    # djdu
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 12), (partial[1][0:Mp], ranger_c))[0]
    # djdT
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 7), (partial[2][0:Mp], ranger_c))[0]
    # djdeta
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 2, 10), (partial[3][0:Mp], ranger_c))[0]
    return J


@jax.partial(jax.jit, static_argnums=(2, 3,))
def build_detap(J, partial, Ma, Mp):
    # detadeta
    ranger_c = np.arange(0, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 2, 11), (partial[0][0:Mp], ranger_c))[0]
    # deta dphis
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 3, 10), (partial[1][0:Mp], ranger_c))[0]
    # deta dphie
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 4, 9), (partial[2][0:Mp], ranger_c))[0]
    # detadT
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 8), (partial[3][0:Mp], ranger_c))[0]
    # deta dj
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 1, 12), (partial[4][0:Mp], ranger_c))[0]
    return J


@jax.partial(jax.jit, static_argnums=(2, 3,))
def build_dphisp(J, partial, Ma, Mp):
    ranger_m = np.arange(0, Mp - 1)
    # dphis dphism
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 3, 17), (partial[0][1:Mp], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[18, Ma + 1 + 1], partial[0][0])
    # dphis dphisc
    ranger_c = np.arange(0, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 3, 11), (partial[1][0:Mp], ranger_c))[0]
    # dphis dphisp
    ranger_p = np.arange(1, Mp)
    # J = jax.ops.index_update(J, jax.ops.index[4, Ma + 1 + 5 + 3], partial[2][0])
    # J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 3, 5), (partial[2][1:Mp], ranger_p))[0]

    # TODO: check
    J = jax.ops.index_update(J, jax.ops.index[7, 5 + 6*Mp + 1 + Ma + 1], partial[2][Mp-1])
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 3, 5), (partial[2][0:Mp-1], ranger_p))[0]


    # dphis dj
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 1, 13), (partial[3][0:Mp], ranger_c))[0]
    return J


@jax.partial(jax.jit, static_argnums=(2, 3,))
def build_dphiep(J, partial, Ma, Mp):
    # dphie/dum
    ranger_m = np.arange(0, Mp - 1)
    J = jax.ops.index_update(J, jax.ops.index[20, Ma + 1], partial[0][0])
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 21), (partial[0][1:Mp], ranger_m))[0]
    # dphie/duc
    ranger_c = np.arange(0, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 15), (partial[1][0:Mp], ranger_c))[0]
    # dphie/dup
    J, _, _ = lax.scan(array_update, (J, 5 + 6 * 1 + Ma + 1, 9), (partial[2][0:Mp], ranger_c))[0]
    # dphie/dphiem
    J = jax.ops.index_update(J, jax.ops.index[18, Ma + 1 + 2], partial[3][0])  # check
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 4, 17), (partial[3][1:Mp], ranger_m))[0]
    # dphie/dphiec
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 4, 11), (partial[4][0:Mp], ranger_c))[0]
    # dphie/dphiep
    ranger_p = np.arange(1, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 4, 5), (partial[5][0:Mp - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[7, 5 + 6 * Mp + 2 + Ma + 1], partial[5][Mp - 1])
    # dphiep/dTm
    # TODO
    # dphpep/dTm[0] check

    J, _, _ = lax.scan(array_update, (J, 5 + 5 + Ma + 1, 16), (partial[6][1:Mp], ranger_m))[0]

    # dphiep/dTc
    J, _, _ = lax.scan(array_update, (J, 5 + 5 + Ma + 1, 10), (partial[7][0:Mp], ranger_c))[0]

    # dphiep/dTp
    J, _, _ = lax.scan(array_update, (J, 5 + 5 + Ma + 1, 4), (partial[8][0:Mp - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[6, 5 + 6 * Mp + 3 + Ma + 1], partial[8][Mp - 1])

    # dphie/dj
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 1, 14), (partial[9][0:Mp], ranger_c))[0]
    return J


@jax.partial(jax.jit, static_argnums=(2, 3,))
def build_dTp(J, partial, Ma, Mp):
    # dT/dum
    ranger_m = np.arange(0, Mp - 1)
    J = jax.ops.index_update(J, jax.ops.index[21, Ma + 1], partial[0][0])
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 22), (partial[0][1:Mp], ranger_m))[0]

    # dT/duc
    ranger_c = np.arange(0, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5, 16), (partial[1][0:Mp], ranger_c))[0]

    # dT/dup
    J, _, _ = lax.scan(array_update, (J, 5 + 6 + Ma + 1, 10), (partial[2][0:Mp], ranger_c))[0]

    # dT/phiem
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 4, 18), (partial[3][1:Mp], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[19, Ma + 1 + 2], partial[3][0])

    # dT/dphiep
    ranger_p = np.arange(1, Mp)
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 4, 6), (partial[4][0:Mp - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[8, 5 + 6 * Mp + 2 + Ma + 1], partial[4][Mp - 1])

    # dT/dphism
    J, _, _ = lax.scan(array_update, (J, 5 + 3 + Ma + 1, 19), (partial[5][1:Mp], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[20, Ma + 1 + 1], partial[5][0])

    # dT/dphisp
    J, _, _ = lax.scan(array_update, (J, 5 + 3 + Ma + 1, 7), (partial[6][0:Mp - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[9, 5 + 6 * Mp + 2 + Ma], partial[6][Mp - 1])

    # dT/dTm
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 17), (partial[7][1:Mp], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[18, Ma + 1 + 3], partial[7][0])

    # dT/dTc
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 11), (partial[8][0:Mp], ranger_c))[0]

    # dT/dTp
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 5, 5), (partial[9][0:Mp - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[7, 5 + 6 * Mp + 2 + Ma + 1 + 1], partial[9][Mp])

    # dT/dj
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 1, 15), (partial[10][0:Mp], ranger_c))[0]
    # dT/deta
    J, _, _ = lax.scan(array_update, (J, Ma + 1 + 5 + 2, 14), (partial[11][0:Mp], ranger_c))[0]
    return J


@jax.jit
def build_bc_p(J, bc, Ma, Mp):
    # u0 and uM
    p0 = Ma + 1
    sep_o = 4 * (Mp + 2) + 2 * Mp + 1
    row_u = np.array([11, 6, 17, 11, 12, 8, 7, 4, 5, 2])
    col_u = p0 + np.array(
        [0, 5, 6 * (Mp - 1) + 5, 6 * Mp + 5, 6 * (Mp - 1) + 10, 6 * Mp + 5 + 3, sep_o, sep_o + 3, sep_o + 2,
         sep_o + 2 + 3])
    J = jax.ops.index_update(J, jax.ops.index[row_u, col_u], bc['u'])

    row_phis = np.array([11, 4, 15, 11])
    col_phis = p0 + np.array([1, 5 + 3, 5 + 6*(Mp-1) + 3, 5 + 6*Mp + 1])
    J = jax.ops.index_update(J, jax.ops.index[row_phis, col_phis], bc['phis'])
    # TODO: check if this is correct
    J = jax.ops.index_update(J, jax.ops.index[7, p0 + 5 + 6*Mp + 1 ], 1)

    row_phie = np.array([11, 4, 15, 11, 8, 5, 19, 13, 9, 6, 14, 10, 7, 4])
    col_phie = p0 + np.array([2, 5 + 4, 5 + 6*(Mp-1) + 4, 5 + 6*Mp +2,
                              sep_o + 1, sep_o + 3 + 1, 5+6*(Mp-1),
                              5 + 6*Mp, sep_o, sep_o + 3, 5 + 6*(Mp-1) + 5, 5 + 6*Mp + 3,
                              sep_o + 2, sep_o + 3 + 2])
    J = jax.ops.index_update(J, jax.ops.index[row_phie, col_phie], bc['phie'])

    row_T = np.array([15, 10, 11, 4, 15, 11, 8, 5])
    col_T = np.array([Ma, Ma + 1 + 4, Ma + 1 + 3, Ma + 1 + 10, 5 + 6*(Mp-1)+5+Ma+1, 5+6*Mp+3+Ma+1, Ma + 1 + sep_o + 2, Ma + 1 + sep_o +2 + 3])
    J = jax.ops.index_update(J, jax.ops.index[row_T, col_T], bc['T'])
    return J

# @jax.jit
def build_bc_s(J, bc, Ma, Mp, Ms):
    p0 = Ma + 1
    sep0 = 4 * (Mp + 2) + 2 * Mp + 1 + p0

    n0 = p0 + 4*(Mp+2) + 2*Mp  + 3*(Ms+2) + 1
    row_u = np.array([11, 8, 21, 15, 8, 4, 14, 11])
    col_u = np.array([sep0, sep0 + 3, 5 + 6*(Mp-1) + p0, 5+6*Mp + p0, n0, n0 + 4, sep0 + 3*Ms, sep0 + 3*(Ms+1) ])
    J = jax.ops.index_update(J, jax.ops.index[row_u, col_u], bc)

    row_phie = np.array([18, 14, 11, 8, 7, 1, 14, 11])
    col_phie = np.array([5+6*(Mp-1) + 4 + p0, 5 + 6*Mp +2 + p0, sep0+1, sep0 + 1 + 3, n0+2, n0+4 + 4, sep0 + 3*Ms + 1, sep0 + 3*Ms + 4])

    J = jax.ops.index_update(J, jax.ops.index[row_phie, col_phie], bc)

    row_T = np.array([18, 14, 11,8, 14, 11, 7, 1])
    col_T = np.array([5 + 6*(Mp-1) + 5 + p0, 5 + 6*Mp + 3 + p0, sep0 + 2, sep0 + 5, sep0 + 3*Ms + 2, sep0 + 3*Ms + 5, n0 + 3, n0 + 4 + 5])
    J = jax.ops.index_update(J, jax.ops.index[row_T, col_T], bc)

    return J

# @jax.partial(jax.jit, static_argnums=(2,3,4))
def build_dus(J, partial, Ma, Mp, Ms):
    ranger_c = np.arange(0,Ms)
    p0 = Ma + 1
    sep0 = 4*(Mp+2) + 2*Mp + 1 + p0
    # dus/dum
    J, _, _ = lax.scan(array_update_sep, (J, sep0, 14), (partial[0][0:Ms], ranger_c))[0]
    # dus/duc
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 3, 11), (partial[1][0:Ms], ranger_c))[0]
    #dus/dup
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 6, 8), (partial[2][0:Ms], ranger_c))[0]
    #dus/dTm
    J,_,_ = lax.scan(array_update_sep,(J, sep0 + 2, 12), (partial[3][0:Ms], ranger_c))[0]
    #dus/dTc
    J,_,_ = lax.scan(array_update_sep, (J, sep0 + 5, 9), (partial[4][0:Ms], ranger_c))[0]
    #dus/dTp
    J,_,_ = lax.scan(array_update_sep, (J, sep0 + 8, 6), (partial[5][0:Ms], ranger_c))[0]
    return J

# @jax.partial(jax.jit, static_argnums=(2,3,4))
def build_dphies(J, partial, Ma, Mp, Ms):
    ranger_c = np.arange(0,Ms)
    p0 = Ma + 1
    sep0 = 4*(Mp+2) + 2*Mp + 1 + p0
    # dphie/dum
    J, _, _ = lax.scan(array_update_sep, (J, sep0, 15), (partial[0][0:Ms], ranger_c))[0]
    # dphie/duc
    J, _, _ = lax.scan(array_update_sep, (J, sep0+3, 12), (partial[1][0:Ms], ranger_c))[0]
    #dphie/dup
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 6, 9), (partial[2][0:Ms], ranger_c))[0]
    #dphie/dphiem
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 1, 14), (partial[3][0:Ms], ranger_c))[0]
    #dphie/dphiec
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 4, 11), (partial[4][0:Ms], ranger_c))[0]
    #dphie/dphiep
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 7, 8), (partial[5][0:Ms], ranger_c))[0]
    #dphie/dTm
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 2, 13), (partial[6][0:Ms], ranger_c))[0]
    #dphie/dTc
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 5, 10), (partial[7][0:Ms], ranger_c))[0]
    #dphie/dTp
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 8, 7), (partial[8][0:Ms], ranger_c))[0]
    return J

# @jax.partial(jax.jit, static_argnums=(2,3,4))
def build_dTs(J, partial, Ma, Mp, Ms):
    ranger_c = np.arange(0,Ms)
    p0 = Ma + 1
    sep0 = 4*(Mp+2) + 2*Mp + 1 + p0
    # dT/dum
    J,_,_ = lax.scan(array_update_sep, (J, sep0, 16), (partial[0][0:Ms], ranger_c))[0]
    # dT/duc
    J, _, _ = lax.scan(array_update_sep, (J, sep0+3, 13), (partial[1][0:Ms], ranger_c))[0]
    # dT/dup
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 6, 10), (partial[2][0:Ms], ranger_c))[0]
    # dT/dphiem
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 1, 15), (partial[3][0:Ms], ranger_c))[0]
    # dT/dphiep
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 7, 9), (partial[4][0:Ms], ranger_c))[0]
    # dT/dTm
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 2, 14), (partial[5][0:Ms], ranger_c))[0]
    # dT/dTc
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 5, 11), (partial[6][0:Ms], ranger_c))[0]
    # dT/dTp
    J, _, _ = lax.scan(array_update_sep, (J, sep0 + 8, 8), (partial[7][0:Ms], ranger_c))[0]
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4))
def build_bc_n(J, bc, Ma, Mp, Ms, Mn):
    p0 = Ma + 1
    sep0 = 4*(Mp+2) + 2*Mp + 1 + p0
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    row_u = np.array([11, 7, 8, 2, 17, 14, 15, 12, 17, 11])
    col_u = np.array([n0, n0 + 4, n0+3, n0 + 9,
                      sep0 + 3*Ms, sep0 + 3*Ms + 3,
                      sep0 + 3*Ms + 2, sep0 + 3*Ms + 5,
                      n0 + 4 + 6*(Mn-1), n0 + 4 + 6*Mn])
    J = jax.ops.index_update(J, jax.ops.index[row_u, col_u], bc['u'])

    row_phie = np.array([11, 5, 18, 15, 13, 9, 19, 16, 10, 4, 17, 14, 15, 11])
    col_phie = np.array([n0 + 2, n0 + 8, sep0 + 3*Ms + 1, sep0 + 3*Ms + 4,
                         n0, n0 + 4, sep0 + 3*Ms, sep0 + 3*(Ms + 1),
                         n0 + 3, n0 + 9, sep0 + 3*Ms + 2, sep0 + 3*Ms + 5, n0 + 4 + 6*(Mn-1) + 4, n0 + 4 + 6*Mn +2])
    J = jax.ops.index_update(J, jax.ops.index[row_phie, col_phie], bc['phie'])

    row_phis = np.array([11, 5, 15, 11])
    col_phis = np.array([n0 + 1, n0 + 7, n0 + 4 + 6*(Mn-1) + 3, n0 + 4 + 6*Mn + 1])
    J = jax.ops.index_update(J, jax.ops.index[row_phis, col_phis], bc['phis'])

    row_T = np.array([18,15,11, 5, 15, 11, 10, 9])
    col_T = np.array([sep0 + 3*Ms + 2, sep0 + 3*Ms + 5, n0 + 3, n0 + 9,
                      n0 + 4 + 6*(Mn-1) + 5, n0 + 4 + 6*Mn + 3, n0 + 4 + 6*Mn + 4, n0 + 4 + 6*Mn + 4 + 1 ])
    J = jax.ops.index_update(J, jax.ops.index[row_T, col_T], bc['T'])
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dun(J, partial, Ma, Mp, Ms, Mn):
    p0 = Ma + 1
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    ranger_p = np.arange(1, Mn)
    ranger_m = np.arange(0, Mn - 1)
    # dudTp
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 0), (partial[5][0:-1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[2, n0 + 4 + 6*Mn], partial[5][-1])
    # dudum

    J, _, _ = lax.scan(array_update, (J, n0 + 4, 17), (partial[0][1:Mn], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[15, n0], partial[0][0])
    # duduc
    ranger_c = np.arange(0, Mn)
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 11), (partial[1][0:Mp], ranger_c))[0]
    # dudup
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 6, 5), (partial[2][0:Mp], ranger_c))[0]
    # dudTm
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 12), (partial[3][1:Mp], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[12, n0 + 3], partial[3][0])
    # dudTc
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 6), (partial[4][0:Mp], ranger_c))[0]
    # dudj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 10), (partial[6][0:Mp], ranger_c))[0]
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_djn(J, partial, Ma, Mp, Ms, Mn):
    ranger_c = np.arange(0, Mn)
    p0 = Ma + 1
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    # djdj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 11), (partial[0][0:Mp], ranger_c))[0]
    # djdu
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 12), (partial[1][0:Mp], ranger_c))[0]
    # djdT
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 7), (partial[2][0:Mp], ranger_c))[0]
    # djdeta
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 2, 10), (partial[3][0:Mp], ranger_c))[0]
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_detan(J, partial, Ma, Mp, Ms, Mn):
    p0 = Ma + 1
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    # detadeta
    ranger_c = np.arange(0, Mn)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 2, 11), (partial[0][0:Mp], ranger_c))[0]
    # deta dphis
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 10), (partial[1][0:Mp], ranger_c))[0]
    # deta dphie
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 9), (partial[2][0:Mp], ranger_c))[0]
    # detadT
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 8), (partial[3][0:Mp], ranger_c))[0]
    # deta dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 12), (partial[4][0:Mp], ranger_c))[0]
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dphisn(J, partial, Ma, Mp, Ms, Mn):
    p0 = Ma + 1
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    ranger_m = np.arange(0, Mn - 1)
    # dphis dphism
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 17), (partial[0][1:Mn], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[17, n0 + 1], partial[0][0])
    # dphis dphisc
    ranger_c = np.arange(0, Mn)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 11), (partial[1][0:Mn], ranger_c))[0]
    # dphis dphisp
    ranger_p = np.arange(1, Mn)
    J = jax.ops.index_update(J, jax.ops.index[7, n0 + 4 + 6*Mn + 1 ], partial[2][Mn-1])
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 3, 5), (partial[2][0:Mn-1], ranger_p))[0]
    # dphis dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 13), (partial[3][0:Mp], ranger_c))[0]
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dphien(J, partial, Ma, Mp, Ms, Mn):
    p0 = Ma + 1
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    # dphie/dum
    ranger_m = np.arange(0, Mn - 1)
    J = jax.ops.index_update(J, jax.ops.index[19, n0], partial[0][0])
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 21), (partial[0][1:Mn], ranger_m))[0]
    # dphie/duc
    ranger_c = np.arange(0, Mn)
    J, _, _ = lax.scan(array_update, (J,n0 + 4, 15), (partial[1][0:Mn], ranger_c))[0]
    # dphie/dup
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 6, 9), (partial[2][0:Mn], ranger_c))[0]
    # dphie/dphiem
    J = jax.ops.index_update(J, jax.ops.index[17, n0 + 2], partial[3][0])  # check
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 17), (partial[3][1:Mn], ranger_m))[0]
    # dphie/dphiec
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 11), (partial[4][0:Mn], ranger_c))[0]
    # dphie/dphiep
    ranger_p = np.arange(1, Mn)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 5), (partial[5][0:Mn - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[7, n0 + 4 + 6*Mn + 2], partial[5][Mn - 1])

    # dphiep/dTm
    J = jax.ops.index_update(J, jax.ops.index[16, n0 + 3], partial[6][0])
    J, _, _ = lax.scan(array_update, (J, 5 + 4 + n0, 16), (partial[6][1:Mn], ranger_m))[0]

    # dphiep/dTc
    J, _, _ = lax.scan(array_update, (J, 5 + n0 + 4, 10), (partial[7][0:Mp], ranger_c))[0]

    # dphiep/dTp
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 4), (partial[8][0:Mp - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[6, n0 + 4 + 6*Mn + 3], partial[8][Mp - 1])

    # dphie/dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 14), (partial[9][0:Mp], ranger_c))[0]
    return J

@jax.partial(jax.jit, static_argnums=(2,3,4,5,))
def build_dTn(J, partial, Ma, Mp, Ms, Mn):
    p0 = Ma + 1
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    # dT/dum
    ranger_m = np.arange(0, Mn - 1)
    J = jax.ops.index_update(J, jax.ops.index[20, n0], partial[0][0])
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 22), (partial[0][1:Mn], ranger_m))[0]

    # dT/duc
    ranger_c = np.arange(0, Mn)
    J, _, _ = lax.scan(array_update, (J, n0 + 4, 16), (partial[1][0:Mn], ranger_c))[0]

    # dT/dup
    J, _, _ = lax.scan(array_update, (J, 4 + 6 + n0, 10), (partial[2][0:Mn], ranger_c))[0]

    # dT/phiem
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 18), (partial[3][1:Mn], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[18, n0 + 2], partial[3][0])

    # dT/dphiep
    ranger_p = np.arange(1, Mp)
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 4, 6), (partial[4][0:Mn - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[8, n0 + 4 + 6*Mn + 2 ], partial[4][Mn - 1])

    # dT/dphism
    J, _, _ = lax.scan(array_update, (J,n0 + 4 + 3, 19), (partial[5][1:Mn], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[19, n0 + 1], partial[5][0])

    # dT/dphisp
    J, _, _ = lax.scan(array_update, (J, 4 + 3 + n0, 7), (partial[6][0:Mn - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[9, n0 + 4 + 6*Mn + 1 ], partial[6][Mn - 1])

    # dT/dTm
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 17), (partial[7][1:Mn], ranger_m))[0]
    J = jax.ops.index_update(J, jax.ops.index[17, n0 + 3], partial[7][0])

    # dT/dTc
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 11), (partial[8][0:Mn], ranger_c))[0]

    # dT/dTp
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 5, 5), (partial[9][0:Mn - 1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[7, n0 + 4 + 6*Mn + 3], partial[9][Mn])

    # dT/dj
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 1, 15), (partial[10][0:Mn], ranger_c))[0]
    # dT/deta
    J, _, _ = lax.scan(array_update, (J, n0 + 4 + 2, 14), (partial[11][0:Mn], ranger_c))[0]
    return J

# @jax.partial(jax.jit, static_argnums=(2,3,4,5,6,))
def build_bc_cc(J, bc, Ma, Mp, Ms, Mn, Mz):
    p0 = Ma + 1
    sep0 = 4 * (Mp + 2) + 2 * Mp + 1 + p0
    n0 = p0 + 4*(Mp+2) + 2*Mp  + 3*(Ms+2) + 1
    row_dTa = np.array([11, 10, 16, 11, 12, 5])
    col_dTa = np.array([0, 1, Ma, 4 + p0, 3 + p0, 10 + p0])

    J = jax.ops.index_update(J, jax.ops.index[row_dTa, col_dTa], bc['acc'])
    row_dTz = np.array([16, 12, 11, 10, 12, 11])
    col_dTz = np.array([n0 + 4 + 6*(Mn-1)+5, n0 + 4 + 6*Mn + 3,n0 + 4 + 6*Mn + 4, n0 + 4 + 6*Mn + 5,
                        n0 + 4 + 6*Mn + 4 + Mz,n0 + 4 + 6*Mn + 4 + Mz+1 ])
    J = jax.ops.index_update(J, jax.ops.index[row_dTz, col_dTz], bc['zcc'])
    return J

@jax.jit
def array_update_acc(state, update_element):
    element, ind = update_element
    J, start_index, row = state
    return (jax.ops.index_update(J, jax.ops.index[row, start_index + ind], element), start_index, row), ind

# @jax.partial(jax.jit, static_argnums=(2,))
def build_dTa(J, partial, Ma):
    ranger = np.arange(0, Ma)
    # dT/dTm
    J,_,_ = lax.scan(array_update_acc, (J, 0, 12), (partial[0][0:Ma], ranger) )[0]
    # dT/dTc
    J,_,_ = lax.scan(array_update_acc, (J, 1, 11), (partial[1][0:Ma], ranger))[0]
    # dT/dTp
    ranger_p = np.arange(0, Ma-1)
    J,_,_ = lax.scan(array_update_acc, (J, 2, 10), (partial[2][0:Ma-1], ranger_p))[0]
    J = jax.ops.index_update(J, jax.ops.index[6,Ma + 1 + 4], partial[2][Ma-1])

    return J

# @jax.partial(jax.jit, static_argnums=(2,3,4,5,6,))
def build_dTz(J, partial, Ma, Mp, Ms, Mn, Mz):
    p0 = Ma + 1
    sep0 = 4 * (Mp + 2) + 2 * Mp + 1 + p0
    n0 = p0 + 4 * (Mp + 2) + 2 * Mp + 3 * (Ms + 2) + 1
    ranger = np.arange(0, Mz)
    # dT/dTm
    J,_,_ = lax.scan(array_update_acc, (J, n0 + 4 + 6*Mn + 4, 12), (partial[0][0:Mz], ranger) )[0]
    # dT/dTc
    J,_,_ = lax.scan(array_update_acc, (J, n0 + 4 + 6*Mn + 5, 11), (partial[1][0:Mz], ranger))[0]
    # dT/dTp
    J,_,_ = lax.scan(array_update_acc, (J,  n0 + 4 + 6*Mn + 6, 10), (partial[2][0:Ma], ranger))[0]

    return J