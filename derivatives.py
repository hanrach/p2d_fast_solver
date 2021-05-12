from jax import vmap, grad, jit
from build_der import *
from functools import partial
from unpack import unpack_fast

class HashableDict(dict):
    def __hash__(self):
        return hash(frozenset(self.items()))

def partials(accq, peq, sepq, neq, zccq):

    dup = jit(vmap(grad(peq.electrolyte_conc, argnums=range(0, 8))))
    djp = jit(vmap(grad(peq.ionic_flux_fast, argnums=range(0, 4))))

    detap = jit(vmap(grad(peq.over_poten_fast, argnums=range(0, 5))))

    dphisp = vmap(grad(peq.solid_poten, argnums=range(0, 4)))

    dphiep = vmap(grad(peq.electrolyte_poten, argnums=range(0, 10)))

    dTp = vmap(grad(peq.temperature_fast, argnums=range(0, 12)))


    dup_bc0 = grad(peq.bc_zero_neumann, argnums=(0, 1))
    dup_bcM = grad(peq.bc_u_sep_p, argnums=range(0, 8))

    dphisp_bc0 = grad(peq.bc_phis, argnums=(0, 1))
    dphisp_bcM = grad(peq.bc_phis, argnums=(0, 1))

    dphiep_bc0 = grad(peq.bc_zero_neumann, argnums=(0, 1))
    dphiep_bcM = grad(peq.bc_phie_p, argnums=range(0, 12))

    dTp_bc0 = grad(peq.bc_temp_ap, argnums=range(0, 4))
    dTp_bcM = grad(peq.bc_temp_ps, argnums=range(0, 4))



    bc_s = grad(peq.bc_inter_cont, argnums=range(0, 4))

    dus = vmap(grad(sepq.electrolyte_conc, argnums=range(0, 6)))

    dphies = vmap(grad(sepq.electrolyte_poten, argnums=range(0, 9)))


    dTs = vmap(grad(sepq.temperature, argnums=range(0, 8)))



    dun_bc0 = grad(neq.bc_u_sep_n, argnums=range(0, 8))
    dun = vmap(grad(neq.electrolyte_conc, argnums=range(0, 8)))
    dun_bcM = grad(neq.bc_zero_neumann, argnums=(0, 1))

    djn = vmap(grad(neq.ionic_flux_fast, argnums=range(0, 4)))


    detan = vmap(grad(neq.over_poten_fast, argnums=range(0, 5)))


    dphisn = vmap(grad(neq.solid_poten, argnums=range(0, 4)))
    dphisn_bc0 = grad(neq.bc_phis, argnums=(0, 1))
    dphisn_bcM = grad(neq.bc_phis, argnums=(0, 1))

    dphien = vmap(grad(neq.electrolyte_poten, argnums=range(0, 10)))

    dphien_bc0 = grad(neq.bc_phie_n, argnums=range(0, 12))
    dphien_bcM = grad(neq.bc_zero_dirichlet, argnums=(0, 1))


    dTn = vmap(grad(neq.temperature_fast, argnums=range(0, 12)))


    dTn_bc0 = grad(neq.bc_temp_sn, argnums=range(0, 4))
    dTn_bcM = grad(neq.bc_temp_n, argnums=range(0, 4))



    dTa_bc0 = grad(accq.bc_temp_a, argnums=(0, 1))
    dTa = vmap(grad(accq.temperature, argnums=range(0, 3)))
    dTa_bcM = grad(peq.bc_inter_cont, argnums=range(0, 4))

    dTz_bc0 = grad(neq.bc_inter_cont, argnums=range(0, 4))
    dTz = vmap(grad(zccq.temperature, argnums=range(0, 3)))
    dTz_bcM = grad(zccq.bc_temp_z, argnums=(0, 1))

    d = dict([
        ('dup', dup),
        ('djp', djp),
        ('detap', detap),
        ('dphisp', dphisp),
        ('dphiep', dphiep),
        ('dTp', dTp),
        ('dup_bc0',dup_bc0),
        ('dup_bcM', dup_bcM),
        ('dphisp_bc0', dphisp_bc0),
        ('dphisp_bcM', dphisp_bcM),
        ('dphiep_bc0', dphiep_bc0),
        ('dphiep_bcM', dphiep_bcM),
        ('dTp_bc0', dTp_bc0),
        ('dTp_bcM', dTp_bcM),

        ('dus', dus),
        ('dphies', dphies),
        ('dTs', dTs),
        ('bc_s', bc_s),

        ('dun', dun),
        ('djn', djn),
        ('detan', detan),
        ('dphisn', dphisn),
        ('dphien', dphien),
        ('dTn', dTn),
        ('dun_bc0', dun_bc0),
        ('dun_bcM', dun_bcM),
        ('dphien_bc0', dphien_bc0),
        ('dphien_bcM', dphien_bcM),
        ('dphisn_bc0', dphisn_bc0),
        ('dphisn_bcM', dphisn_bcM),
        ('dTn_bc0', dTn_bc0),
        ('dTn_bcM', dTn_bcM),

        ('dTa_bc0', dTa_bc0),
        ('dTa', dTa),
        ('dTa_bcM', dTa_bcM),

        ('dTz_bc0', dTz_bc0),
        ('dTz', dTz),
        ('dTz_bcM', dTz_bcM)
    ])

    return HashableDict(d)


# @partial(jax.jit, static_argnums=(4,5,6))
def compute_jac(gamma_p_vec, gamma_n_vec, d, grid_num, vars):
    @jax.jit
    def jacfn(U, Uold, cs_pe1, cs_ne1):
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
    return jacfn






