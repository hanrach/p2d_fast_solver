from jax import vmap, grad, jit

def partials(accq,peq, sepq, neq, zccq):

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



    dun_bc0 = grad(neq.bc_u_sep_n, argnums=(0, 1))
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




