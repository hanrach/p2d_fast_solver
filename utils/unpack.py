
import jax.numpy as np
import jax
from jax.lax import dynamic_slice
from functools import partial
def unpack(U, Mp, Np, Mn, Nn, Ms, Ma, Mz):
    
    cmat_pe = np.reshape(U[0:Mp*(Np+2)], [Np+2, Mp], order="F") 
    cmat_ne = np.reshape(U[Mp*(Np+2): Mp*(Np+2) + Mn*(Nn+2)], [Nn+2, Mn], order="F")
    
    up0 =  Mp*(Np+2) + Mn*(Nn+2)
    usep0 = Mp*(Np+2) + Mn*(Nn+2) + Mp + 2
    un0 = Mp*(Np+2) + Mn*(Nn+2) + Mp + 2 + Ms+2
    
    jp0 = un0 + Mn + 2
    jn0 = jp0 + Mp
    
    etap0 = jn0 + Mn 
    etan0 = etap0 + Mp
    
    phisp0 = etan0 + Mn
    phisn0 = phisp0 + Mp + 2
    
    
    phiep0 = phisn0 + Mn +2
    phiesep0 = phiep0 + Mp + 2
    phien0 = phiesep0 + Ms + 2
    
    
    ta0 = phien0 + Mn + 2
    tp0 = ta0 + Ma+2
    tsep0 = tp0 + Mp+2
    tn0 = tsep0+ Ms+2
    tz0 = tn0 + Mn+2

#    up0 =  Mp*(Np+2) + Mn*(Nn+2)
    uvec_pe = U[up0: usep0]
    uvec_sep = U[usep0: un0]
    uvec_ne = U[un0: un0 + Mn + 2]

    Tvec_acc = U[ta0: tp0]
    Tvec_pe = U[tp0:tsep0]
    Tvec_sep = U[tsep0:tn0]
    Tvec_ne = U[tn0:tz0]
    Tvec_zcc = U[tz0:tz0 + Mz + 2]
    
    phie_pe = U[phiep0 : phiesep0]
    phie_sep = U[phiesep0: phien0]
    phie_ne = U[phien0:phien0 + Mn + 2]
    
   
    phis_pe = U[phisp0: phisn0]
    phis_ne = U[phisn0:phisn0 + Mn + 2]
    
    j_pe = U[jp0: jn0]
    j_ne = U[jn0: jn0 + Mn]
    
    eta_pe = U[etap0: etan0]
    eta_ne = U[etan0 : etan0+ Mn]
    
    return cmat_pe, cmat_ne, uvec_pe, uvec_sep, uvec_ne, Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, j_pe,j_ne,eta_pe,eta_ne
 
@partial(jax.jit, static_argnums=(1,2,3,4,))
def unpack_vars(U, Mp, Mn, Ms, Ma):
    up0 = 0
    usep0 = up0 + Mp + 2
    un0 = usep0 + Ms + 2

    jp0 = un0 + Mn + 2
    jn0 = jp0 + Mp

    etap0 = jn0 + Mn
    etan0 = etap0 + Mp

    phisp0 = etan0 + Mn
    phisn0 = phisp0 + Mp + 2

    phiep0 = phisn0 + Mn + 2
    phiesep0 = phiep0 + Mp + 2
    phien0 = phiesep0 + Ms + 2

    ta0 = phien0 + Mn + 2
    tp0 = ta0 + Ma + 2
    tsep0 = tp0 + Mp + 2
    tn0 = tsep0 + Ms + 2
    tz0 = tn0 + Mn + 2

    Tvec_pe = dynamic_slice(U, [tp0], [tsep0 - tp0])

    Tvec_ne = dynamic_slice(U, [tn0], [tz0 - tn0])

    phis_pe = dynamic_slice(U, [phisp0], [phisn0 - phisp0])
    phis_ne = dynamic_slice(U, [phisn0], [phisn0 + Mn + 2 - phisn0])

    j_pe = dynamic_slice(U, [jp0], [jn0 - jp0])
    j_ne = dynamic_slice(U, [jn0], [jn0 + Mn - jn0])

    return Tvec_pe, Tvec_ne, phis_pe, phis_ne, j_pe, j_ne

@partial(jax.jit, static_argnums=(1,2,3,4,5,6,7,))
def unpack_fast(U, Mp, Np, Mn, Nn, Ms, Ma, Mz):
    
    
    up0 =  0
    usep0 = up0 + Mp + 2
    un0 = usep0 + Ms+2
    
    jp0 = un0 + Mn + 2
    jn0 = jp0 + Mp
    
    etap0 = jn0 + Mn 
    etan0 = etap0 + Mp
    
    phisp0 = etan0 + Mn
    phisn0 = phisp0 + Mp + 2
    
    
    phiep0 = phisn0 + Mn +2
    phiesep0 = phiep0 + Mp + 2
    phien0 = phiesep0 + Ms + 2
    
    
    ta0 = phien0 + Mn + 2
    tp0 = ta0 + Ma+2
    tsep0 = tp0 + Mp+2
    tn0 = tsep0+ Ms+2
    tz0 = tn0 + Mn+2

#    up0 =  Mp*(Np+2) + Mn*(Nn+2)
#    uvec_pe = U[up0: usep0]
#    uvec_sep = U[usep0: un0]
#    uvec_ne = U[un0: un0 + Mn + 2]
#
#    Tvec_acc = U[ta0: tp0]
#    Tvec_pe = U[tp0:tsep0]
#    Tvec_sep = U[tsep0:tn0]
#    Tvec_ne = U[tn0:tz0]
#    Tvec_zcc = U[tz0:tz0 + Mz + 2]
#    
#    phie_pe = U[phiep0 : phiesep0]
#    phie_sep = U[phiesep0: phien0]
#    phie_ne = U[phien0:phien0 + Mn + 2]
#    
#   
#    phis_pe = U[phisp0: phisn0]
#    phis_ne = U[phisn0:phisn0 + Mn + 2]
#    
#    j_pe = U[jp0: jn0]
#    j_ne = U[jn0: jn0 + Mn]
#    
#    eta_pe = U[etap0: etan0]
#    eta_ne = U[etan0 : etan0+ Mn]
    
    uvec_pe = dynamic_slice(U, [up0], [usep0-up0])
    uvec_sep = dynamic_slice(U, [usep0], [un0-usep0])
    uvec_ne = dynamic_slice(U, [un0], [un0+Mn+2 - un0])

    Tvec_acc = dynamic_slice(U, [ta0], [tp0-ta0])
    Tvec_pe = dynamic_slice(U, [tp0], [tsep0-tp0])
    Tvec_sep = dynamic_slice(U, [tsep0], [tn0-tsep0])
    Tvec_ne = dynamic_slice(U, [tn0], [tz0-tn0])
    Tvec_zcc = dynamic_slice(U, [tz0], [tz0+Mz+2-tz0])
    
    phie_pe =dynamic_slice(U, [phiep0],[phiesep0-phiep0])
    phie_sep = dynamic_slice(U, [phiesep0],[phien0-phiesep0])
    phie_ne = dynamic_slice(U, [phien0],[phien0+Mn+2-phien0])
    
   
    phis_pe = dynamic_slice(U,[phisp0],[phisn0-phisp0])
    phis_ne = dynamic_slice(U,[phisn0],[phisn0+Mn+2-phisn0])
    
    j_pe = dynamic_slice(U,[jp0],[jn0-jp0])
    j_ne = dynamic_slice(U,[jn0],[jn0+Mn-jn0])
    
    eta_pe =dynamic_slice(U,[etap0],[etan0-etap0])
    eta_ne =dynamic_slice(U,[etan0],[etan0+Mn-etan0])
    
    return uvec_pe, uvec_sep, uvec_ne, Tvec_acc, Tvec_pe, Tvec_sep, Tvec_ne, Tvec_zcc, phie_pe, phie_sep, phie_ne, phis_pe, phis_ne, j_pe,j_ne,eta_pe,eta_ne
 
        
def unpack_fast_reorder(U, Mp):
    
    
    up0 =  0
    tp0 = up0 + Mp +2
    phiep0 = tp0 + Mp + 2
    phisp0 = phiep0 + Mp + 2
    jp0 = phisp0 + Mp +2
    etap0 = jp0 + Mp
    
    uvec_pe = U[up0:up0 + Mp +2]
    Tvec_pe = U[tp0:tp0+Mp+2]
    phie_pe = U[phiep0:phiep0+Mp+2]
    phis_pe = U[phisp0:phisp0+Mp+2]
    j_pe = U[jp0:jp0+Mp]
    eta_pe = U[etap0:etap0+Mp]
    
    return uvec_pe, Tvec_pe, phie_pe, phis_pe, j_pe, eta_pe

def unpack_fast_reorder2(U, Mp, Mn, Ms, Ma, Mz):
    
    
    up0 =  0
    tp0 = up0 + Mp +2
    phiep0 = tp0 + Mp + 2
    phisp0 = phiep0 + Mp +2
    
    usep0 = phisp0 + Mp + 2
    tsep0 = usep0 + Ms + 2
    phiesep0 = tsep0 + Ms + 2
    
    un0=  phiesep0 + Ms +2
    tn0 = un0 + Mn +2
    phien0 = tn0 + Mn + 2
    phisn0 = phien0 + Mn +2
    
    ta0 = phisn0 + Mn +2
    tz0 = ta0  + Ma+2
    jp0 = tz0 + Mz+2
    etap0 = jp0 + Mp
    jn0 = etap0 + Mp
    etan0 = jn0 + Mn
    
    
    uvec_pe = U[up0:up0 + Mp +2]
    Tvec_pe = U[tp0:tp0+Mp+2]
    phie_pe = U[phiep0:phiep0+Mp+2]
    phis_pe = U[phisp0:phisp0+Mp+2]
    
    uvec_ne = U[un0:un0 + Mn +2]
    Tvec_ne = U[tn0:tn0+Mn+2]
    phie_ne = U[phien0:phien0+Mn+2]
    phis_ne = U[phisn0:phisn0+Mn+2]
    
    uvec_sep = U[usep0:usep0 + Ms +2]
    Tvec_sep = U[tsep0:tsep0+Ms+2]
    phie_sep = U[phiesep0:phiesep0+Ms+2]
    
    Tvec_acc = U[ta0:ta0+Ma+2]
    Tvec_zcc = U[tz0:tz0+Mz+2]
    
    
    j_pe = U[jp0:jp0+Mp]
    eta_pe = U[etap0:etap0+Mp]
    j_ne = U[jn0:jn0+Mn]
    eta_ne = U[etan0:etan0+Mn]
    
    return uvec_pe, Tvec_pe, phie_pe, phis_pe, uvec_ne, Tvec_ne, phie_ne, phis_ne, uvec_sep, Tvec_sep, phie_sep, Tvec_acc, Tvec_zcc, j_pe, eta_pe, j_ne, eta_ne