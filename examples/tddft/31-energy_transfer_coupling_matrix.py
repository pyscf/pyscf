#!/usr/bin/env python

'''
The coupling matrix
        < D^* A | H | D A^* >
between donor and acceptor (* means excited state) for singlet energy transfer
(SET) and triplet energy transfer (TET) involves two types of intermolecular
2e integrals. They are the J-type integrals (D_i D_a | A_b A_j) and the K-type
integrals (D_i A_j | A_b D_a). The SET coupling corresponds to the
spin-conserved transfer process. The matrix element has two terms: J - K. The
TET coupling corresponds to the spin-flipped process and only the J integral
is required in the coupling matrix.
'''

import numpy as np
import scipy.linalg
from pyscf import gto, scf, tdscf, lib

# CIS calculations for the excited states of two molecules
molA = gto.M(atom='H 0.5 0.2 0.1; F 0 -0.1 -0.2', basis='ccpvdz')
mfA = scf.RHF(molA).run()
moA = mfA.mo_coeff
o_A = moA[:,mfA.mo_occ!=0]
v_A = moA[:,mfA.mo_occ==0]
tdA = mfA.TDA().run()

molB = gto.M(atom='C 0.9 0.2 0; O 0.1 .2 .1', basis='ccpvtz')
mfB = scf.RHF(molB).run()
moB = mfB.mo_coeff
o_B = moB[:,mfB.mo_occ!=0]
v_B = moB[:,mfB.mo_occ==0]
tdB = mfB.TDA().run()

# CIS coeffcients
state_id = 2  # The third excited state
t1_A = tdA.xy[state_id][0]
t1_B = tdB.xy[state_id][0]

# The intermolecular 2e integrals
molAB = molA + molB
naoA = molA.nao
eri = molAB.intor('int2e')
eri_AABB = eri[:naoA,:naoA,naoA:,naoA:]
eri_ABBA = eri[:naoA,naoA:,naoA:,:naoA]

# Transform integrals to MO basis
eri_iabj = lib.einsum('pqrs,pi,qa,rb,sj->iabj', eri_AABB, o_A, v_A, v_B, o_B)
eri_ijba = lib.einsum('pqrs,pi,qj,rb,sa->ijba', eri_ABBA, o_A, o_B, v_B, v_A)

# J-type coupling and K-type coupling
cJ = np.einsum('iabj,ia,jb->', eri_iabj, t1_A, t1_B)
cK = np.einsum('ijba,ia,jb->', eri_ijba, t1_A, t1_B)
print(cJ * 2 - cK)



#
# The coupling integrals can be computed more efficiently using the functions
# defined in the following
#
def jk_ints(molA, molB, dm_ia, dm_jb):
    '''Given two molecules and their (transition) density matrices, compute
    the Coulomb integrals and exchange integrals across the two molecules

    On return,
    cJ = ( ia | jb ) * dm_ia * dm_jb
    cK = ( ij | ab ) * dm_ia * dm_jb
    '''
    from pyscf.scf import jk, _vhf
    naoA = molA.nao
    naoB = molB.nao
    assert(dm_ia.shape == (naoA, naoA))
    assert(dm_jb.shape == (naoB, naoB))

    molAB = molA + molB
    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf',
                         'CVHFsetnr_direct_scf_dm')
    dmAB = scipy.linalg.block_diag(dm_ia, dm_jb)
    #### Initialization for AO-direct JK builder
    # The prescreen function CVHFnrs8_prescreen indexes q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dmAB, molAB._atm, molAB._bas, molAB._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None
    ####

    # Coulomb integrals
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0        , molA.nbas , 0        , molA.nbas,
                      molA.nbas, molAB.nbas, molA.nbas, molAB.nbas)  # AABB
        vJ = jk.get_jk(molAB, dm_jb, 'ijkl,lk->s2ij', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, dm_ia)

    # Exchange integrals
    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
        shls_slice = (0        , molA.nbas , molA.nbas, molAB.nbas,
                      molA.nbas, molAB.nbas, 0        , molA.nbas)  # ABBA
        vK = jk.get_jk(molAB, dm_jb, 'ijkl,jk->il', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s1', hermi=0)
        cK = np.einsum('ia,ia->', vK, dm_ia)

    return cJ, cK

def eval_coupling(molA, molB, dmA, dmB, dm_ia, dm_jb, xc=None):
    '''
    Evaluate the coupling term including J, K and DFT XC contributions
    Eq. (11) of JCTC 13, 3493 (2017)

        2J - c_HF*K + (1-c_HF) fxc

    dmA and dmB are ground state density matrices

    dm_ia = MO_i * MO_a  of molA
    dm_jb = MO_j * MO_b  of molB
    '''
    from pyscf import dft
    from pyscf.scf import jk, _vhf
    from pyscf.dft import numint
    molAB = molA + molB
    naoA = molA.nao
    naoB = molB.nao
    nao = naoA + naoB
    assert(dm_ia.shape == (naoA, naoA))
    assert(dm_jb.shape == (naoB, naoB))

    vhfopt = _vhf.VHFOpt(molAB, 'int2e', 'CVHFnrs8_prescreen',
                         'CVHFsetnr_direct_scf',
                         'CVHFsetnr_direct_scf_dm')
    dmAB = scipy.linalg.block_diag(dm_ia, dm_jb)
    #### Initialization for AO-direct JK builder
    # The prescreen function CVHFnrs8_prescreen indexes q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dmAB, molAB._atm, molAB._bas, molAB._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None
    ####

    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        shls_slice = (0        , molA.nbas , 0        , molA.nbas,
                      molA.nbas, molAB.nbas, molA.nbas, molAB.nbas)  # AABB
        vJ = jk.get_jk(molAB, dm_jb, 'ijkl,lk->s2ij', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s4', hermi=1)
        cJ = np.einsum('ia,ia->', vJ, dm_ia)

    with lib.temporary_env(vhfopt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
        shls_slice = (0        , molA.nbas , molA.nbas, molAB.nbas,
                      molA.nbas, molAB.nbas, 0        , molA.nbas)  # ABBA
        vK = jk.get_jk(molAB, dm_jb, 'ijkl,jk->il', shls_slice=shls_slice,
                       vhfopt=vhfopt, aosym='s1', hermi=0)
        cK = np.einsum('ia,ia->', vK, dm_ia)

    if xc is None:  # CIS coupling term
        return cJ * 2 - cK

    else:
        ni = numint.NumInt()
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc)

        cK *= hyb

        if omega > 1e-10:  # For range separated Coulomb
            with lib.temporary_env(vhfopt._this.contents,
                                   fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
                with molAB.with_range_coulomb(omega):
                    vK = jk.get_jk(molAB, dm_jb, 'ijkl,jk->il', shls_slice=shls_slice,
                                   vhfopt=vhfopt, aosym='s1', hermi=0)
                cK += np.einsum('ia,ia->', vK, dm_ia) * (alpha - hyb)

    grids = dft.Grids(molAB)
    xctype = ni._xc_type(xc)

    def make_rhoA(ao, dmA):
        return ni.eval_rho(molA, ao[...,:naoA], dmA, xctype=xctype)
    def make_rhoB(ao, dmB):
        return ni.eval_rho(molB, ao[...,naoA:], dmB, xctype=xctype)

    cXC = 0
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords in ni.block_loop(molAB, grids, nao, ao_deriv):
            # rho0 = ground state density of A + B
            rho0 = make_rhoA(ao, dmA) + make_rhoB(ao, dmB)
            fxc = ni.eval_xc(xc, rho0, 0, deriv=2)[2]
            frr = fxc[0]

            rhoA = make_rhoA(ao, dm_ia)
            rhoB = make_rhoB(ao, dm_jb)
            cXC += np.einsum('i,i,i,i->', weight, frr, rhoA, rhoB)

    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords in ni.block_loop(molAB, grids, nao, ao_deriv):
            # rho0 = ground state density of A + B
            rho0 = make_rhoA(ao, dmA) + make_rhoB(ao, dmB)
            vxc, fxc = ni.eval_xc(xc, rho0, 0, deriv=2)[1:3]
            vgamma = vxc[1]
            frho, frhogamma, fgg = fxc[:3]

            rhoA = make_rhoA(ao, dm_ia)
            rhoB = make_rhoB(ao, dm_jb)
            sigmaA = np.einsum('xi,xi->i', rho0[1:4], rhoA[1:4])
            sigmaB = np.einsum('xi,xi->i', rho0[1:4], rhoB[1:4])
            cXC += np.einsum('i,i,i,i->', weight, frho, rhoA[0], rhoB[0])
            cXC += np.einsum('i,i,i,i->', weight, frhogamma, sigmaA, rhoB[0]) * 2
            cXC += np.einsum('i,i,i,i->', weight, frhogamma, sigmaB, rhoA[0]) * 2
            cXC += np.einsum('i,i,i,i->', weight, fgg, sigmaA, sigmaB) * 4
            cXC += np.einsum('i,i,xi,xi->', weight, vgamma, rhoA[1:4], rhoB[1:4]) * 2

    return cJ * 2 - cK + cXC

dm_ia = o_A.dot(t1_A).dot(v_A.T)
dm_jb = o_B.dot(t1_B).dot(v_B.T)
cJ, cK = jk_ints(molA, molB, dm_ia, dm_jb)
print(cJ * 2 - cK)

# Evaluate the overall coupling term
print(eval_coupling(molA, molB, mfA.make_rdm1(), mfB.make_rdm1(), dm_ia, dm_jb))
print(eval_coupling(molA, molB, mfA.make_rdm1(), mfB.make_rdm1(), dm_ia, dm_jb, 'b3lyp'))
