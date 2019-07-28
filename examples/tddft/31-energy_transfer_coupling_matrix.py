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
t1_A = tdA.xy[state_id][0] * np.sqrt(2)
t1_B = tdB.xy[state_id][0] * np.sqrt(2)

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
print(cJ, cK)



#
# Below is an efficient implementation
#
def jk_ints(molA, molB, dm_ia, dm_jb):
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
    # The prescreen function CVHFnrs8_prescreen indexes q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dmAB, molAB._atm, molAB._bas, molAB._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None

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

    return cJ, cK

# Trasition density matrices of the two molecules
dm_A = o_A.dot(t1_A).dot(v_A.T)
dm_B = o_B.dot(t1_B).dot(v_B.T)

cJ, cK = jk_ints(molA, molB, dm_A, dm_B)
print(cJ, cK)
