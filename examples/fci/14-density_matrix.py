#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Compute FCI 1,2,3,4-particle density matrices
'''

#
# Note: Environment variable LD_PRELOAD=...libmkl_def.so may cause this script
# crashing
#

import numpy
from pyscf import gto, scf, fci

mol = gto.Mole()
mol.build(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '6-31g',
    spin = 2,
)
myhf = scf.RHF(mol)
myhf.kernel()

#
# First create FCI solver with function fci.FCI and solve the FCI problem
#
cisolver = fci.FCI(mol, myhf.mo_coeff)
e, fcivec = cisolver.kernel()

#
# Spin-traced 1-particle density matrix
#
norb = myhf.mo_coeff.shape[1]
# 6 alpha electrons, 4 beta electrons because spin = nelec_a-nelec_b = 2
nelec_a = 6
nelec_b = 4
dm1 = cisolver.make_rdm1(fcivec, norb, (nelec_a,nelec_b))

#
# alpha and beta 1-particle density matrices
#
dm1a, dm1b = cisolver.make_rdm1s(fcivec, norb, (nelec_a,nelec_b))
assert(numpy.allclose(dm1a+dm1b, dm1))

#
# Spin-traced 1 and 2-particle density matrices
#
dm1, dm2 = cisolver.make_rdm12(fcivec, norb, (nelec_a,nelec_b))

#
# alpha and beta 1-particle density matrices
# For 2-particle density matrix,
# dm2aa corresponds to alpha spin for both 1st electron and 2nd electron
# dm2ab corresponds to alpha spin for 1st electron and beta spin for 2nd electron
# dm2bb corresponds to beta spin for both 1st electron and 2nd electron
#
(dm1a, dm1b), (dm2aa,dm2ab,dm2bb) = cisolver.make_rdm12s(fcivec, norb, (nelec_a,nelec_b))
assert(numpy.allclose(dm2aa+dm2ab+dm2ab.transpose(2,3,0,1)+dm2bb, dm2))


#########################################
#
# Transition density matrices
#
#########################################

#
# First generate two CI vectors, of which the transition density matrices will
# be computed
#
cisolver.nroots = 2
(e0,e1), (fcivec0,fcivec1) = cisolver.kernel()

#
# Spin-traced 1-particle transition density matrix
# <0| p^+ q |1>
#
norb = myhf.mo_coeff.shape[1]
nelec_a = 6
nelec_b = 4
dm1 = cisolver.trans_rdm1(fcivec0, fcivec1, norb, (nelec_a,nelec_b))

#
# alpha and beta 1-particle transition density matrices
#
dm1a, dm1b = cisolver.trans_rdm1s(fcivec0, fcivec1, norb, (nelec_a,nelec_b))
assert(numpy.allclose(dm1a+dm1b, dm1))

#
# Spin-traced 1 and 2-particle transition density matrices
#
dm1, dm2 = cisolver.trans_rdm12(fcivec0, fcivec1, norb, (nelec_a,nelec_b))

#
# alpha and beta 1-particle transition density matrices
# For 2-particle density matrix,
# dm2aa corresponds to alpha spin for both 1st electron and 2nd electron
# dm2ab corresponds to alpha spin for 1st electron and beta spin for 2nd electron
# dm2ba corresponds to beta spin for 1st electron and alpha spin for 2nd electron
# dm2bb corresponds to beta spin for both 1st electron and 2nd electron
#
(dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb) = \
        cisolver.trans_rdm12s(fcivec0, fcivec1, norb, (nelec_a,nelec_b))
assert(numpy.allclose(dm2aa+dm2ab+dm2ba+dm2bb, dm2))


#########################################
#
# 3 and 4-particle density matrices
#
#########################################

#
# Spin-traced 3-particle density matrix
# 1,2,3-pdm can be computed together.
# Note make_dm123 computes  dm3[p,q,r,s,t,u] = <p^+ q r^+ s t^+ u>  which is
# NOT the 3-particle density matrices.  Funciton reorder_dm123 transforms it
# to the true 3-particle DM  dm3[p,q,r,s,t,u] = <p^+ r^+ t^+ u s q> (as well
# as the 2-particle DM)
#
dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', fcivec0, fcivec0, norb,
                                   (nelec_a,nelec_b))
dm1, dm2, dm3 = fci.rdm.reorder_dm123(dm1, dm2, dm3)

#
# Spin-separated 3-particle density matrix
#

(dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb) = \
        fci.direct_spin1.make_rdm123s(fcivec0, norb, (nelec_a,nelec_b))

assert(numpy.allclose(dm1a+dm1b, dm1))
assert(numpy.allclose(dm2aa+dm2bb+dm2ab+dm2ab.transpose(2,3,0,1), dm2))
assert(numpy.allclose(dm3aaa+dm3bbb+dm3aab+dm3aab.transpose(0,1,4,5,2,3)+\
dm3aab.transpose(4,5,0,1,2,3)+dm3abb+dm3abb.transpose(2,3,0,1,4,5)+dm3abb.transpose(2,3,4,5,0,1), dm3))

#
# Spin-traced 3-particle transition density matrix
#
dm1, dm2, dm3 = fci.rdm.make_dm123('FCI3pdm_kern_sf', fcivec0, fcivec1, norb,
                                   (nelec_a,nelec_b))
dm1, dm2, dm3 = fci.rdm.reorder_dm123(dm1, dm2, dm3)

#
# NOTE computing 4-pdm is very slow
#
#
# Spin-traced 4-particle density matrix
# Note make_dm1234 computes  dm4[p,q,r,s,t,u,v,w] = <p^+ q r^+ s t^+ u v^+ w>  which is
# NOT the 4-particle density matrices.  Funciton reorder_dm1234 transforms it
# to the true 4-particle DM  dm4[p,q,r,s,t,u,v,w] = <p^+ r^+ t^+ v^+ w u s q>
# (as well as the 2-particle and 3-particle DMs)
#
dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', fcivec0, fcivec0, norb,
                                         (nelec_a,nelec_b))
dm1, dm2, dm3, dm4 = fci.rdm.reorder_dm1234(dm1, dm2, dm3, dm4)

#
# Spin-separated 4-particle density matrix
#

(dm1a, dm1b), (dm2aa, dm2ab, dm2bb), (dm3aaa, dm3aab, dm3abb, dm3bbb), (dm4aaaa, dm4aaab, dm4aabb, dm4abbb, dm4bbbb) = (
    fci.direct_spin1.make_rdm1234s(fcivec0, norb, (nelec_a, nelec_b))
)
assert(numpy.allclose(dm1a+dm1b, dm1))
assert(numpy.allclose(dm2aa+dm2bb+dm2ab+dm2ab.transpose(2,3,0,1), dm2))
assert(numpy.allclose(dm3aaa+dm3bbb+dm3aab+dm3aab.transpose(0,1,4,5,2,3)+\
dm3aab.transpose(4,5,0,1,2,3)+dm3abb+dm3abb.transpose(2,3,0,1,4,5)+dm3abb.transpose(2,3,4,5,0,1), dm3))
assert(numpy.allclose(
    dm4aaaa
    + dm4bbbb
    + dm4aaab
    + dm4aaab.transpose(0, 1, 2, 3, 6, 7, 4, 5)
    + dm4aaab.transpose(0, 1, 6, 7, 2, 3, 4, 5)
    + dm4aaab.transpose(6, 7, 0, 1, 2, 3, 4, 5)
    + dm4aabb
    + dm4aabb.transpose(0, 1, 4, 5, 2, 3, 6, 7)
    + dm4aabb.transpose(4, 5, 0, 1, 2, 3, 6, 7)
    + dm4aabb.transpose(0, 1, 4, 5, 6, 7, 2, 3)
    + dm4aabb.transpose(4, 5, 0, 1, 6, 7, 2, 3)
    + dm4aabb.transpose(4, 5, 6, 7, 0, 1, 2, 3)
    + dm4abbb
    + dm4abbb.transpose(2, 3, 0, 1, 4, 5, 6, 7)
    + dm4abbb.transpose(2, 3, 4, 5, 0, 1, 6, 7)
    + dm4abbb.transpose(2, 3, 4, 5, 6, 7, 0, 1),
    dm4,
))

#
# Spin-traced 4-particle transition density matrix
#
dm1, dm2, dm3, dm4 = fci.rdm.make_dm1234('FCI4pdm_kern_sf', fcivec0, fcivec1, norb,
                                         (nelec_a,nelec_b))
dm1, dm2, dm3, dm4 = fci.rdm.reorder_dm1234(dm1, dm2, dm3, dm4)
