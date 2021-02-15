#!/usr/bin/env python

'''
Extract CI excitation amplitudes from FCI wavefunction.

Note the basis of FCI wavefunction is the determinants which are based on
physical vacuum while the truncated CI amplitudes are based on HF vacuum.
The phase of the basis needs to be considered when converting the FCI
coefficients to CI amplitudes.
'''

import numpy
from pyscf import gto, scf, ci, fci
from pyscf.ci.cisd import tn_addrs_signs

mol = gto.Mole()
mol.atom = [
    ['O', ( 0., 0.    , 0.   )],
    ['H', ( 0., -0.857, 0.587)],
    ['H', ( 0., 0.757 , 0.687)],]
mol.basis = '321g'
mol.build()
mf = scf.RHF(mol).run()
e, fcivec = fci.FCI(mf).kernel(verbose=5)

nmo = mol.nao
nocc = mol.nelectron // 2

t1addrs, t1signs = tn_addrs_signs(nmo, nocc, 1)
t2addrs, t2signs = tn_addrs_signs(nmo, nocc, 2)
t3addrs, t3signs = tn_addrs_signs(nmo, nocc, 3)
t4addrs, t4signs = tn_addrs_signs(nmo, nocc, 4)

# CIS includes two types of amplitudes: alpha -> alpha and beta -> beta
cis_a = fcivec[t1addrs, 0] * t1signs
cis_b = fcivec[0, t1addrs] * t1signs

# CID has:
#    alpha,alpha -> alpha,alpha
#    alpha,beta  -> alpha,beta
#    beta ,beta  -> beta ,beta
# For alpha,alpha -> alpha,alpha excitations, the redundant coefficients are
# excluded. The number of coefficients is nocc*(nocc-1)//2 * nvir*(nvir-1)//2,
# which corresponds to C2_{ijab}, i > j and a > b.
cid_aa = fcivec[t2addrs, 0] * t2signs
cid_bb = fcivec[0, t2addrs] * t2signs
cid_ab = numpy.einsum('ij,i,j->ij', fcivec[t1addrs[:,None], t1addrs], t1signs, t1signs)

# CIT has:
#    alpha,alpha,alpha -> alpha,alpha,alpha
#    alpha,alpha,beta  -> alpha,alpha,beta
#    alpha,beta ,beta  -> alpha,beta ,beta
#    beta ,beta ,beta  -> beta ,beta ,beta
# For alpha,alpha,alpha -> alpha,alpha,alpha excitations, the number of
# coefficients is nocc*(nocc-1)*(nocc-2)//6 * nvir*(nvir-1)*(nvir-2)//6.
# It corresponds to C3_{ijkabc}, i > j > k and a > b > c.
cit_aaa = fcivec[t3addrs, 0] * t3signs
cit_bbb = fcivec[0, t3addrs] * t3signs
cit_aab = numpy.einsum('ij,i,j->ij', fcivec[t2addrs[:,None], t1addrs], t2signs, t1signs)
cit_abb = numpy.einsum('ij,i,j->ij', fcivec[t1addrs[:,None], t2addrs], t1signs, t2signs)

# CIQ has:
#    alpha,alpha,alpha,alpha -> alpha,alpha,alpha,alpha
#    alpha,alpha,alpha,beta  -> alpha,alpha,alpha,beta
#    alpha,alpha,beta ,beta  -> alpha,alpha,beta ,beta
#    alpha,beta ,beta ,beta  -> alpha,beta ,beta ,beta
#    beta ,beta ,beta ,beta  -> beta ,beta ,beta ,beta
# For alpha,alpha,alpha,alpha -> alpha,alpha,alpha,alpha excitations, the
# coefficients corresponds to C4_{ijklabcd}, i > j > k > l and a > b > c > d.
ciq_aaaa = fcivec[t4addrs, 0] * t4signs
ciq_bbbb = fcivec[0, t4addrs] * t4signs
ciq_aaab = numpy.einsum('ij,i,j->ij', fcivec[t3addrs[:,None], t1addrs], t3signs, t1signs)
ciq_aabb = numpy.einsum('ij,i,j->ij', fcivec[t2addrs[:,None], t2addrs], t2signs, t2signs)
ciq_abbb = numpy.einsum('ij,i,j->ij', fcivec[t1addrs[:,None], t3addrs], t1signs, t3signs)

