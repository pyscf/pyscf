#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Transition density matrix between CASCI ground and excited state
'''

import numpy
from pyscf import gto, scf, mcscf

mol = gto.M(
    atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],],
    basis = '6-31g',
    symmetry = 1)

mf = scf.RHF(mol).run()

ncas = 4
ne = 4
mc = mcscf.CASCI(mf, ncas, ne)
mc.fcisolver.nroots = 3
mc.kernel()

# CASCI 1-particle transition density matrix between ground state and 2nd
# excited state
t_dm1 = mc.fcisolver.trans_rdm1(mc.ci[0], mc.ci[2], ncas, ne)

# Transform to AO representation
mo_cas = mf.mo_coeff[:,3:7]
t_dm1 = numpy.einsum('pi,ij,qj->pq', mo_cas, t_dm1, mo_cas)

charge_center = (numpy.einsum('z,zx->x', mol.atom_charges(), mol.atom_coords())
                 / mol.atom_charges().sum())
with mol.with_common_origin(charge_center):
    t_dip = numpy.einsum('xij,ji->x', mol.intor('int1e_r'), t_dm1)
print(t_dip)
