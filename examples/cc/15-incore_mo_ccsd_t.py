#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Janus J. Eriksen <januseriksen@gmail.com>
#

'''
Fully in-core MO-based CCSD with (T) correction
'''

import numpy as np
from functools import reduce

from pyscf import gto, ao2mo, scf, cc

mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = scf.RHF(mol).run()

# get core hamiltonian and AO eris
hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
eri = mol.intor('int2e_sph', aosym=4)

# freeze core
nmo = mf.mo_coeff.shape[0]
core_idx = np.array([0])
cas_idx = np.arange(1, nmo)

# extract cas integrals
core_dm = np.dot(mf.mo_coeff[:, core_idx], np.transpose(mf.mo_coeff[:, core_idx])) * 2
vj, vk = scf.hf.get_jk(mol, core_dm)
core_vhf = vj - vk * .5
h1e_cas = reduce(np.dot, (np.transpose(mf.mo_coeff[:, cas_idx]), \
                  hcore + core_vhf, mf.mo_coeff[:, cas_idx]))
h2e_cas = ao2mo.incore.full(eri, mf.mo_coeff[:, cas_idx])

# init fake mf
mol_tmp = gto.M(verbose=1)
mol_tmp.incore_anyway = True
mf_tmp = scf.RHF(mol_tmp)
mf_tmp.get_hcore = lambda *args: h1e_cas
mf_tmp._eri = h2e_cas

# init ccsd
ccsd = cc.ccsd.CCSD(mf_tmp, mo_coeff=np.eye(len(cas_idx)), mo_occ=mf.mo_occ[cas_idx])
# avoid I/O (generally requires a lot of memory)
ccsd.incore_complete = True
# avoid async function execution
ccsd.async_io = False

# calculate ccsd energy
eris = ccsd.ao2mo()
ccsd.kernel(eris=eris)
e_corr = ccsd.e_corr

print('CCSD correlation energy =', e_corr)

# calculate (t) correction
e_corr += ccsd.ccsd_t(eris=eris)

print('CCSD(T) correlation energy =', e_corr)


