#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Jose Luis Casals Sainz <jluiscasalssainz@gmail.com>
#

"""Export HF/CASSCF orbitals, CI determinants and density matrices.

WFN and Molden orbital files, a determinant expansion, and a custom sparse
MO-density file are written in the current directory. A small N2 calculation
keeps the example inexpensive; increase the basis/active space for production.
"""

import numpy
from pyscf import gto, scf, mcscf, symm
from pyscf.tools import molden, wfn_format

mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g', symmetry='D2h')
mf = scf.RHF(mol).run()
occupied = mf.mo_occ > 0
with open('n2_hf.wfn', 'w') as output:
    wfn_format.write_mo(output, mol, mf.mo_coeff[:, occupied],
                        mf.mo_energy[occupied], mf.mo_occ[occupied])

mc = mcscf.CASSCF(mf, 4, 4).run()
nmo = mc.ncore + mc.ncas
mo = mc.mo_coeff[:, :nmo]
casdm1, casdm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
# Include doubly occupied core orbitals in the MO-basis density matrices.
rdm1, rdm2 = mcscf.addons._make_rdm12_on_mo(
    casdm1, casdm2, mc.ncore, mc.ncas, nmo)

# This is a custom CCIQA-style density dump, not part of the WFN format.
# Indices are one-based; the 2-RDM uses PySCF's chemists' index ordering.
# Sparse entries smaller than 1e-12 are omitted. Energy is in Hartree.
kinetic = numpy.einsum('ij,ji', mol.intor('int1e_kin'), mc.make_rdm1())
virial = (kinetic - mc.e_tot) / kinetic
with open('n2_cas.den', 'w') as output:
    output.write('CCIQA    ENERGY = %.12f THE VIRIAL(-V/T)= %.8f\n'
                 % (mc.e_tot, virial))
    output.write('La matriz D es:\n')
    for i, j in numpy.ndindex(rdm1.shape):
        output.write('%i %i %.16f\n' % (i+1, j+1, rdm1[i, j]))
    output.write('La matriz d es:\n')
    for i, j, k, l in numpy.ndindex(rdm2.shape):
        if abs(rdm2[i, j, k, l]) > 1e-12:
            output.write('%i %i %i %i %.16f\n'
                         % (i+1, j+1, k+1, l+1, rdm2[i, j, k, l]))

# Diagonalize within symmetry blocks to obtain natural orbitals. Fix their
# arbitrary signs using the largest component in the original MO basis.
orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo)
neg_occ, rotation = symm.eigh(-rdm1, orbsym)
for i, k in enumerate(numpy.argmax(abs(rotation), axis=0)):
    if rotation[k, i] < 0:
        rotation[:, i] *= -1
natural_mo, natural_occ = mo @ rotation, -neg_occ

# The CI expansion refers to the original optimized CASSCF orbitals, not
# the natural orbitals: rotating the orbitals also requires rotating CI.
with open('n2_cas.det', 'w') as output:
    wfn_format.write_ci(output, mc.ci, mc.ncas, mc.nelecas, mc.ncore)
with open('n2_cas_orbitals.wfn', 'w') as output:
    wfn_format.write_mo(output, mol, mo)
with open('n2_cas_natural.wfn', 'w') as output:
    wfn_format.write_mo(output, mol, natural_mo, mo_occ=natural_occ)
# One WFN document per file. Natural occupations reproduce the 1-RDM;
# the determinant file carries the correlated many-electron wavefunction.
molden.from_mo(mol, 'n2_cas.molden', mo, occ=numpy.diag(rdm1))
molden.from_mo(mol, 'n2_cas_natural.molden', natural_mo, occ=natural_occ)
print('Wrote N2 HF/CASSCF WFN, determinant, density and Molden files')
