#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#         Susi Lehtola <susi.lehtola@gmail.com>

import numpy
from pyscf import gto
from pyscf.lib import logger
from pyscf.lib import param
from pyscf.data import elements
from pyscf.scf import atom_hf, ADIIS
from pyscf.dft import rks

def get_atm_nrks(mol, atomic_configuration=elements.NRSRHFS_CONFIGURATION, xc='slater', grid=(100, 434)):
    elements = {a[0] for a in mol._atom}
    logger.info(mol, 'Spherically averaged atomic KS for %s', elements)

    atm_template = mol.copy(deep=False)
    atm_template.charge = 0
    atm_template.symmetry = False  # TODO: enable SO3 symmetry here
    atm_template.atom = atm_template._atom = []
    atm_template.cart = False  # AtomSphericAverageRKS does not support cartesian basis

    atm_scf_result = {}
    for ia, a in enumerate(mol._atom):
        element = a[0]
        if element in atm_scf_result:
            continue

        atm = atm_template
        atm._atom = [a]
        atm._atm = mol._atm[ia:ia+1]
        atm._bas = mol._bas[mol._bas[:,0] == ia].copy()
        atm._ecpbas = mol._ecpbas[mol._ecpbas[:,0] == ia].copy()
        # Point to the only atom
        atm._bas[:,0] = 0
        atm._ecpbas[:,0] = 0
        if element in mol._pseudo:
            atm._pseudo = {element: mol._pseudo.get(element)}
            raise NotImplementedError
        atm.spin = atm.nelectron % 2

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            mo_occ = mo_energy = numpy.zeros(nao)
            mo_coeff = numpy.zeros((nao,nao))
            atm_scf_result[element] = (0, mo_energy, mo_coeff, mo_occ)
        else:
            atm_ks = AtomSphericAverageRKS(atm)
            atm_ks.atomic_configuration = atomic_configuration
            atm_ks.xc = xc
            atm_ks.grids.atom_grid = grid
            atm_ks.verbose = mol.verbose
            my_diis_obj = ADIIS()
            my_diis_obj.space = 12
            atm_ks.diis = my_diis_obj
            atm_ks.run()
            atm_scf_result[element] = (atm_ks.e_tot, atm_ks.mo_energy,
                                       atm_ks.mo_coeff, atm_ks.mo_occ)
    return atm_scf_result


class AtomSphAverageRKS(rks.RKS, atom_hf.AtomSphericAverageRHF):
    def __init__(self, mol, *args, **kwargs):
        atom_hf.AtomSphericAverageRHF.__init__(self, mol)
        rks.RKS.__init__(self, mol, *args, **kwargs)

        # SAP guess is perfect for atoms
        self.init_guess = 'vsap'

    eig = atom_hf.AtomSphericAverageRHF.eig

    get_occ = atom_hf.AtomSphericAverageRHF.get_occ

    get_grad = atom_hf.AtomSphericAverageRHF.get_grad

AtomSphericAverageRKS = AtomSphAverageRKS
