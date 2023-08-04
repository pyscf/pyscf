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
#

import copy
import numpy
from pyscf import gto
from pyscf.lib import logger
from pyscf.lib import param
from pyscf.data import elements
from pyscf.scf import hf, rohf, addons


def get_atm_nrhf(mol, atomic_configuration=elements.NRSRHF_CONFIGURATION):
    elements = set([a[0] for a in mol._atom])
    logger.info(mol, 'Spherically averaged atomic HF for %s', elements)

    atm_template = copy.copy(mol)
    atm_template.charge = 0
    atm_template.symmetry = False  # TODO: enable SO3 symmetry here
    atm_template.atom = atm_template._atom = []
    atm_template.cart = False  # AtomSphAverageRHF does not support cartesian basis

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
            if atm.nelectron == 1:
                atm_hf = AtomHF1e(atm)
            else:
                atm_hf = AtomSphAverageRHF(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = mol.verbose
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)
    return atm_scf_result


class AtomSphAverageRHF(hf.RHF):
    def __init__(self, mol):
        self._eri = None
        self.atomic_configuration = elements.NRSRHF_CONFIGURATION
        hf.SCF.__init__(self, mol)

        # The default initial guess minao does not have super-heavy elements
        if mol.atom_charge(0) > 96:
            self.init_guess = '1e'

        self = self.apply(addons.remove_linear_dep_)

    def check_sanity(self):
        pass

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        hf.RHF.dump_flags(self, log)
        log.info('atom = %s', self.mol.atom_symbol(0))

    def eig(self, f, s):
        mol = self.mol
        ao_ang = _angular_momentum_for_each_ao(mol)

        nao = mol.nao
        mo_coeff = []
        mo_energy = []

        for l in range(param.L_MAX):
            degen = 2 * l + 1
            idx = numpy.where(ao_ang == l)[0]
            nao_l = len(idx)

            if nao_l > 0:
                nsh = nao_l // degen
                f_l = f[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                s_l = s[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                # Average over angular parts
                f_l = numpy.einsum('piqi->pq', f_l) / degen
                s_l = numpy.einsum('piqi->pq', s_l) / degen

                e, c = self._eigh(f_l, s_l)
                for i, ei in enumerate(e):
                    logger.debug1(self, 'l = %d  e_%d = %.9g', l, i, ei)
                mo_energy.append(numpy.repeat(e, degen))

                mo = numpy.zeros((nao, nsh, degen))
                for i in range(degen):
                    mo[idx[i::degen],:,i] = c
                mo_coeff.append(mo.reshape(nao, nao_l))

        return numpy.hstack(mo_energy), numpy.hstack(mo_coeff)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''spherically averaged fractional occupancy'''
        mol = self.mol
        symb = mol.atom_symbol(0)

        nelec_ecp = mol.atom_nelec_core(0)
        coreshl = gto.ecp.core_configuration(nelec_ecp, atom_symbol=gto.mole._std_symbol(symb))

        occ = []
        for l in range(param.L_MAX):
            n2occ, frac = frac_occ(symb, l, self.atomic_configuration)
            degen = 2 * l + 1
            idx = mol._bas[:,gto.ANG_OF] == l
            nbas_l = mol._bas[idx,gto.NCTR_OF].sum()
            if l < 4:
                n2occ -= coreshl[l]
                assert n2occ <= nbas_l

                logger.debug1(self, 'l = %d  occ = %d + %.4g', l, n2occ, frac)

                if nbas_l > 0:
                    occ_l = numpy.zeros(nbas_l)
                    occ_l[:n2occ] = 2
                    if frac > 0:
                        occ_l[n2occ] = frac
                    occ.append(numpy.repeat(occ_l, degen))
            else:
                occ.append(numpy.zeros(nbas_l * degen))

        return numpy.hstack(occ)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        return 0

    def scf(self, *args, **kwargs):
        kwargs['dump_chk'] = False
        return hf.RHF.scf(self, *args, **kwargs)

    def _finalize(self):
        if self.converged:
            logger.info(self, 'Atomic HF for atom  %s  converged. SCF energy = %.15g\n',
                        self.mol.atom_symbol(0), self.e_tot)
        else:
            logger.info(self, 'Atomic HF for atom  %s  not converged. SCF energy = %.15g\n',
                        self.mol.atom_symbol(0), self.e_tot)
        return self

AtomSphericAverageRHF = AtomSphAverageRHF


class AtomHF1e(rohf.HF1e, AtomSphAverageRHF):
    eig = AtomSphAverageRHF.eig


def frac_occ(symb, l, atomic_configuration=elements.NRSRHF_CONFIGURATION):
    nuc = gto.charge(symb)
    if l < 4 and atomic_configuration[nuc][l] > 0:
        ne = atomic_configuration[nuc][l]
        nd = (l * 2 + 1) * 2
        ndocc = ne.__floordiv__(nd)
        frac = (float(ne) / nd - ndocc) * 2
    else:
        ndocc = frac = 0
    return ndocc, frac


def _angular_momentum_for_each_ao(mol):
    ao_ang = numpy.zeros(mol.nao, dtype=int)
    ao_loc = mol.ao_loc_nr()
    for i in range(mol.nbas):
        p0, p1 = ao_loc[i], ao_loc[i+1]
        ao_ang[p0:p1] = mol.bas_angular(i)
    return ao_ang


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = None

    mol.atom = [["N", (0. , 0., .5)],
                ["N", (0. , 0.,-.5)] ]

    mol.basis = {"N": '6-31g'}
    mol.build()
    print(get_atm_nrhf(mol))
