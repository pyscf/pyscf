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
#         Elvira R. Sayfutyarova
#

'''
Automated construction of molecular active spaces from atomic valence orbitals.
Ref. arXiv:1701.07862 [physics.chem-ph]
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf.lib import logger
from pyscf import __config__

THRESHOLD = getattr(__config__, 'mcscf_avas_threshold', 0.2)
MINAO = getattr(__config__, 'mcscf_avas_minao', 'minao')
WITH_IAO = getattr(__config__, 'mcscf_avas_with_iao', False)
OPENSHELL_OPTION = getattr(__config__, 'mcscf_avas_openshell_option', 2)
CANONICALIZE = getattr(__config__, 'mcscf_avas_canonicalize', True)


def kernel(mf, aolabels, threshold=THRESHOLD, minao=MINAO, with_iao=WITH_IAO,
           openshell_option=OPENSHELL_OPTION, canonicalize=CANONICALIZE,
           ncore=0, verbose=None):
    '''AVAS method to construct mcscf active space.
    Ref. arXiv:1701.07862 [physics.chem-ph]

    Args:
        mf : an :class:`SCF` object

        aolabels : string or a list of strings
            AO labels for AO active space

    Kwargs:
        threshold : float
            Tructing threshold of the AO-projector above which AOs are kept in
            the active space.
        minao : str
            A reference AOs for AVAS.
        with_iao : bool
            Whether to use IAO localization to construct the reference active AOs.
        openshell_option : int
            How to handle singly-occupied orbitals in the active space. The
            singly-occupied orbitals are projected as part of alpha orbitals
            if openshell_option=2, or completely kept in active space if
            openshell_option=3.  See Section III.E option 2 or 3 of the
            reference paper for more details.
        canonicalize : bool
            Orbitals defined in AVAS method are local orbitals.  Symmetrizing
            the core, active and virtual space.
        ncore : integer
            Number of core orbitals to be excluded from the AVAS method.

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess-for-CASCI/CASSCF

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.mcscf import avas
    >>> mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.6', basis='ccpvtz')
    >>> mf = scf.RHF(mol).run()
    >>> ncas, nelecas, mo = avas.avas(mf, ['Cr 3d', 'Cr 4s'])
    >>> mc = mcscf.CASSCF(mf, ncas, nelecas).run(mo)
    '''
    avas_obj = AVAS(mf, aolabels, threshold, minao, with_iao,
                    openshell_option, canonicalize, ncore, verbose)
    return avas_obj.kernel()

def _kernel(avas_obj):
    mf = avas_obj._scf
    mol = mf.mol
    log = logger.new_logger(avas_obj)
    log.info('\n** AVAS **')

    assert avas_obj.openshell_option != 1

    if isinstance(mf, scf.uhf.UHF):
        log.note('UHF/UKS object is found.  AVAS takes alpha orbitals only')
        mo_coeff = mf.mo_coeff[0]
        mo_occ = mf.mo_occ[0]
        mo_energy = mf.mo_energy[0]
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy

    ncore = avas_obj.ncore
    nocc = numpy.count_nonzero(mo_occ != 0)
    ovlp = mol.intor_symmetric('int1e_ovlp')
    log.info('  Total number of HF MOs  is equal to    %d' ,mo_coeff.shape[1])
    log.info('  Number of occupied HF MOs is equal to  %d', nocc)

    mol = mf.mol
    pmol = mol.copy()
    pmol.atom = mol._atom
    pmol.unit = 'B'
    pmol.symmetry = False
    pmol.basis = avas_obj.minao
    pmol.build(False, False)

    baslst = pmol.search_ao_label(avas_obj.aolabels)
    log.info('reference AO indices for %s %s:\n %s',
             avas_obj.minao, avas_obj.aolabels, baslst)

    if avas_obj.with_iao:
        from pyscf.lo import iao
        c = iao.iao(mol, mo_coeff[:,ncore:nocc], avas_obj.minao)[:,baslst]
        s2 = reduce(numpy.dot, (c.T, ovlp, c))
        s21 = reduce(numpy.dot, (c.T, ovlp, mo_coeff[:, ncore:]))
    else:
        s2 = pmol.intor_symmetric('int1e_ovlp')[baslst][:,baslst]
        s21 = gto.intor_cross('int1e_ovlp', pmol, mol)[baslst]
        s21 = numpy.dot(s21, mo_coeff[:, ncore:])
    sa = s21.T.dot(scipy.linalg.solve(s2, s21, assume_a='pos'))

    threshold = avas_obj.threshold
    if avas_obj.openshell_option == 2:
        wocc, u = numpy.linalg.eigh(sa[:(nocc-ncore), :(nocc-ncore)])
        log.info('Option 2: threshold %s', threshold)
        ncas_occ = (wocc > threshold).sum()
        nelecas = (mol.nelectron - ncore * 2) - (wocc < threshold).sum() * 2
        mocore = mo_coeff[:,ncore:nocc].dot(u[:,wocc<threshold])
        mocas = mo_coeff[:,ncore:nocc].dot(u[:,wocc>=threshold])

        wvir, u = numpy.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
        ncas_vir = (wvir > threshold).sum()
        mocas = numpy.hstack((mocas,
                              mo_coeff[:,nocc:].dot(u[:,wvir>=threshold])))
        movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
        ncas = mocas.shape[1]

        occ_weights = numpy.hstack([wocc[wocc<threshold], wocc[wocc>=threshold]])
        vir_weights = numpy.hstack([wvir[wvir>=threshold], wvir[wvir<threshold]])

    elif avas_obj.openshell_option == 3:
        docc = nocc - mol.spin
        wocc, u = numpy.linalg.eigh(sa[:(docc-ncore),:(docc-ncore)])
        log.info('Option 3: threshold %s, num open shell %d', threshold, mol.spin)
        ncas_occ = (wocc > threshold).sum()
        nelecas = (mol.nelectron - ncore * 2) - (wocc < threshold).sum() * 2
        mocore = mo_coeff[:,ncore:docc].dot(u[:,wocc<threshold])
        mocas = mo_coeff[:,ncore:docc].dot(u[:,wocc>=threshold])

        wvir, u = numpy.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
        ncas_vir = (wvir > threshold).sum()
        mocas = numpy.hstack((mocas,
                              mo_coeff[:,docc:nocc],
                              mo_coeff[:,nocc:].dot(u[:,wvir>=threshold])))
        movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
        ncas = mocas.shape[1]

        occ_weights = numpy.hstack([wocc[wocc<threshold], numpy.ones(nocc-docc),
                                    wocc[wocc>=threshold]])
        vir_weights = numpy.hstack([wvir[wvir>=threshold], wvir[wvir<threshold]])
    else:
        raise RuntimeError(f'Unknown option openshell_option {avas_obj.openshell_option}')

    log.debug('projected occ eig %s', occ_weights)
    log.debug('projected vir eig %s', vir_weights)
    log.info('Active from occupied = %d , eig %s', ncas_occ, occ_weights[occ_weights>=threshold])
    log.info('Inactive from occupied = %d', mocore.shape[1])
    log.info('Active from unoccupied = %d , eig %s', ncas_vir, vir_weights[vir_weights>=threshold])
    log.info('Inactive from unoccupied = %d', movir.shape[1])
    log.info('Dimensions of active %d', ncas)
    nalpha = (nelecas + mol.spin) // 2
    nbeta = nelecas - nalpha
    log.info('# of alpha electrons %d', nalpha)
    log.info('# of beta electrons %d', nbeta)

    mofreeze = mo_coeff[:,:ncore]
    if avas_obj.canonicalize:
        from pyscf.mcscf import dmet_cas

        def trans(c):
            if c.shape[1] == 0:
                return c
            else:
                csc = reduce(numpy.dot, (c.T, ovlp, mo_coeff))
                fock = numpy.dot(csc*mo_energy, csc.T)
                e, u = scipy.linalg.eigh(fock)
                return dmet_cas.symmetrize(mol, e, numpy.dot(c, u), ovlp, log)
        if ncore > 0:
            mofreeze = trans(mofreeze)
        mocore = trans(mocore)
        mocas = trans(mocas)
        movir = trans(movir)
    mo = numpy.hstack((mofreeze, mocore, mocas, movir))
    return ncas, nelecas, mo, occ_weights, vir_weights
avas = kernel

@lib.with_doc(kernel.__doc__)
class AVAS(lib.StreamObject):
    def __init__(self, mf, aolabels, threshold=THRESHOLD, minao=MINAO,
                 with_iao=WITH_IAO, openshell_option=OPENSHELL_OPTION,
                 canonicalize=CANONICALIZE, ncore=0, verbose=None):
        self._scf = mf
        self.aolabels = aolabels
        self.threshold = threshold
        self.minao = minao
        self.with_iao = with_iao
        self.openshell_option = openshell_option
        self.canonicalize = canonicalize
        self.ncore = ncore
        self.stdout = mf.stdout
        self.verbose = verbose or mf.verbose

##################################################
# don't modify the following attributes, they are not input options
        self.ncas = None
        self.nelecas = None
        # Orbitals of entire space, including cores, actives, externals
        self.mo_coeff = None
        # occ_weights and vir_weights to filter active and inactive orbitals
        self.occ_weights = None
        self.vir_weights = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** AVAS flags ********')
        log.info('aolabels = %s', self.aolabels)
        log.info('ncore = %s', self.ncore)
        log.info('minao = %s', self.minao)
        log.info('threshold = %s', self.threshold)
        log.info('with_iao = %s', self.with_iao)
        log.info('openshell_option = %s', self.openshell_option)
        log.info('canonicalize = %s', self.canonicalize)
        return self

    def kernel(self):
        self.dump_flags()
        self.ncas, self.nelecas, self.mo_coeff, \
                self.occ_weights, self.vir_weights = _kernel(self)
        return self.ncas, self.nelecas, self.mo_coeff
