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

'''
Non-relativistic unrestricted Hartree-Fock with point group symmetry.
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.scf import uhf
from pyscf.scf import chkfile
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mol = mf.mol
    if not mol.symmetry:
        return uhf.analyze(mf, verbose, with_meta_lowdin, **kwargs)

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        mf.dump_scf_summary(log)

        nirrep = len(mol.irrep_id)
        ovlp_ao = mf.get_ovlp()
        orbsyma, orbsymb = get_orbsym(mf.mol, mo_coeff, ovlp_ao, False)
        tot_sym = 0
        noccsa = [sum(orbsyma[mo_occ[0]>0]==ir) for ir in mol.irrep_id]
        noccsb = [sum(orbsymb[mo_occ[1]>0]==ir) for ir in mol.irrep_id]
        for i, ir in enumerate(mol.irrep_id):
            if (noccsa[i]+noccsb[i]) % 2:
                tot_sym ^= ir
        if mol.groupname in ('Dooh', 'Coov', 'SO3'):
            log.note('TODO: total wave-function symmetry for %s', mol.groupname)
        else:
            log.note('Wave-function symmetry = %s',
                     symm.irrep_id2name(mol.groupname, tot_sym))
        log.note('alpha occupancy for each irrep:  '+(' %4s'*nirrep),
                 *mol.irrep_name)
        log.note('                                 '+(' %4d'*nirrep),
                 *noccsa)
        log.note('beta  occupancy for each irrep:  '+(' %4s'*nirrep),
                 *mol.irrep_name)
        log.note('                                 '+(' %4d'*nirrep),
                 *noccsb)

        log.note('**** MO energy ****')
        irname_full = {}
        for k, ir in enumerate(mol.irrep_id):
            irname_full[ir] = mol.irrep_name[k]
        irorbcnt = {}
        for k, j in enumerate(orbsyma):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.note('alpha MO #%d (%s #%d), energy= %.15g occ= %g',
                     k+MO_BASE, irname_full[j], irorbcnt[j],
                     mo_energy[0][k], mo_occ[0][k])
        irorbcnt = {}
        for k, j in enumerate(orbsymb):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.note('beta  MO #%d (%s #%d), energy= %.15g occ= %g',
                     k+MO_BASE, irname_full[j], irorbcnt[j],
                     mo_energy[1][k], mo_occ[1][k])

    if mf.verbose >= logger.DEBUG:
        label = mol.ao_labels()
        molabel = []
        irorbcnt = {}
        for k, j in enumerate(orbsyma):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            molabel.append('#%-d(%s #%d)' %
                           (k+MO_BASE, irname_full[j], irorbcnt[j]))
        if with_meta_lowdin:
            log.debug(' ** alpha MO coefficients (expansion on meta-Lowdin AOs) **')
            orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=ovlp_ao)
            c_inv = numpy.dot(orth_coeff.conj().T, ovlp_ao)
            mo = c_inv.dot(mo_coeff[0])
        else:
            log.debug(' ** alpha MO coefficients (expansion on AOs) **')
            mo = mo_coeff[0]
        dump_mat.dump_rec(mf.stdout, mo, label, start=MO_BASE, **kwargs)

        molabel = []
        irorbcnt = {}
        for k, j in enumerate(orbsymb):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            molabel.append('#%-d(%s #%d)' %
                           (k+MO_BASE, irname_full[j], irorbcnt[j]))
        if with_meta_lowdin:
            log.debug(' ** beta MO coefficients (expansion on meta-Lowdin AOs) **')
            mo = c_inv.dot(mo_coeff[1])
        else:
            log.debug(' ** beta MO coefficients (expansion on AOs) **')
            mo = mo_coeff[1]
        dump_mat.dump_rec(mol.stdout, mo, label, molabel, start=MO_BASE, **kwargs)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        pop_and_charge = mf.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)
    else:
        pop_and_charge = mf.mulliken_pop(mol, dm, s=ovlp_ao, verbose=log)
    dip = mf.dip_moment(mol, dm, verbose=log)
    return pop_and_charge, dip

def get_irrep_nelec(mol, mo_coeff, mo_occ, s=None):
    '''Alpha/beta electron numbers for each irreducible representation.

    Args:
        mol : an instance of :class:`Mole`
            To provide irrep_id, and spin-adapted basis
        mo_occ : a list of 1D ndarray
            Regular occupancy, without grouping for irreps
        mo_coeff : a list of 2D ndarray
            Regular orbital coefficients, without grouping for irreps

    Returns:
        irrep_nelec : dict
            The number of alpha/beta electrons for each irrep {'ir_name':(int,int), ...}.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, charge=1, spin=1, verbose=0)
    >>> mf = scf.UHF(mol)
    >>> mf.scf()
    -75.623975516256721
    >>> scf.uhf_symm.get_irrep_nelec(mol, mf.mo_coeff, mf.mo_occ)
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 1), 'B2': (1, 0)}
    '''
    if getattr(mo_coeff[0], 'orbsym', None) is not None:
        orbsyma = mo_coeff[0].orbsym
    else:
        orbsyma = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                      mo_coeff[0], s, False)
    if getattr(mo_coeff[1], 'orbsym', None) is not None:
        orbsymb = mo_coeff[1].orbsym
    else:
        orbsymb = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                      mo_coeff[1], s, False)
    orbsyma = numpy.array(orbsyma)
    orbsymb = numpy.array(orbsymb)
    irrep_nelec = dict([(mol.irrep_name[k], (int(sum(mo_occ[0][orbsyma==ir])),
                                             int(sum(mo_occ[1][orbsymb==ir]))))
                        for k, ir in enumerate(mol.irrep_id)])
    return irrep_nelec

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix in occupied, virtual
    subspaces separatedly (without change occupancy).
    '''
    mol = mf.mol
    if not mol.symmetry:
        return uhf.canonicalize(mf, mo_coeff, mo_occ, fock)

    mo_occ = numpy.asarray(mo_occ)
    assert(mo_occ.ndim == 2)
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    occidxa = mo_occ[0] == 1
    occidxb = mo_occ[1] == 1
    viridxa = ~occidxa
    viridxb = ~occidxb
    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.shape)

    if (getattr(mo_coeff, 'orbsym', None) is not None or
        (getattr(mo_coeff[0], 'orbsym', None) is not None and
         getattr(mo_coeff[1], 'orbsym', None) is not None)):
        orbsyma, orbsymb = get_orbsym(mol, mo_coeff)
        def eig_(fock, mo_coeff, idx, es, cs):
            if numpy.count_nonzero(idx) > 0:
                orb = mo_coeff[:,idx]
                f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
                e, c = scipy.linalg.eigh(f1)
                es[idx] = e
                cs[:,idx] = numpy.dot(mo_coeff[:,idx], c)

        for ir in set(orbsyma):
            idx_ir = orbsyma == ir
            eig_(fock[0], mo_coeff[0], idx_ir & occidxa, mo_e[0], mo[0])
            eig_(fock[0], mo_coeff[0], idx_ir & viridxa, mo_e[0], mo[0])
        for ir in set(orbsymb):
            idx_ir = orbsymb == ir
            eig_(fock[1], mo_coeff[1], idx_ir & occidxb, mo_e[1], mo[1])
            eig_(fock[1], mo_coeff[1], idx_ir & viridxb, mo_e[1], mo[1])

    else:
        s = mf.get_ovlp()
        def eig_(fock, mo_coeff, idx, es, cs):
            if numpy.count_nonzero(idx) > 0:
                orb = mo_coeff[:,idx]
                f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
                e, c = scipy.linalg.eigh(f1)
                es[idx] = e
                c = numpy.dot(mo_coeff[:,idx], c)
                cs[:,idx] = hf_symm._symmetrize_canonicalization_(mf, e, c, s)

        eig_(fock[0], mo_coeff[0], occidxa, mo_e[0], mo[0])
        eig_(fock[0], mo_coeff[0], viridxa, mo_e[0], mo[0])
        eig_(fock[1], mo_coeff[1], occidxb, mo_e[1], mo[1])
        eig_(fock[1], mo_coeff[1], viridxb, mo_e[1], mo[1])
        orbsyma, orbsymb = get_orbsym(mol, mo, s, False)

    mo = (lib.tag_array(mo[0], orbsym=orbsyma),
          lib.tag_array(mo[1], orbsym=orbsymb))
    return mo_e, mo

def get_orbsym(mol, mo_coeff, s=None, check=False):
    if getattr(mo_coeff, 'orbsym', None) is not None:
        orbsym = numpy.asarray(mo_coeff.orbsym)
    else:
        orbsym = (hf_symm.get_orbsym(mol, mo_coeff[0], s, check),
                  hf_symm.get_orbsym(mol, mo_coeff[1], s, check))
    return orbsym

def get_wfnsym(mf, mo_coeff=None, mo_occ=None):
    orbsyma, orbsymb = mf.get_orbsym(mo_coeff)
    if mf.mol.groupname in ('SO3', 'Dooh', 'Coov'):
        if numpy.any(orbsyma > 7):
            logger.warn(mf, 'Wave-function symmetry for %s not supported. '
                        'Wfn symmetry is mapped to D2h/C2v group.',
                        mf.mol.groupname)
            orbsyma = orbsyma % 10
            orbsymb = orbsymb % 10

    if mo_occ is None:
        mo_occ = mf.mo_occ
    wfnsym = 0
    for ir in orbsyma[mo_occ[0] == 1]:
        wfnsym ^= ir
    for ir in orbsymb[mo_occ[1] == 1]:
        wfnsym ^= ir
    return wfnsym


class SymAdaptedUHF(uhf.UHF):
    __doc__ = uhf.UHF.__doc__ + '''
    Attributes for symmetry allowed UHF:
        irrep_nelec : dict
            Specify the number of alpha/beta electrons for particular irrep
            {'ir_name':(int,int), ...}.
            For the irreps not listed in these dicts, the program will choose the
            occupancy based on the orbital energies.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, charge=1, spin=1, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    -75.623975516256692
    >>> mf.get_irrep_nelec()
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 1), 'B2': (1, 0)}
    >>> mf.irrep_nelec = {'B1': (1, 0)}
    >>> mf.scf()
    -75.429189192031131
    >>> mf.get_irrep_nelec()
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 0), 'B2': (1, 1)}
    '''
    def __init__(self, mol):
        uhf.UHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nelec = {}
        self._keys = self._keys.union(['irrep_nelec'])

    def dump_flags(self, verbose=None):
        uhf.UHF.dump_flags(self, verbose)
        if self.irrep_nelec:
            logger.info(self, 'irrep_nelec %s', self.irrep_nelec)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if mol.symmetry:
            for irname in self.irrep_nelec:
                if irname not in self.mol.irrep_name:
                    logger.warn(self, 'No irrep %s', irname)

            hf_symm.check_irrep_nelec(mol, self.irrep_nelec, self.nelec)
        return uhf.UHF.build(self, mol)

    def eig(self, h, s):
        mol = self.mol
        if not mol.symmetry:
            return self._eigh(h, s)

        nirrep = mol.symm_orb.__len__()
        s = symm.symmetrize_matrix(s, mol.symm_orb)
        ha = symm.symmetrize_matrix(h[0], mol.symm_orb)
        cs = []
        es = []
        orbsym = []
        for ir in range(nirrep):
            e, c = self._eigh(ha[ir], s[ir])
            cs.append(c)
            es.append(e)
            orbsym.append([mol.irrep_id[ir]] * e.size)
        ea = numpy.hstack(es)
        ca = hf_symm.so2ao_mo_coeff(mol.symm_orb, cs)
        ca = lib.tag_array(ca, orbsym=numpy.hstack(orbsym))

        hb = symm.symmetrize_matrix(h[1], mol.symm_orb)
        cs = []
        es = []
        orbsym = []
        for ir in range(nirrep):
            e, c = self._eigh(hb[ir], s[ir])
            cs.append(c)
            es.append(e)
            orbsym.append([mol.irrep_id[ir]] * e.size)
        eb = numpy.hstack(es)
        cb = hf_symm.so2ao_mo_coeff(mol.symm_orb, cs)
        cb = lib.tag_array(cb, orbsym=numpy.hstack(orbsym))
        return (ea,eb), (ca,cb)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        g = uhf.UHF.get_grad(self, mo_coeff, mo_occ, fock)
        if self.mol.symmetry:
            occidxa = mo_occ[0] > 0
            occidxb = mo_occ[1] > 0
            viridxa = ~occidxa
            viridxb = ~occidxb
            orbsyma, orbsymb = get_orbsym(self.mol, mo_coeff)
            sym_forbida = orbsyma[viridxa].reshape(-1,1) != orbsyma[occidxa]
            sym_forbidb = orbsymb[viridxb].reshape(-1,1) != orbsymb[occidxb]
            sym_forbid = numpy.hstack((sym_forbida.ravel(),
                                       sym_forbidb.ravel()))
            g[sym_forbid] = 0
        return g

    def get_occ(self, mo_energy=None, mo_coeff=None):
        ''' We assumed mo_energy are grouped by symmetry irreps, (see function
        self.eig). The orbitals are sorted after SCF.
        '''
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        if not mol.symmetry:
            return uhf.UHF.get_occ(self, mo_energy, mo_coeff)

        orbsyma, orbsymb = get_orbsym(mol, mo_coeff)

        mo_occ = numpy.zeros_like(mo_energy)
        idx_ea_left = []
        idx_eb_left = []
        neleca_fix = nelecb_fix = 0
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idxa = numpy.where(orbsyma == ir)[0]
            ir_idxb = numpy.where(orbsymb == ir)[0]
            if irname in self.irrep_nelec:
                if isinstance(self.irrep_nelec[irname], (int, numpy.integer)):
                    nelecb = self.irrep_nelec[irname] // 2
                    neleca = self.irrep_nelec[irname] - nelecb
                else:
                    neleca, nelecb = self.irrep_nelec[irname]
                ea_idx = numpy.argsort(mo_energy[0][ir_idxa].round(9), kind='mergesort')
                eb_idx = numpy.argsort(mo_energy[1][ir_idxb].round(9), kind='mergesort')
                mo_occ[0,ir_idxa[ea_idx[:neleca]]] = 1
                mo_occ[1,ir_idxb[eb_idx[:nelecb]]] = 1
                neleca_fix += neleca
                nelecb_fix += nelecb
            else:
                idx_ea_left.append(ir_idxa)
                idx_eb_left.append(ir_idxb)

        nelec = self.nelec
        neleca_float = nelec[0] - neleca_fix
        nelecb_float = nelec[1] - nelecb_fix
        assert(neleca_float >= 0)
        assert(nelecb_float >= 0)
        if len(idx_ea_left) > 0:
            idx_ea_left = numpy.hstack(idx_ea_left)
            ea_left = mo_energy[0][idx_ea_left]
            ea_sort = numpy.argsort(ea_left.round(9), kind='mergesort')
            occ_idx = idx_ea_left[ea_sort][:neleca_float]
            mo_occ[0][occ_idx] = 1
        if len(idx_eb_left) > 0:
            idx_eb_left = numpy.hstack(idx_eb_left)
            eb_left = mo_energy[1][idx_eb_left]
            eb_sort = numpy.argsort(eb_left.round(9), kind='mergesort')
            occ_idx = idx_eb_left[eb_sort][:nelecb_float]
            mo_occ[1][occ_idx] = 1

        vir_idx = (mo_occ[0]==0)
        if self.verbose >= logger.INFO and numpy.count_nonzero(vir_idx) > 0:
            noccsa = []
            noccsb = []
            for i, ir in enumerate(mol.irrep_id):
                irname = mol.irrep_name[i]
                ir_idxa = orbsyma == ir
                ir_idxb = orbsymb == ir
                noccsa.append(numpy.count_nonzero(mo_occ[0][ir_idxa]))
                noccsb.append(numpy.count_nonzero(mo_occ[1][ir_idxb]))

            ir_id2name = dict(zip(mol.irrep_id, mol.irrep_name))
            ehomo = ehomoa = max(mo_energy[0][mo_occ[0]>0 ])
            elumo = elumoa = min(mo_energy[0][mo_occ[0]==0])
            irhomoa = ir_id2name[orbsyma[mo_energy[0] == ehomoa][0]]
            irlumoa = ir_id2name[orbsyma[mo_energy[0] == elumoa][0]]
            logger.info(self, 'alpha HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                        irhomoa, ehomoa, irlumoa, elumoa)
            if nelecb_float > 0:
                ehomob = max(mo_energy[1][mo_occ[1]>0 ])
                elumob = min(mo_energy[1][mo_occ[1]==0])
                irhomob = ir_id2name[orbsymb[mo_energy[1] == ehomob][0]]
                irlumob = ir_id2name[orbsymb[mo_energy[1] == elumob][0]]
                logger.info(self, 'beta  HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                            irhomob, ehomob, irlumob, elumob)
                ehomo = max(ehomoa,ehomob)
                elumo = min(elumoa,elumob)

            logger.debug(self, 'alpha irrep_nelec = %s', noccsa)
            logger.debug(self, 'beta  irrep_nelec = %s', noccsb)
            hf_symm._dump_mo_energy(mol, mo_energy[0], mo_occ[0], ehomo, elumo,
                                    orbsyma, 'alpha-', verbose=self.verbose)
            hf_symm._dump_mo_energy(mol, mo_energy[1], mo_occ[1], ehomo, elumo,
                                    orbsymb, 'beta-', verbose=self.verbose)

            if mo_coeff is not None and self.verbose >= logger.DEBUG:
                ovlp_ao = self.get_ovlp()
                ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                          mo_coeff[1][:,mo_occ[1]>0]), ovlp_ao)
                logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def _finalize(self):
        uhf.UHF._finalize(self)

        ea = numpy.hstack(self.mo_energy[0])
        eb = numpy.hstack(self.mo_energy[1])
        # Using mergesort because it is stable. We don't want to change the
        # ordering of the symmetry labels when two orbitals are degenerated.
        oa_sort = numpy.argsort(ea[self.mo_occ[0]>0 ].round(9), kind='mergesort')
        va_sort = numpy.argsort(ea[self.mo_occ[0]==0].round(9), kind='mergesort')
        ob_sort = numpy.argsort(eb[self.mo_occ[1]>0 ].round(9), kind='mergesort')
        vb_sort = numpy.argsort(eb[self.mo_occ[1]==0].round(9), kind='mergesort')
        idxa = numpy.arange(ea.size)
        idxa = numpy.hstack((idxa[self.mo_occ[0]> 0][oa_sort],
                             idxa[self.mo_occ[0]==0][va_sort]))
        idxb = numpy.arange(eb.size)
        idxb = numpy.hstack((idxb[self.mo_occ[1]> 0][ob_sort],
                             idxb[self.mo_occ[1]==0][vb_sort]))
        self.mo_energy = (ea[idxa], eb[idxb])
        orbsyma, orbsymb = get_orbsym(self.mol, self.mo_coeff)
        self.mo_coeff = (lib.tag_array(self.mo_coeff[0][:,idxa], orbsym=orbsyma[idxa]),
                         lib.tag_array(self.mo_coeff[1][:,idxb], orbsym=orbsymb[idxb]))
        self.mo_occ = (self.mo_occ[0][idxa], self.mo_occ[1][idxb])
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self

    @lib.with_doc(analyze.__doc__)
    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, with_meta_lowdin, **kwargs)

    @lib.with_doc(get_irrep_nelec.__doc__)
    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None, s=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if s is None: s = self.get_ovlp()
        return get_irrep_nelec(mol, mo_coeff, mo_occ, s)

    def get_orbsym(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('SCF object %s not initialized' % self)
        return get_orbsym(self.mol, mo_coeff)
    orbsym = property(get_orbsym)

    get_wfnsym = get_wfnsym
    wfnsym = property(get_wfnsym)

    canonicalize = canonicalize

UHF = SymAdaptedUHF


class HF1e(UHF):
    def scf(self, *args):
        logger.info(self, '\n')
        logger.info(self, '******** 1 electron system ********')
        self.converged = True
        h1e = self.get_hcore(self.mol)
        s1e = self.get_ovlp(self.mol)
        self.mo_energy, self.mo_coeff = self.eig([h1e]*2, s1e)
        self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
        self.e_tot = self.mo_energy[0][self.mo_occ[0]>0][0] + self.mol.energy_nuc()
        self._finalize()
        return self.e_tot


del(WITH_META_LOWDIN)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.build(
        verbose = 1,
        output = None,
        atom = [['H', (0.,0.,0.)],
                ['H', (0.,0.,1.)], ],
        basis = {'H': 'ccpvdz'},
        symmetry = True,
        charge = -1,
        spin = 1
    )

    method = UHF(mol)
    method.verbose = 5
    method.irrep_nelec['A1u'] = (1,0)
    energy = method.kernel()
    print(energy)
    method.analyze()
