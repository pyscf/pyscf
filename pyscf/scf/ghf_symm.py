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
Non-relativistic generalized Hartree-Fock with point group symmetry.
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf_symm
from pyscf.scf import ghf
from pyscf.scf import chkfile
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    mol = mf.mol
    if not mol.symmetry:
        return ghf.analyze(mf, verbose, **kwargs)

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        nirrep = len(mol.irrep_id)
        orbsym = get_orbsym(mf.mol, mo_coeff, ovlp_ao, False)
        wfnsym = 0
        noccs = [sum(orbsym[mo_occ>0]==ir) for ir in mol.irrep_id]
        log.note('total symmetry = %s', symm.irrep_id2name(mol.groupname, wfnsym))
        log.note('occupancy for each irrep:  ' + (' %4s'*nirrep), *mol.irrep_name)
        log.note('double occ                 ' + (' %4d'*nirrep), *noccs)
        log.note('**** MO energy ****')
        irname_full = {}
        for k,ir in enumerate(mol.irrep_id):
            irname_full[ir] = mol.irrep_name[k]
        irorbcnt = {}
        for k, j in enumerate(orbsym):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.note('MO #%d (%s #%d), energy= %.15g occ= %g',
                     k+MO_BASE, irname_full[j], irorbcnt[j], mo_energy[k],
                     mo_occ[k])

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    dip = mf.dip_moment(mol, dm, verbose=log)
    if with_meta_lowdin:
        pop_and_chg = mf.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)
    else:
        pop_and_chg = mf.mulliken_pop(mol, dm, s=ovlp_ao, verbose=log)
    return pop_and_chg, dip

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the UHF Fock matrix in occupied, virtual
    subspaces separatedly (without change occupancy).
    '''
    mol = mf.mol
    if not mol.symmetry:
        return ghf.canonicalize(mf, mo_coeff, mo_occ, fock)

    if getattr(mo_coeff, 'orbsym', None) is not None:
        return hf_symm.canonicalize(mf, mo_coeff, mo_occ, fock)
    else:
        raise NotImplementedError

class GHF(ghf.GHF):
    __doc__ = ghf.GHF.__doc__ + '''
    Attributes for symmetry allowed GHF:
        irrep_nelec : dict
            Specify the number of electrons for particular irrep
            {'ir_name':int, ...}.
            For the irreps not listed in these dicts, the program will choose the
            occupancy based on the orbital energies.
    '''
    def __init__(self, mol):
        ghf.GHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nelec = {}
        self._keys = self._keys.union(['irrep_nelec'])

    def dump_flags(self, verbose=None):
        ghf.GHF.dump_flags(self, verbose)
        if self.irrep_nelec:
            logger.info(self, 'irrep_nelec %s', self.irrep_nelec)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if mol.symmetry:
            for irname in self.irrep_nelec:
                if irname not in mol.irrep_name:
                    logger.warn(self, 'Molecule does not have irrep %s', irname)

            nelec_fix = self.irrep_nelec.values()
            if any(isinstance(x, (tuple, list)) for x in nelec_fix):
                msg =('Number of alpha/beta electrons cannot be assigned '
                      'separately in GHF.  irrep_nelec = %s' % self.irrep_nelec)
                raise ValueError(msg)
            nelec_fix = sum(nelec_fix)
            float_irname = set(mol.irrep_name) - set(self.irrep_nelec)
            if nelec_fix > mol.nelectron:
                msg =('More electrons defined by irrep_nelec than total num electrons. '
                      'mol.nelectron = %d  irrep_nelec = %s' %
                      (mol.nelectron, self.irrep_nelec))
                raise ValueError(msg)
            else:
                logger.info(mol, 'Freeze %d electrons in irreps %s',
                            nelec_fix, self.irrep_nelec.keys())

            if len(float_irname) == 0 and nelec_fix != mol.nelectron:
                msg =('Num electrons defined by irrep_nelec != total num electrons. '
                      'mol.nelectron = %d  irrep_nelec = %s' %
                      (mol.nelectron, self.irrep_nelec))
                raise ValueError(msg)
            else:
                logger.info(mol, '    %d free electrons in irreps %s',
                            mol.nelectron-nelec_fix, ' '.join(float_irname))
        return ghf.GHF.build(self, mol)

    def eig(self, h, s):
        mol = self.mol
        if not mol.symmetry:
            return self._eigh(h, s)

        nirrep = len(mol.symm_orb)
        symm_orb = [scipy.linalg.block_diag(c, c) for c in mol.symm_orb]
        s = [reduce(numpy.dot, (c.T,s,c)) for c in symm_orb]
        h = [reduce(numpy.dot, (c.T,h,c)) for c in symm_orb]
        cs = []
        es = []
        orbsym = []
        for ir in range(nirrep):
            e, c = self._eigh(h[ir], s[ir])
            cs.append(c)
            es.append(e)
            orbsym.append([mol.irrep_id[ir]] * e.size)
        e = numpy.hstack(es)
        c = hf_symm.so2ao_mo_coeff(symm_orb, cs)
        c = lib.tag_array(c, orbsym=numpy.hstack(orbsym))
        return e, c

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        g = ghf.GHF.get_grad(self, mo_coeff, mo_occ, fock)
        if self.mol.symmetry:
            occidx = mo_occ > 0
            viridx = ~occidx
            orbsym = get_orbsym(self.mol, mo_coeff)
            sym_forbid = orbsym[viridx].reshape(-1,1) != orbsym[occidx]
            g[sym_forbid.ravel()] = 0
        return g

    def get_occ(self, mo_energy=None, mo_coeff=None):
        ''' We assumed mo_energy are grouped by symmetry irreps, (see function
        self.eig). The orbitals are sorted after SCF.
        '''
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        if not mol.symmetry:
            return ghf.GHF.get_occ(self, mo_energy, mo_coeff)

        orbsym = get_orbsym(mol, mo_coeff)
        mo_occ = numpy.zeros_like(mo_energy)
        rest_idx = numpy.ones(mo_occ.size, dtype=bool)
        nelec_fix = 0
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idx = numpy.where(orbsym == ir)[0]
            if irname in self.irrep_nelec:
                n = self.irrep_nelec[irname]
                occ_sort = numpy.argsort(mo_energy[ir_idx].round(9), kind='mergesort')
                occ_idx  = ir_idx[occ_sort[:n]]
                mo_occ[occ_idx] = 1
                nelec_fix += n
                rest_idx[ir_idx] = False
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        if nelec_float > 0:
            rest_idx = numpy.where(rest_idx)[0]
            occ_sort = numpy.argsort(mo_energy[rest_idx].round(9), kind='mergesort')
            occ_idx  = rest_idx[occ_sort[:nelec_float]]
            mo_occ[occ_idx] = 1

        vir_idx = (mo_occ==0)
        if self.verbose >= logger.INFO and numpy.count_nonzero(vir_idx) > 0:
            ehomo = max(mo_energy[~vir_idx])
            elumo = min(mo_energy[ vir_idx])
            noccs = []
            for i, ir in enumerate(mol.irrep_id):
                irname = mol.irrep_name[i]
                ir_idx = (orbsym == ir)

                noccs.append(int(mo_occ[ir_idx].sum()))
                if ehomo in mo_energy[ir_idx]:
                    irhomo = irname
                if elumo in mo_energy[ir_idx]:
                    irlumo = irname
            logger.info(self, 'HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                        irhomo, ehomo, irlumo, elumo)

            logger.debug(self, 'irrep_nelec = %s', noccs)
            hf_symm._dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym,
                                    verbose=self.verbose)

            if mo_coeff is not None and self.verbose >= logger.DEBUG:
                ss, s = self.spin_square(mo_coeff[:,mo_occ>0], self.get_ovlp())
                logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def _finalize(self):
        ghf.GHF._finalize(self)

        # Using mergesort because it is stable. We don't want to change the
        # ordering of the symmetry labels when two orbitals are degenerated.
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ> 0].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        orbsym = get_orbsym(self.mol, self.mo_coeff)
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ> 0][o_sort],
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ> 0].take(o_sort, axis=1),
                                      self.mo_coeff[:,self.mo_occ==0].take(v_sort, axis=1)))
        orbsym = numpy.hstack((orbsym[self.mo_occ> 0][o_sort],
                               orbsym[self.mo_occ==0][v_sort]))
        self.mo_coeff = lib.tag_array(self.mo_coeff, orbsym=orbsym)
        nocc = len(o_sort)
        self.mo_occ[:nocc] = 1
        self.mo_occ[nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self

    def analyze(self, verbose=None, **kwargs):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose, **kwargs)

    @lib.with_doc(hf_symm.get_irrep_nelec.__doc__)
    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None, s=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if s is None: s = self.get_ovlp()
        return hf_symm.get_irrep_nelec(mol, mo_coeff, mo_occ, s)

    canonicalize = canonicalize

def get_orbsym(mol, mo_coeff, s=None, check=False):
    if mo_coeff is None:
        orbsym = numpy.hstack([[ir] * mol.symm_orb[i].shape[1]
                               for i, ir in enumerate(mol.irrep_id)])
    elif getattr(mo_coeff, 'orbsym', None) is not None:
        orbsym = mo_coeff.orbsym
    else:
        nao = mo_coeff.shape[0] // 2
        if isinstance(s, numpy.ndarray):
            assert(s.size == nao**2 or numpy.allclose(s[:nao,:nao], s[nao:,nao:]))
            s = s[:nao,:nao]
        mo_a = mo_coeff[:nao].copy()
        mo_b = mo_coeff[nao:]
        zero_alpha_idx = numpy.linalg.norm(mo_a, axis=0) < 1e-7
        mo_a[:,zero_alpha_idx] = mo_b[:,zero_alpha_idx]
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                     mo_a, s, check)
    return numpy.asarray(orbsym)


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.build(
        verbose = 1,
        output = None,
        atom = [['O', (0.,0.,0.)],
                ['O', (0.,0.,1.)], ],
        basis = {'O': 'ccpvdz'},
        symmetry = True,
        charge = -1,
        spin = 1
    )

    method = GHF(mol)
    method.verbose = 5
    method.irrep_nelec['A1u'] = 1
    energy = method.kernel()
    print(energy - -126.117033823738)
    method.canonicalize(method.mo_coeff, method.mo_occ)
    method.analyze()
