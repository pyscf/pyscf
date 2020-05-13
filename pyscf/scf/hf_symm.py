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
Non-relativistic restricted Hartree-Fock with point group symmetry.

The symmetry are not handled in a separate data structure.  Note that during
the SCF iteration,  the orbitals are grouped in terms of symmetry irreps.
But the orbitals in the result are sorted based on the orbital energies.
Function symm.label_orb_symm can be used to detect the symmetry of the
molecular orbitals.
'''

from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import rohf
from pyscf.scf import chkfile
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
MO_BASE = getattr(__config__, 'MO_BASE', 1)


# mo_energy, mo_coeff, mo_occ are all in nosymm representation

def analyze(mf, verbose=logger.DEBUG, with_meta_lowdin=WITH_META_LOWDIN,
            **kwargs):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Occupancy for each irreps; Mulliken population analysis
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mol = mf.mol
    if not mol.symmetry:
        return hf.analyze(mf, verbose, with_meta_lowdin, **kwargs)

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    ovlp_ao = mf.get_ovlp()
    log = logger.new_logger(mf, verbose)
    if log.verbose >= logger.NOTE:
        mf.dump_scf_summary(log)

        nirrep = len(mol.irrep_id)
        orbsym = get_orbsym(mf.mol, mo_coeff, ovlp_ao, False)
        wfnsym = 0
        noccs = [sum(orbsym[mo_occ>0]==ir) for ir in mol.irrep_id]
        if mol.groupname in ('SO3', 'Dooh', 'Coov'):
            log.note('TODO: total wave-function symmetry for %s', mol.groupname)
        else:
            log.note('Wave-function symmetry = %s',
                     symm.irrep_id2name(mol.groupname, wfnsym))
        log.note('occupancy for each irrep:  ' + (' %4s'*nirrep), *mol.irrep_name)
        log.note('                           ' + (' %4d'*nirrep), *noccs)
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
                     k+MO_BASE, irname_full[j], irorbcnt[j],
                     mo_energy[k], mo_occ[k])

    if log.verbose >= logger.DEBUG:
        label = mol.ao_labels()
        molabel = []
        irorbcnt = {}
        for k, j in enumerate(orbsym):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            molabel.append('#%-d(%s #%d)' %
                           (k+MO_BASE, irname_full[j], irorbcnt[j]))
        if with_meta_lowdin:
            log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) **')
            orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=ovlp_ao)
            c = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mo_coeff))
        else:
            log.debug(' ** MO coefficients (expansion on AOs) **')
            c = mo_coeff
        dump_mat.dump_rec(mf.stdout, c, label, molabel, start=MO_BASE, **kwargs)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    if with_meta_lowdin:
        pop_and_charge = mf.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)
    else:
        pop_and_charge = mf.mulliken_pop(mol, dm, s=ovlp_ao, verbose=log)
    dip = mf.dip_moment(mol, dm, verbose=log)
    return pop_and_charge, dip

def get_irrep_nelec(mol, mo_coeff, mo_occ, s=None):
    '''Electron numbers for each irreducible representation.

    Args:
        mol : an instance of :class:`Mole`
            To provide irrep_id, and spin-adapted basis
        mo_coeff : 2D ndarray
            Regular orbital coefficients, without grouping for irreps
        mo_occ : 1D ndarray
            Regular occupancy, without grouping for irreps

    Returns:
        irrep_nelec : dict
            The number of electrons for each irrep {'ir_name':int,...}.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    -76.016789472074251
    >>> scf.hf_symm.get_irrep_nelec(mol, mf.mo_coeff, mf.mo_occ)
    {'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2}
    '''
    orbsym = get_orbsym(mol, mo_coeff, s, False)
    irrep_nelec = dict([(mol.irrep_name[k], int(sum(mo_occ[orbsym==ir])))
                        for k, ir in enumerate(mol.irrep_id)])
    return irrep_nelec

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix in occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    mol = mf.mol
    if not mol.symmetry:
        return hf.canonicalize(mf, mo_coeff, mo_occ, fock)

    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_hcore() + mf.get_veff(mf.mol, dm)
    coreidx = mo_occ == 2
    viridx = mo_occ == 0
    openidx = ~(coreidx | viridx)
    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.size)

    if getattr(mo_coeff, 'orbsym', None) is not None:
        orbsym = mo_coeff.orbsym
        irreps = set(orbsym)
        for ir in irreps:
            idx0 = orbsym == ir
            for idx1 in (coreidx, openidx, viridx):
                idx = idx0 & idx1
                if numpy.count_nonzero(idx) > 0:
                    orb = mo_coeff[:,idx]
                    f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
                    e, c = scipy.linalg.eigh(f1)
                    mo[:,idx] = numpy.dot(mo_coeff[:,idx], c)
                    mo_e[idx] = e
    else:
        s = mf.get_ovlp()
        for idx in (coreidx, openidx, viridx):
            if numpy.count_nonzero(idx) > 0:
                orb = mo_coeff[:,idx]
                f1 = reduce(numpy.dot, (orb.conj().T, fock, orb))
                e, c = scipy.linalg.eigh(f1)
                c = numpy.dot(mo_coeff[:,idx], c)
                mo[:,idx] = _symmetrize_canonicalization_(mf, e, c, s)
                mo_e[idx] = e
        orbsym = get_orbsym(mol, mo, s, False)

    mo = lib.tag_array(mo, orbsym=orbsym)
    return mo_e, mo

def _symmetrize_canonicalization_(mf, mo_energy, mo_coeff, s):
    '''Restore symmetry for canonicalized orbitals
    '''
    def search_for_degeneracy(mo_energy):
        idx = numpy.where(abs(mo_energy[1:] - mo_energy[:-1]) < 1e-6)[0]
        return numpy.unique(numpy.hstack((idx, idx+1)))

    mol = mf.mol
    degidx = search_for_degeneracy(mo_energy)
    logger.debug1(mf, 'degidx %s', degidx)
    if degidx.size > 0:
        esub = mo_energy[degidx]
        csub = mo_coeff[:,degidx]
        scsub = numpy.dot(s, csub)
        emin = abs(esub).min() * .5
        es = []
        cs = []
        for i,ir in enumerate(mol.irrep_id):
            so = mol.symm_orb[i]
            sosc = numpy.dot(so.conj().T, scsub)
            s_ir = reduce(numpy.dot, (so.conj().T, s, so))
            fock_ir = numpy.dot(sosc*esub, sosc.conj().T)
            mo_energy, u = mf._eigh(fock_ir, s_ir)
            idx = abs(mo_energy) > emin
            es.append(mo_energy[idx])
            cs.append(numpy.dot(mol.symm_orb[i], u[:,idx]))
        es = numpy.hstack(es).round(7)
        idx = numpy.argsort(es, kind='mergesort')
        assert(numpy.allclose(es[idx], esub.round(7)))
        mo_coeff[:,degidx] = numpy.hstack(cs)[:,idx]
    return mo_coeff

def so2ao_mo_coeff(so, irrep_mo_coeff):
    '''Transfer the basis of MO coefficients, from spin-adapted basis to AO basis
    '''
    return numpy.hstack([numpy.dot(so[ir],irrep_mo_coeff[ir]) \
                         for ir in range(so.__len__())])

def check_irrep_nelec(mol, irrep_nelec, nelec):
    for irname in irrep_nelec:
        if irname not in mol.irrep_name:
            logger.warn(mol, 'Molecule does not have irrep %s', irname)

    float_irname = []
    fix_na = 0
    fix_nb = 0
    free_irrep_norbs = []
    for i, irname in enumerate(mol.irrep_name):
        if irname in irrep_nelec:
            if isinstance(irrep_nelec[irname], (int, numpy.integer)):
                nelecb = irrep_nelec[irname] // 2
                neleca = irrep_nelec[irname] - nelecb
            else:
                neleca, nelecb = irrep_nelec[irname]
            norb = mol.symm_orb[i].shape[1]
            if neleca > norb or nelecb > norb:
                msg =('More electrons than orbitals for irrep %s '
                      'nelec = %d + %d, norb = %d' %
                      (irname, neleca,nelecb, norb))
                raise ValueError(msg)
            fix_na += neleca
            fix_nb += nelecb
        else:
            float_irname.append(irname)
            free_irrep_norbs.append(mol.symm_orb[i].shape[1])

    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    fix_ne = fix_na + fix_nb
    float_neleca = neleca - fix_na
    float_nelecb = nelecb - fix_nb
    free_norb = sum(free_irrep_norbs)
    if ((fix_na > neleca) or (fix_nb > nelecb) or
        (fix_na+nelecb > mol.nelectron) or
        (fix_nb+neleca > mol.nelectron)):
        msg =('More electrons defined by irrep_nelec than total num electrons. '
              'mol.nelectron = %d  irrep_nelec = %s' %
              (mol.nelectron, irrep_nelec))
        raise ValueError(msg)
    else:
        logger.info(mol, 'Freeze %d electrons in irreps %s',
                    fix_ne, list(irrep_nelec.keys()))

    if len(set(float_irname)) == 0 and fix_ne != mol.nelectron:
        msg =('Num electrons defined by irrep_nelec != total num electrons. '
              'mol.nelectron = %d  irrep_nelec = %s' %
              (mol.nelectron, irrep_nelec))
        raise ValueError(msg)
    elif float_neleca > free_norb or float_nelecb > free_norb:
        raise ValueError('Not enough orbitals for (%d, %d) electrons in irreps %s '
                         '(irrep_norb: %s)' %
                         (float_neleca, float_nelecb, ' '.join(float_irname), free_norb))
    else:
        logger.info(mol, '    %d free electrons in irreps %s',
                    mol.nelectron-fix_ne, ' '.join(float_irname))
    return fix_na, fix_nb, float_irname

#TODO: force E1gx/E1gy ... use the same coefficients
def eig(mf, h, s):
    '''Solve generalized eigenvalue problem, for each irrep.  The
    eigenvalues and eigenvectors are not sorted to ascending order.
    Instead, they are grouped based on irreps.
    '''
    mol = mf.mol
    if not mol.symmetry:
        return mf._eigh(h, s)

    nirrep = mol.symm_orb.__len__()
    h = symm.symmetrize_matrix(h, mol.symm_orb)
    s = symm.symmetrize_matrix(s, mol.symm_orb)
    cs = []
    es = []
    orbsym = []
    for ir in range(nirrep):
        e, c = mf._eigh(h[ir], s[ir])
        cs.append(c)
        es.append(e)
        orbsym.append([mol.irrep_id[ir]] * e.size)
    e = numpy.hstack(es)
    c = so2ao_mo_coeff(mol.symm_orb, cs)
    c = lib.tag_array(c, orbsym=numpy.hstack(orbsym))
    return e, c

def get_orbsym(mol, mo_coeff, s=None, check=False):
    if mo_coeff is None:
        orbsym = numpy.hstack([[ir] * mol.symm_orb[i].shape[1]
                               for i, ir in enumerate(mol.irrep_id)])
    elif getattr(mo_coeff, 'orbsym', None) is not None:
        orbsym = mo_coeff.orbsym
    else:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                     mo_coeff, s, check)
    return numpy.asarray(orbsym)

def get_wfnsym(mf, mo_coeff=None, mo_occ=None):
    orbsym = mf.get_orbsym(mo_coeff)
    if mf.mol.groupname in ('SO3', 'Dooh', 'Coov'):
        if numpy.any(orbsym > 7):
            logger.warn(mf, 'Wave-function symmetry for %s not supported. '
                        'Wfn symmetry is mapped to D2h/C2v group.',
                        mf.mol.groupname)
            orbsym = orbsym % 10

    if mo_occ is None:
        mo_occ = mf.mo_occ
    wfnsym = 0
    for ir in orbsym[mo_occ == 1]:
        wfnsym ^= ir
    return wfnsym


class SymAdaptedRHF(hf.RHF):
    __doc__ = hf.SCF.__doc__ + '''
    Attributes for symmetry allowed RHF:
        irrep_nelec : dict
            Specify the number of electrons for particular irrep {'ir_name':int,...}.
            For the irreps not listed in this dict, the program will choose the
            occupancy based on the orbital energies.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    -76.016789472074251
    >>> mf.get_irrep_nelec()
    {'A1': 6, 'A2': 0, 'B1': 2, 'B2': 2}
    >>> mf.irrep_nelec = {'A2': 2}
    >>> mf.scf()
    -72.768201804695622
    >>> mf.get_irrep_nelec()
    {'A1': 6, 'A2': 2, 'B1': 2, 'B2': 0}
    '''
    def __init__(self, mol):
        hf.RHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nelec = {} # {'ir_name':int,...}
        self._keys = self._keys.union(['irrep_nelec'])

    def build(self, mol=None):
        if mol is None: mol = self.mol
        for irname in self.irrep_nelec:
            if irname not in self.mol.irrep_name:
                logger.warn(self, 'No irrep %s', irname)
        if mol.symmetry:
            check_irrep_nelec(self.mol, self.irrep_nelec, self.mol.nelectron)
        return hf.RHF.build(self, mol)

    eig = eig

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        g = hf.RHF.get_grad(self, mo_coeff, mo_occ, fock)
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
            return hf.RHF.get_occ(self, mo_energy, mo_coeff)

        orbsym = get_orbsym(self.mol, mo_coeff)
        mo_occ = numpy.zeros_like(mo_energy)
        rest_idx = numpy.ones(mo_occ.size, dtype=bool)
        nelec_fix = 0
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            if irname in self.irrep_nelec:
                ir_idx = numpy.where(orbsym == ir)[0]
                n = self.irrep_nelec[irname]
                occ_sort = numpy.argsort(mo_energy[ir_idx].round(9), kind='mergesort')
                occ_idx  = ir_idx[occ_sort[:n//2]]
                mo_occ[occ_idx] = 2
                nelec_fix += n
                rest_idx[ir_idx] = False
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        if nelec_float > 0:
            rest_idx = numpy.where(rest_idx)[0]
            occ_sort = numpy.argsort(mo_energy[rest_idx].round(9), kind='mergesort')
            occ_idx  = rest_idx[occ_sort[:nelec_float//2]]
            mo_occ[occ_idx] = 2

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
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym,
                            verbose=self.verbose)
        return mo_occ

    def _finalize(self):
        hf.RHF._finalize(self)

        # sort MOs wrt orbital energies, it should be done last.
        # Using mergesort because it is stable. We don't want to change the
        # ordering of the symmetry labels when two orbitals are degenerated.
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ> 0].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        idx = numpy.arange(self.mo_energy.size)
        idx = numpy.hstack((idx[self.mo_occ> 0][o_sort],
                            idx[self.mo_occ==0][v_sort]))
        self.mo_energy = self.mo_energy[idx]
        orbsym = get_orbsym(self.mol, self.mo_coeff)
        self.mo_coeff = lib.tag_array(self.mo_coeff[:,idx], orbsym=orbsym[idx])
        self.mo_occ = self.mo_occ[idx]
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
        return numpy.asarray(get_orbsym(self.mol, mo_coeff))
    orbsym = property(get_orbsym)

    get_wfnsym = get_wfnsym
    wfnsym = property(get_wfnsym)

    canonicalize = canonicalize
RHF = SymAdaptedRHF


class SymAdaptedROHF(rohf.ROHF):
    __doc__ = hf.SCF.__doc__ + '''
    Attributes for symmetry allowed ROHF:
        irrep_nelec : dict
            Specify the number of alpha/beta electrons for particular irrep
            {'ir_name':(int,int), ...}.
            For the irreps not listed in these dicts, the program will choose the
            occupancy based on the orbital energies.

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz', symmetry=True, charge=1, spin=1, verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    -75.619358861084052
    >>> mf.get_irrep_nelec()
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 1), 'B2': (1, 0)}
    >>> mf.irrep_nelec = {'B1': (1, 0)}
    >>> mf.scf()
    -75.425669486776457
    >>> mf.get_irrep_nelec()
    {'A1': (3, 3), 'A2': (0, 0), 'B1': (1, 0), 'B2': (1, 1)}
    '''
    def __init__(self, mol):
        rohf.ROHF.__init__(self, mol)
        self.irrep_nelec = {}
# use _irrep_doccs and _irrep_soccs help self.eig to compute orbital energy,
# do not overwrite them
        self._irrep_doccs = []
        self._irrep_soccs = []
        self._keys = self._keys.union(['irrep_nelec'])

    def dump_flags(self, verbose=None):
        rohf.ROHF.dump_flags(self, verbose)
        if self.irrep_nelec:
            logger.info(self, 'irrep_nelec %s', self.irrep_nelec)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if mol.symmetry:
            for irname in self.irrep_nelec:
                if irname not in self.mol.irrep_name:
                    logger.warn(self, 'No irrep %s', irname)

            fix_na, fix_nb = check_irrep_nelec(mol, self.irrep_nelec, self.nelec)[:2]
            alpha_open = beta_open = False
            for ne in self.irrep_nelec.values():
                if not isinstance(ne, (int, numpy.integer)):
                    alpha_open |= ne[0] > ne[1]
                    beta_open  |= ne[0] < ne[1]

            frozen_spin = fix_na - fix_nb
            if ((alpha_open and beta_open) or
                (0 < mol.spin < frozen_spin) or (frozen_spin < 0 < mol.spin) or
                (frozen_spin < mol.spin < 0) or (mol.spin < 0 < frozen_spin)):
                raise ValueError('Low-spin configuration was found in '
                                 'the irrep_nelec input. ROHF does not '
                                 'support low-spin configuration.')
        return hf.RHF.build(self, mol)

    @lib.with_doc(eig.__doc__)
    def eig(self, fock, s):
        e, c = eig(self, fock, s)
        if getattr(fock, 'focka', None) is not None:
            mo_ea = numpy.einsum('pi,pi->i', c, fock.focka.dot(c))
            mo_eb = numpy.einsum('pi,pi->i', c, fock.fockb.dot(c))
            e = lib.tag_array(e, mo_ea=mo_ea, mo_eb=mo_eb)
        return e, c

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        g = rohf.ROHF.get_grad(self, mo_coeff, mo_occ, fock)
        if self.mol.symmetry:
            occidxa = mo_occ > 0
            occidxb = mo_occ == 2
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa.reshape(-1,1) & occidxa
            uniq_var_b = viridxb.reshape(-1,1) & occidxb

            orbsym = get_orbsym(self.mol, mo_coeff)
            sym_forbid = orbsym.reshape(-1,1) != orbsym
            sym_forbid = sym_forbid[uniq_var_a | uniq_var_b]
            g[sym_forbid.ravel()] = 0
        return g

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        if not self.mol.symmetry:
            return rohf.ROHF.get_occ(self, mo_energy, mo_coeff)

        if getattr(mo_energy, 'mo_ea', None) is not None:
            mo_ea = mo_energy.mo_ea
            mo_eb = mo_energy.mo_eb
        else:
            mo_ea = mo_eb = mo_energy
        nmo = mo_ea.size
        mo_occ = numpy.zeros(nmo)

        orbsym = get_orbsym(self.mol, mo_coeff)

        rest_idx = numpy.ones(mo_occ.size, dtype=bool)
        neleca_fix = 0
        nelecb_fix = 0
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            if irname in self.irrep_nelec:
                ir_idx = numpy.where(orbsym == ir)[0]
                if isinstance(self.irrep_nelec[irname], (int, numpy.integer)):
                    nelecb = self.irrep_nelec[irname] // 2
                    neleca = self.irrep_nelec[irname] - nelecb
                else:
                    neleca, nelecb = self.irrep_nelec[irname]
                if neleca > nelecb:
                    ncore, nopen = nelecb, neleca - nelecb
                else:
                    ncore, nopen = neleca, nelecb - neleca
                mo_occ[ir_idx] = rohf._fill_rohf_occ(mo_energy[ir_idx],
                                                     mo_ea[ir_idx], mo_eb[ir_idx],
                                                     ncore, nopen)
                neleca_fix += neleca
                nelecb_fix += nelecb
                rest_idx[ir_idx] = False

        nelec_float = mol.nelectron - neleca_fix - nelecb_fix
        assert(nelec_float >= 0)
        if len(rest_idx) > 0:
            rest_idx = numpy.where(rest_idx)[0]
            nopen = abs(mol.spin - (neleca_fix - nelecb_fix))
            ncore = (nelec_float - nopen)//2
            mo_occ[rest_idx] = rohf._fill_rohf_occ(mo_energy[rest_idx],
                                                   mo_ea[rest_idx], mo_eb[rest_idx],
                                                   ncore, nopen)

        nocc, ncore = self.nelec
        nopen = nocc - ncore
        vir_idx = (mo_occ==0)
        if self.verbose >= logger.INFO and nocc < nmo and ncore > 0:
            ehomo = max(mo_energy[~vir_idx])
            elumo = min(mo_energy[ vir_idx])
            ndoccs = []
            nsoccs = []
            for i, ir in enumerate(mol.irrep_id):
                irname = mol.irrep_name[i]
                ir_idx = (orbsym == ir)

                ndoccs.append(numpy.count_nonzero(mo_occ[ir_idx]==2))
                nsoccs.append(numpy.count_nonzero(mo_occ[ir_idx]==1))
                if ehomo in mo_energy[ir_idx]:
                    irhomo = irname
                if elumo in mo_energy[ir_idx]:
                    irlumo = irname

            # to help self.eigh compute orbital energy
            self._irrep_doccs = ndoccs
            self._irrep_soccs = nsoccs

            logger.info(self, 'HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                        irhomo, ehomo, irlumo, elumo)

            logger.debug(self, 'double occ irrep_nelec = %s', ndoccs)
            logger.debug(self, 'single occ irrep_nelec = %s', nsoccs)
            #_dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym,
            #                verbose=self.verbose)
            if nopen > 0:
                core_idx = mo_occ == 2
                open_idx = mo_occ == 1
                vir_idx = mo_occ == 0
                logger.debug(self, '                  Roothaan           | alpha              | beta')
                logger.debug(self, '  Highest 2-occ = %18.15g | %18.15g | %18.15g',
                             max(mo_energy[core_idx]),
                             max(mo_ea[core_idx]), max(mo_eb[core_idx]))
                logger.debug(self, '  Lowest 0-occ =  %18.15g | %18.15g | %18.15g',
                             min(mo_energy[vir_idx]),
                             min(mo_ea[vir_idx]), min(mo_eb[vir_idx]))
                for i in numpy.where(open_idx)[0]:
                    logger.debug(self, '  1-occ =         %18.15g | %18.15g | %18.15g',
                                 mo_energy[i], mo_ea[i], mo_eb[i])

            numpy.set_printoptions(threshold=nmo)
            logger.debug(self, '  Roothaan mo_energy =\n%s', mo_energy)
            logger.debug1(self, '  alpha mo_energy =\n%s', mo_ea)
            logger.debug1(self, '  beta  mo_energy =\n%s', mo_eb)
            numpy.set_printoptions(threshold=1000)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        mo_a = mo_coeff[:,mo_occ>0]
        mo_b = mo_coeff[:,mo_occ==2]
        dm_a = numpy.dot(mo_a, mo_a.conj().T)
        dm_b = numpy.dot(mo_b, mo_b.conj().T)
        return numpy.array((dm_a, dm_b))

    def _finalize(self):
        rohf.ROHF._finalize(self)

        # sort MOs wrt orbital energies, it should be done last.
        # Using mergesort because it is stable. We don't want to change the
        # ordering of the symmetry labels when two orbitals are degenerated.
        c_sort = numpy.argsort(self.mo_energy[self.mo_occ==2].round(9), kind='mergesort')
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ==1].round(9), kind='mergesort')
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9), kind='mergesort')
        idx = numpy.arange(self.mo_energy.size)
        idx = numpy.hstack((idx[self.mo_occ==2][c_sort],
                            idx[self.mo_occ==1][o_sort],
                            idx[self.mo_occ==0][v_sort]))
        if getattr(self.mo_energy, 'mo_ea', None) is not None:
            mo_ea = self.mo_energy.mo_ea[idx]
            mo_eb = self.mo_energy.mo_eb[idx]
            self.mo_energy = lib.tag_array(self.mo_energy[idx],
                                           mo_ea=mo_ea, mo_eb=mo_eb)
        else:
            self.mo_energy = self.mo_energy[idx]
        orbsym = get_orbsym(self.mol, self.mo_coeff)
        self.mo_coeff = lib.tag_array(self.mo_coeff[:,idx], orbsym=orbsym[idx])
        self.mo_occ = self.mo_occ[idx]
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self

    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        from pyscf.lo import orth
        from pyscf.tools import dump_mat
        if not self.mol.symmetry:
            return rohf.ROHF.analyze(self, verbose, with_meta_lowdin, **kwargs)

        mol = self.mol
        mo_energy = self.mo_energy
        mo_occ = self.mo_occ
        mo_coeff = self.mo_coeff
        ovlp_ao = self.get_ovlp()
        log = logger.new_logger(self, verbose)
        if log.verbose >= logger.NOTE:
            self.dump_scf_summary(log)

            nirrep = len(mol.irrep_id)
            orbsym = get_orbsym(self.mol, mo_coeff)
            wfnsym = 0
            ndoccs = []
            nsoccs = []
            for k,ir in enumerate(mol.irrep_id):
                ndoccs.append(sum(orbsym[mo_occ==2] == ir))
                nsoccs.append(sum(orbsym[mo_occ==1] == ir))
                if nsoccs[k] % 2 == 1:
                    wfnsym ^= ir
            if mol.groupname in ('SO3', 'Dooh', 'Coov'):
                log.note('TODO: total wave-function symmetry for %s', mol.groupname)
            else:
                log.note('Wave-function symmetry = %s',
                         symm.irrep_id2name(mol.groupname, wfnsym))
            log.note('occupancy for each irrep:  ' + (' %4s'*nirrep),
                     *mol.irrep_name)
            log.note('double occ                 ' + (' %4d'*nirrep), *ndoccs)
            log.note('single occ                 ' + (' %4d'*nirrep), *nsoccs)
            log.note('**** MO energy ****')
            irname_full = {}
            for k,ir in enumerate(mol.irrep_id):
                irname_full[ir] = mol.irrep_name[k]
            irorbcnt = {}
            if getattr(mo_energy, 'mo_ea', None) is not None:
                mo_ea = mo_energy.mo_ea
                mo_eb = mo_energy.mo_eb
                log.note('                          Roothaan           | alpha              | beta')
                for k, j in enumerate(orbsym):
                    if j in irorbcnt:
                        irorbcnt[j] += 1
                    else:
                        irorbcnt[j] = 1
                    log.note('MO #%-4d(%-3s #%-2d) energy= %-18.15g | %-18.15g | %-18.15g occ= %g',
                             k+MO_BASE, irname_full[j], irorbcnt[j],
                             mo_energy[k], mo_ea[k], mo_eb[k], mo_occ[k])
            else:
                for k, j in enumerate(orbsym):
                    if j in irorbcnt:
                        irorbcnt[j] += 1
                    else:
                        irorbcnt[j] = 1
                    log.note('MO #%-3d (%s #%-2d), energy= %-18.15g occ= %g',
                             k+MO_BASE, irname_full[j], irorbcnt[j],
                             mo_energy[k], mo_occ[k])

        if log.verbose >= logger.DEBUG:
            label = mol.ao_labels()
            molabel = []
            irorbcnt = {}
            for k, j in enumerate(orbsym):
                if j in irorbcnt:
                    irorbcnt[j] += 1
                else:
                    irorbcnt[j] = 1
                molabel.append('#%-d(%s #%d)' %
                               (k+MO_BASE, irname_full[j], irorbcnt[j]))
            if with_meta_lowdin:
                log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) **')
                orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=ovlp_ao)
                c = reduce(numpy.dot, (orth_coeff.conj().T, ovlp_ao, mo_coeff))
            else:
                log.debug(' ** MO coefficients (expansion on AOs) **')
                c = mo_coeff
            dump_mat.dump_rec(self.stdout, c, label, molabel, start=MO_BASE, **kwargs)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        if with_meta_lowdin:
            pop_and_charge = self.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)
        else:
            pop_and_charge = self.mulliken_pop(mol, dm, s=ovlp_ao, verbose=log)
        dip = self.dip_moment(mol, dm, verbose=log)
        return pop_and_charge, dip

    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import uhf_symm
        if mol is None: mol = self.mol
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
            mo_coeff = (mo_coeff, mo_coeff)
        if isinstance(mo_occ, numpy.ndarray) and mo_occ.ndim == 1:
            mo_occ = (numpy.array(mo_occ>0, dtype=numpy.double),
                      numpy.array(mo_occ==2, dtype=numpy.double))
        return uhf_symm.get_irrep_nelec(mol, mo_coeff, mo_occ)

    @lib.with_doc(canonicalize.__doc__)
    def canonicalize(self, mo_coeff, mo_occ, fock=None):
        if getattr(fock, 'focka', None) is None:
            fock = self.get_fock(dm=self.make_rdm1(mo_coeff, mo_occ))
        mo_e, mo_coeff = canonicalize(self, mo_coeff, mo_occ, fock)
        mo_ea = numpy.einsum('pi,pi->i', mo_coeff, fock.focka.dot(mo_coeff))
        mo_eb = numpy.einsum('pi,pi->i', mo_coeff, fock.fockb.dot(mo_coeff))
        mo_e = lib.tag_array(mo_e, mo_ea=mo_ea, mo_eb=mo_eb)
        return mo_e, mo_coeff

    def get_orbsym(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_coeff is None:
            raise RuntimeError('SCF object %s not initialized' % self)
        return numpy.asarray(get_orbsym(self.mol, mo_coeff))
    orbsym = property(get_orbsym)

    get_wfnsym = get_wfnsym
    wfnsym = property(get_wfnsym)

ROHF = SymAdaptedROHF


def _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym, title='',
                    verbose=logger.DEBUG):
    log = logger.new_logger(mol, verbose)
    for i, ir in enumerate(mol.irrep_id):
        irname = mol.irrep_name[i]
        ir_idx = (orbsym == ir)
        nso = numpy.count_nonzero(ir_idx)
        nocc = numpy.count_nonzero(mo_occ[ir_idx])
        e_ir = mo_energy[ir_idx]
        if nocc == 0:
            log.debug('%s%s nocc = 0', title, irname)
        elif nocc == nso:
            log.debug('%s%s nocc = %d  HOMO = %.15g',
                      title, irname, nocc, e_ir[nocc-1])
        else:
            log.debug('%s%s nocc = %d  HOMO = %.15g  LUMO = %.15g',
                      title, irname, nocc, e_ir[nocc-1], e_ir[nocc])
            if e_ir[nocc-1]+1e-3 > elumo:
                log.warn('%s%s HOMO %.15g > system LUMO %.15g',
                         title, irname, e_ir[nocc-1], elumo)
            if e_ir[nocc] < ehomo+1e-3:
                log.warn('%s%s LUMO %.15g < system HOMO %.15g',
                         title, irname, e_ir[nocc], ehomo)
        log.debug('   mo_energy = %s', e_ir)


class HF1e(ROHF):
    def scf(self, *args):
        logger.info(self, '\n')
        logger.info(self, '******** 1 electron system ********')
        self.converged = True
        h1e = self.get_hcore(self.mol)
        s1e = self.get_ovlp(self.mol)
        self.mo_energy, self.mo_coeff = self.eig(h1e, s1e)
        self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
        self.e_tot = self.mo_energy[self.mo_occ>0][0] + self.mol.energy_nuc()
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
        symmetry = True
    )

    method = RHF(mol)
    method.verbose = 5
    method.irrep_nelec['A1u'] = 2
    energy = method.kernel()
    print(energy)
    method.analyze()
