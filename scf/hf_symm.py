#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic restricted Hartree Fock with symmetry.

The symmetry are not handled in a separate data structure.  Note that during
the SCF iteration,  the orbitals are grouped in terms of symmetry irreps.
But the orbitals in the result are sorted based on the orbital energies.
Function symm.label_orb_symm can be used to detect the symmetry of the
molecular orbitals.
'''

import time
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import symm
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf import rohf
from pyscf.scf import chkfile

# mo_energy, mo_coeff, mo_occ are all in nosymm representation

def analyze(mf, verbose=logger.DEBUG):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Occupancy for each irreps; Mulliken population analysis
    '''
    from pyscf.lo import orth
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    log = logger.Logger(mf.stdout, verbose)
    mol = mf.mol
    nirrep = len(mol.irrep_id)
    ovlp_ao = mf.get_ovlp()
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff,
                                 s=ovlp_ao)
    orbsym = numpy.array(orbsym)
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
                 k+1, irname_full[j], irorbcnt[j], mo_energy[k], mo_occ[k])

    if verbose >= logger.DEBUG:
        label = mol.spheric_labels(True)
        molabel = []
        irorbcnt = {}
        for k, j in enumerate(orbsym):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            molabel.append('#%-d(%s #%d)' % (k+1, irname_full[j], irorbcnt[j]))
        log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) **')
        orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=ovlp_ao)
        c = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mo_coeff))
        dump_mat.dump_rec(mf.stdout, c, label, molabel, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_meta(mol, dm, s=ovlp_ao, verbose=log)

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
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff, s)
    orbsym = numpy.array(orbsym)
    irrep_nelec = dict([(mol.irrep_name[k], int(sum(mo_occ[orbsym==ir])))
                        for k, ir in enumerate(mol.irrep_id)])
    return irrep_nelec

def canonicalize(mf, mo_coeff, mo_occ, fock=None):
    '''Canonicalization diagonalizes the Fock matrix in occupied, open,
    virtual subspaces separatedly (without change occupancy).
    '''
    if fock is None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        fock = mf.get_hcore() + mf.get_jk(mol, dm)
    coreidx = mo_occ == 2
    viridx = mo_occ == 0
    openidx = ~(coreidx | viridx)
    mo = numpy.empty_like(mo_coeff)
    mo_e = numpy.empty(mo_occ.size)
    s = mf.get_ovlp()
    for idx in (coreidx, openidx, viridx):
        if numpy.count_nonzero(idx) > 0:
            orb = mo_coeff[:,idx]
            f1 = reduce(numpy.dot, (orb.T.conj(), fock, orb))
            e, c = scipy.linalg.eigh(f1)
            c = numpy.dot(mo_coeff[:,idx], c)
            mo[:,idx] = _symmetrize_canonicalization_(mf.mol, e, c, s)
            mo_e[idx] = e
    return mo_e, mo

def _symmetrize_canonicalization_(mol, mo_energy, mo_coeff, s):
    '''Restore symmetry for canonicalized orbitals
    '''
    def search_for_degeneracy(mo_energy):
        idx = numpy.where(abs(mo_energy[1:] - mo_energy[:-1]) < 1e-6)[0]
        return numpy.unique(numpy.hstack((idx, idx+1)))

    degidx = search_for_degeneracy(mo_energy)
    logger.debug1(mol, 'degidx %s', degidx)
    if degidx.size > 0:
        esub = mo_energy[degidx]
        csub = mo_coeff[:,degidx]
        scsub = numpy.dot(s, csub)
        emin = abs(esub).min() * .5
        es = []
        cs = []
        for i,ir in enumerate(mol.irrep_id):
            so = mol.symm_orb[i]
            sosc = numpy.dot(so.T, scsub)
            s_ir = reduce(numpy.dot, (so.T, s, so))
            fock_ir = numpy.dot(sosc*esub, sosc.T)
            mo_energy, u = scipy.linalg.eigh(fock_ir, s_ir)
            idx = abs(mo_energy) > emin
            es.append(mo_energy[idx])
            cs.append(numpy.dot(mol.symm_orb[i], u[:,idx]))
        es = numpy.hstack(es).round(9)
        idx = numpy.argsort(es)
        assert(numpy.allclose(es[idx], esub))
        mo_coeff[:,degidx] = numpy.hstack(cs)[:,idx]
    return mo_coeff

def so2ao_mo_coeff(so, irrep_mo_coeff):
    '''Transfer the basis of MO coefficients, from spin-adapted basis to AO basis
    '''
    return numpy.hstack([numpy.dot(so[ir],irrep_mo_coeff[ir]) \
                         for ir in range(so.__len__())])

def check_irrep_nelec(mol, irrep_nelec, nelec):
    for irname in irrep_nelec.keys():
        if irname not in mol.irrep_name:
            logger.warn(mol, 'Molecule does not have irrep %s', irname)

    float_irname = []
    fix_na = 0
    fix_nb = 0
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
    if fix_na < fix_nb:
        raise ValueError('alpha electrons %d < beta electrons %d\n'
                         'irrep_nelec %s' % (fix_na, fix_nb, irrep_nelec))
    if mol.spin < fix_na-fix_nb:
        raise ValueError('alpha electrons %d - beta electrons %d > mol.spin %d\n'
                         'irrep_nelec %s' % (fix_na, fix_nb, mol.spin, irrep_nelec))

    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    fix_ne = fix_na + fix_nb
    if ((fix_na > neleca) or (fix_nb > nelecb) or
        (fix_na+nelecb > mol.nelectron) or
        (fix_nb+neleca > mol.nelectron)):
        msg =('More electrons defined by irrep_nelec than total num electrons. '
              'mol.nelectron = %d  irrep_nelec = %s' %
              (mol.nelectron, irrep_nelec))
        raise ValueError(msg)
    else:
        logger.info(mol, 'fix %d electrons in irreps %s',
                    fix_ne, irrep_nelec.items())

    if len(set(float_irname)) == 0 and fix_ne != mol.nelectron:
        msg =('Num electrons defined by irrep_nelec != total num electrons. '
              'mol.nelectron = %d  irrep_nelec = %s' %
              (mol.nelectron, irrep_nelec))
        raise ValueError(msg)
    else:
        logger.info(mol, '    %d free electrons in irreps %s',
                    mol.nelectron-fix_ne, ' '.join(float_irname))
    return fix_na, fix_nb, float_irname


class RHF(hf.RHF):
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

    def dump_flags(self):
        hf.RHF.dump_flags(self)
        check_irrep_nelec(self.mol, self.irrep_nelec, self.mol.nelectron)

#TODO: force E1gx/E1gy ... use the same coefficients
    def eig(self, h, s):
        '''Solve generalized eigenvalue problem, for each irrep.  The
        eigenvalues and eigenvectors are not sorted to ascending order.
        Instead, they are grouped based on irreps.
        '''
        nirrep = self.mol.symm_orb.__len__()
        h = symm.symmetrize_matrix(h, self.mol.symm_orb)
        s = symm.symmetrize_matrix(s, self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = hf.SCF.eig(self, h[ir], s[ir])
            cs.append(c)
            es.append(e)
        e = numpy.hstack(es)
        c = so2ao_mo_coeff(self.mol.symm_orb, cs)
        return e, c

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        mol = self.mol
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(mol) + self.get_veff(self.mol, dm1)
        orbsym = symm.label_orb_symm(self, mol.irrep_id, mol.symm_orb,
                                     mo_coeff, self.get_ovlp(), False)
        orbsym = numpy.asarray(orbsym)

        occidx = mo_occ > 0
        viridx = ~occidx
        g = reduce(numpy.dot, (mo_coeff[:,occidx].T.conj(), fock,
                               mo_coeff[:,viridx]))
        sym_allow = orbsym[occidx].reshape(-1,1) == orbsym[viridx]
        g[~sym_allow] = 0
        return g.ravel()

    def get_occ(self, mo_energy=None, mo_coeff=None, orbsym=None):
        ''' We assumed mo_energy are grouped by symmetry irreps, (see function
        self.eig). The orbitals are sorted after SCF.
        '''
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        if orbsym is None:
            if mo_coeff is not None:
                orbsym = symm.label_orb_symm(self, mol.irrep_id, mol.symm_orb,
                                             mo_coeff, self.get_ovlp(), False)
                orbsym = numpy.asarray(orbsym)
            else:
                orbsym = [numpy.repeat(ir, mol.symm_orb[i].shape[1])
                          for i, ir in enumerate(mol.irrep_id)]
                orbsym = numpy.hstack(orbsym)
        else:
            orbsym = numpy.asarray(orbsym)
        assert(mo_energy.size == orbsym.size)

        mo_occ = numpy.zeros_like(mo_energy)
        rest_idx = []
        nelec_fix = 0
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idx = numpy.where(orbsym == ir)[0]
            if irname in self.irrep_nelec:
                n = self.irrep_nelec[irname]
                occ_sort = numpy.argsort(mo_energy[ir_idx].round(9))
                occ_idx  = ir_idx[occ_sort[:n//2]]
                mo_occ[occ_idx] = 2
                nelec_fix += n
            else:
                rest_idx.append(ir_idx)
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        if nelec_float > 0:
            rest_idx = numpy.hstack(rest_idx)
            occ_sort = numpy.argsort(mo_energy[rest_idx].round(9))
            occ_idx  = rest_idx[occ_sort[:nelec_float//2]]
            mo_occ[occ_idx] = 2

        vir_idx = (mo_occ==0)
        if self.verbose >= logger.INFO and numpy.count_nonzero(vir_idx) > 0:
            ehomo = max(mo_energy[mo_occ>0 ])
            elumo = min(mo_energy[mo_occ==0])
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
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ>0 ].round(9))
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9))
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ>0][o_sort],
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ>0].take(o_sort, axis=1),
                                      self.mo_coeff[:,self.mo_occ==0].take(v_sort, axis=1)))
        nocc = len(o_sort)
        self.mo_occ[:nocc] = 2
        self.mo_occ[nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        return analyze(self, verbose)

    @lib.with_doc(get_irrep_nelec.__doc__)
    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None, s=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if s is None: s = self.get_ovlp()
        return get_irrep_nelec(mol, mo_coeff, mo_occ, s)

    canonicalize = canonicalize


class HF1e(hf.SCF):
    def scf(self, *args):
        logger.info(self, '\n')
        logger.info(self, '******** 1 electron system ********')
        self.converged = True
        h1e = self.get_hcore(self.mol)
        s1e = self.get_ovlp(self.mol)
        nirrep = self.mol.symm_orb.__len__()
        h1e = symm.symmetrize_matrix(h1e, self.mol.symm_orb)
        s1e = symm.symmetrize_matrix(s1e, self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = hf.SCF.eig(self, h1e[ir], s1e[ir])
            cs.append(c)
            es.append(e)
        e = numpy.hstack(es).round(9)
        idx = numpy.argsort(e)
        self.mo_energy = e[idx]
        self.mo_coeff = so2ao_mo_coeff(self.mol.symm_orb, cs)[:,idx]
        self.mo_occ = numpy.zeros_like(self.mo_energy)
        self.mo_occ[0] = 1
        self.e_tot = self.mo_energy[0] + self.mol.energy_nuc()
        return self.e_tot


class ROHF(rohf.ROHF):
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

    def dump_flags(self):
        rohf.ROHF.dump_flags(self)
        check_irrep_nelec(self.mol, self.irrep_nelec, self.nelec)

    def eig(self, h, s):
        nirrep = self.mol.symm_orb.__len__()
        h = symm.symmetrize_matrix(h, self.mol.symm_orb)
        s = symm.symmetrize_matrix(s, self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = hf.SCF.eig(self, h[ir], s[ir])
            cs.append(c)
            es.append(e)
        e = numpy.hstack(es)
        c = so2ao_mo_coeff(self.mol.symm_orb, cs)
        return e, c

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
                 diis_start_cycle=None, level_shift_factor=None,
                 damp_factor=None):
# Roothaan's effective fock
# http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html
#          |  closed     open    virtual
#  ----------------------------------------
#  closed  |    Fc        Fb       Fc
#  open    |    Fb        Fc       Fa
#  virtual |    Fc        Fa       Fc
# Fc = (Fa+Fb)/2
        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.array((dm*.5, dm*.5))
        self._focka_ao = h1e + vhf[0]
        self._fockb_ao = h1e + vhf[1]
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        dmsf = dm[0]+dm[1]
        mo_space = scipy.linalg.eigh(-dmsf, s1e, type=2)[1]
        fa = reduce(numpy.dot, (mo_space.T, self._focka_ao, mo_space))
        fb = reduce(numpy.dot, (mo_space.T, self._fockb_ao, mo_space))
        feff = (fa + fb) * .5
        feff[:ncore,ncore:nocc] = fb[:ncore,ncore:nocc]
        feff[ncore:nocc,:ncore] = fb[ncore:nocc,:ncore]
        feff[nocc:,ncore:nocc] = fa[nocc:,ncore:nocc]
        feff[ncore:nocc,nocc:] = fa[ncore:nocc,nocc:]
        cinv = numpy.dot(mo_space.T, s1e)
        f = reduce(numpy.dot, (cinv.T, feff, cinv))

        if 0 <= cycle < diis_start_cycle-1:
            f = hf.damping(s1e, dm[0], f, damp_factor)
        if adiis and cycle >= diis_start_cycle:
            f = adiis.update(s1e, dm[0], f)
        f = hf.level_shift(s1e, dm[0], f, level_shift_factor)
        return f

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        mol = self.mol
        if fock is None:
            dm1 = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(mol) + self.get_veff(self.mol, dm1)
        orbsym = symm.label_orb_symm(self, mol.irrep_id, mol.symm_orb,
                                     mo_coeff, self.get_ovlp(), False)
        orbsym = numpy.asarray(orbsym)
        sym_allow = orbsym.reshape(-1,1) == orbsym

        occidxa = mo_occ > 0
        occidxb = mo_occ == 2
        viridxa = ~occidxa
        viridxb = ~occidxb
        focka = reduce(numpy.dot, (mo_coeff.T, fock[0], mo_coeff))
        fockb = reduce(numpy.dot, (mo_coeff.T, fock[1], mo_coeff))
        uniq_var_a = viridxa.reshape(-1,1) & occidxa
        uniq_var_b = viridxb.reshape(-1,1) & occidxb
        g = focka + fockb
        g[~sym_allow] = 0
        return g[uniq_var_a | uniq_var_b].ravel()

    def get_occ(self, mo_energy=None, mo_coeff=None, orbsym=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        if mo_coeff is None or self._focka_ao is None:
            mo_ea = mo_eb = mo_energy
        else:
            mo_ea = numpy.einsum('ik,ik->k', mo_coeff, self._focka_ao.dot(mo_coeff))
            mo_eb = numpy.einsum('ik,ik->k', mo_coeff, self._fockb_ao.dot(mo_coeff))
        nmo = mo_ea.size
        mo_occ = numpy.zeros(nmo)
        if orbsym is None:
            if mo_coeff is not None:
                orbsym = symm.label_orb_symm(self, mol.irrep_id, mol.symm_orb,
                                             mo_coeff, self.get_ovlp(), False)
                orbsym = numpy.asarray(orbsym)
            else:
                orbsym = [numpy.repeat(ir, mol.symm_orb[i].shape[1])
                          for i, ir in enumerate(mol.irrep_id)]
                orbsym = numpy.hstack(orbsym)
        else:
            orbsym = numpy.asarray(orbsym)
        assert(mo_energy.size == orbsym.size)

        float_idx = []
        neleca_fix = 0
        nelecb_fix = 0
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idx = numpy.where(orbsym == ir)[0]
            if irname in self.irrep_nelec:
                if isinstance(self.irrep_nelec[irname], (int, numpy.integer)):
                    nelecb = self.irrep_nelec[irname] // 2
                    neleca = self.irrep_nelec[irname] - nelecb
                else:
                    neleca, nelecb = self.irrep_nelec[irname]
                mo_occ[ir_idx] = rohf._fill_rohf_occ(mo_energy[ir_idx],
                                                     mo_ea[ir_idx], mo_eb[ir_idx],
                                                     nelecb, neleca-nelecb)
                neleca_fix += neleca
                nelecb_fix += nelecb
            else:
                float_idx.append(ir_idx)

        nelec_float = mol.nelectron - neleca_fix - nelecb_fix
        assert(nelec_float >= 0)
        if len(float_idx) > 0:
            float_idx = numpy.hstack(float_idx)
            nopen = mol.spin - (neleca_fix - nelecb_fix)
            ncore = (nelec_float - nopen)//2
            mo_occ[float_idx] = rohf._fill_rohf_occ(mo_energy[float_idx],
                                                    mo_ea[float_idx],
                                                    mo_eb[float_idx],
                                                    ncore, nopen)

        ncore = self.nelec[1]
        nocc  = self.nelec[0]
        nopen = nocc - ncore
        vir_idx = (mo_occ==0)
        if self.verbose >= logger.INFO and nocc < nmo and ncore > 0:
            ehomo = max(mo_energy[mo_occ> 0])
            elumo = min(mo_energy[mo_occ==0])
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

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        mo_a = mo_coeff[:,mo_occ>0]
        mo_b = mo_coeff[:,mo_occ==2]
        dm_a = numpy.dot(mo_a, mo_a.T)
        dm_b = numpy.dot(mo_b, mo_b.T)
        return numpy.array((dm_a, dm_b))

    def _finalize(self):
        rohf.ROHF._finalize(self)

        # sort MOs wrt orbital energies, it should be done last.
        c_sort = numpy.argsort(self.mo_energy[self.mo_occ==2].round(9))
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ==1].round(9))
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0].round(9))
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ==2][c_sort],
                                       self.mo_energy[self.mo_occ==1][o_sort],
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ==2].take(c_sort, axis=1),
                                      self.mo_coeff[:,self.mo_occ==1].take(o_sort, axis=1),
                                      self.mo_coeff[:,self.mo_occ==0].take(v_sort, axis=1)))
        ncore = len(c_sort)
        nocc = ncore + len(o_sort)
        self.mo_occ[:ncore] = 2
        self.mo_occ[ncore:nocc] = 1
        self.mo_occ[nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                             self.mo_coeff, self.mo_occ, overwrite_mol=False)
        return self

    def analyze(self, verbose=logger.DEBUG):
        from pyscf.lo import orth
        from pyscf.tools import dump_mat
        mo_energy = self.mo_energy
        mo_occ = self.mo_occ
        mo_coeff = self.mo_coeff
        log = logger.Logger(self.stdout, verbose)
        mol = self.mol
        nirrep = len(mol.irrep_id)
        ovlp_ao = self.get_ovlp()
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff,
                                     s=ovlp_ao)
        orbsym = numpy.array(orbsym)
        wfnsym = 0
        ndoccs = []
        nsoccs = []
        for k,ir in enumerate(mol.irrep_id):
            ndoccs.append(sum(orbsym[mo_occ==2] == ir))
            nsoccs.append(sum(orbsym[mo_occ==1] == ir))
            if nsoccs[k] % 2:
                wfnsym ^= ir
        if mol.groupname in ('Dooh', 'Coov'):
            log.info('TODO: total symmetry for %s', mol.groupname)
        else:
            log.info('total symmetry = %s',
                     symm.irrep_id2name(mol.groupname, wfnsym))
        log.info('occupancy for each irrep:  ' + (' %4s'*nirrep),
                 *mol.irrep_name)
        log.info('double occ                 ' + (' %4d'*nirrep), *ndoccs)
        log.info('single occ                 ' + (' %4d'*nirrep), *nsoccs)
        log.info('**** MO energy ****')
        irname_full = {}
        for k,ir in enumerate(mol.irrep_id):
            irname_full[ir] = mol.irrep_name[k]
        irorbcnt = {}
        if self._focka_ao is None:
            for k, j in enumerate(orbsym):
                if j in irorbcnt:
                    irorbcnt[j] += 1
                else:
                    irorbcnt[j] = 1
                log.note('MO #%-3d (%s #%-2d), energy= %-18.15g occ= %g',
                         k+1, irname_full[j], irorbcnt[j],
                         mo_energy[k], mo_occ[k])
        else:
            mo_ea = numpy.einsum('ik,ik->k', mo_coeff, self._focka_ao.dot(mo_coeff))
            mo_eb = numpy.einsum('ik,ik->k', mo_coeff, self._fockb_ao.dot(mo_coeff))
            log.note('                          Roothaan           | alpha              | beta')
            for k, j in enumerate(orbsym):
                if j in irorbcnt:
                    irorbcnt[j] += 1
                else:
                    irorbcnt[j] = 1
                log.note('MO #%-4d(%-3s #%-2d) energy= %-18.15g | %-18.15g | %-18.15g occ= %g',
                         k+1, irname_full[j], irorbcnt[j],
                         mo_energy[k], mo_ea[k], mo_eb[k], mo_occ[k])

        if verbose >= logger.DEBUG:
            label = mol.spheric_labels(True)
            molabel = []
            irorbcnt = {}
            for k, j in enumerate(orbsym):
                if j in irorbcnt:
                    irorbcnt[j] += 1
                else:
                    irorbcnt[j] = 1
                molabel.append('#%-d(%s #%d)' % (k+1, irname_full[j], irorbcnt[j]))
            log.debug(' ** MO coefficients (expansion on meta-Lowdin AOs) **')
            orth_coeff = orth.orth_ao(mol, 'meta_lowdin', s=ovlp_ao)
            c = reduce(numpy.dot, (orth_coeff.T, ovlp_ao, mo_coeff))
            dump_mat.dump_rec(self.stdout, c, label, molabel, start=1)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        return self.mulliken_meta(mol, dm, s=ovlp_ao, verbose=verbose)

    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import uhf_symm
        if mol is None: mol = self.mol
        if mo_coeff is None: mo_coeff = (self.mo_coeff,self.mo_coeff)
        if mo_occ is None: mo_occ = ((self.mo_occ>0), (self.mo_occ==2))
        return uhf_symm.get_irrep_nelec(mol, mo_coeff, mo_occ)

    @lib.with_doc(canonicalize.__doc__)
    def canonicalize(self, mo_coeff, mo_occ, fock=None):
        dm = self.make_rdm1(mo_coeff, mo_occ)
        if fock is None:
            fock = self.get_hcore() + self.get_jk(mol, dm)
        if isinstance(fock, numpy.ndarray) and fock.ndim == 3:
            fock = rohf.get_roothaan_fock(fock, dm, self.get_ovlp())
        return canonicalize(self, mo_coeff, mo_occ, fock)


def _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym, title='',
                    verbose=logger.DEBUG):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    nirrep = mol.symm_orb.__len__()
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
                log.warn('!! %s%s HOMO %.15g > system LUMO %.15g',
                         title, irname, e_ir[nocc-1], elumo)
            if e_ir[nocc] < ehomo+1e-3:
                log.warn('!! %s%s LUMO %.15g < system HOMO %.15g',
                         title, irname, e_ir[nocc], ehomo)
        log.debug('   mo_energy = %s', e_ir)


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
    method.irrep_nelec['A1u'] = 2
    energy = method.scf()
    print(energy)
