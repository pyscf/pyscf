#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
from pyscf import symm
from pyscf.scf import hf
from pyscf.scf import rohf
from pyscf.scf import chkfile


'''
Non-relativistic restricted Hartree Fock with symmetry.

The symmetry are not handled in a separate data structure.  Note that during
the SCF iteration,  the orbitals are grouped in terms of symmetry irreps.
But the orbitals in the result are sorted based on the orbital energies.
Function symm.label_orb_symm can be used to detect the symmetry of the
molecular orbitals.
'''

# mo_energy, mo_coeff, mo_occ are all in nosymm representation

def analyze(mf, verbose=logger.DEBUG):
    '''Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Occupancy for each irreps; Mulliken population analysis
    '''
    from pyscf.tools import dump_mat
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    log = pyscf.lib.logger.Logger(mf.stdout, verbose)
    mol = mf.mol
    nirrep = len(mol.irrep_id)
    ovlp_ao = mf.get_ovlp()
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo_coeff,
                                 s=ovlp_ao)
    orbsym = numpy.array(orbsym)
    wfnsym = 0
    noccs = [sum(orbsym[mo_occ>0]==ir) for ir in mol.irrep_id]
    log.info('total symmetry = %s', symm.irrep_id2name(mol.groupname, wfnsym))
    log.info('occupancy for each irrep:  ' + (' %4s'*nirrep), *mol.irrep_name)
    log.info('double occ                 ' + (' %4d'*nirrep), *noccs)
    log.info('**** MO energy ****')
    irname_full = {}
    for k,ir in enumerate(mol.irrep_id):
        irname_full[ir] = mol.irrep_name[k]
    irorbcnt = {}
    for k, j in enumerate(orbsym):
        if j in irorbcnt:
            irorbcnt[j] += 1
        else:
            irorbcnt[j] = 1
        log.info('MO #%d (%s #%d), energy= %.15g occ= %g',
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
        log.debug(' ** MO coefficients **')
        dump_mat.dump_rec(mol.stdout, mo_coeff, label, molabel, start=1)

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

def so2ao_mo_coeff(so, irrep_mo_coeff):
    '''Transfer the basis of MO coefficients, from spin-adapted basis to AO basis
    '''
    return numpy.hstack([numpy.dot(so[ir],irrep_mo_coeff[ir]) \
                         for ir in range(so.__len__())])


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
        logger.info(self, '%s with symmetry adapted basis',
                    self.__class__.__name__)
        float_irname = []
        fix_ne = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if irname in self.irrep_nelec:
                fix_ne += self.irrep_nelec[irname]
            else:
                float_irname.append(irname)
        if fix_ne > 0:
            logger.info(self, 'fix %d electrons in irreps %s',
                        fix_ne, self.irrep_nelec.items())
            if fix_ne > self.mol.nelectron:
                logger.error(self, 'num. electron error in irrep_nelec %s',
                             self.irrep_nelec.items())
                raise ValueError('irrep_nelec')
        if float_irname:
            logger.info(self, '%d free electrons in irreps %s',
                        self.mol.nelectron-fix_ne, ' '.join(float_irname))
        elif fix_ne != self.mol.nelectron:
            logger.error(self, 'number of electrons error in irrep_nelec %s',
                         self.irrep_nelec.items())
            raise ValueError('irrep_nelec')

    def build_(self, mol=None):
        for irname in self.irrep_nelec.keys():
            if irname not in self.mol.irrep_name:
                logger.warn(self, '!! No irrep %s', irname)
        return hf.RHF.build_(self, mol)

#TODO: force E1gx/E1gy ... use the same coefficients
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

    def get_occ(self, mo_energy, mo_coeff=None):
        ''' We cannot assume default mo_energy value, because the orbital
        energies are sorted after doing SCF.  But in this function, we need
        the orbital energies are grouped by symmetry irreps
        '''
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nirrep = mol.symm_orb.__len__()
        mo_e_left = []
        idx_e_left = []
        nelec_fix = 0
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nelec:
                n = self.irrep_nelec[irname]
                mo_occ[p0:p0+n//2] = 2
                nelec_fix += n
            else:
                idx_e_left.append(range(p0,p0+nso))
            p0 += nso
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        if nelec_float > 0:
            idx_e_left = numpy.hstack(idx_e_left)
            mo_e_left = mo_energy[idx_e_left]
            mo_e_sort = numpy.argsort(mo_e_left)
            occ_idx = idx_e_left[mo_e_sort][:(nelec_float//2)]
            mo_occ[occ_idx] = 2

        ehomo = max(mo_energy[mo_occ>0 ])
        elumo = min(mo_energy[mo_occ==0])
        noccs = []
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]

            noccs.append(int(mo_occ[p0:p0+nso].sum()))
            if ehomo in mo_energy[p0:p0+nso]:
                irhomo = irname
            if elumo in mo_energy[p0:p0+nso]:
                irlumo = irname
            p0 += nso
        logger.info(self, 'HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                    irhomo, ehomo, irlumo, elumo)
        if self.verbose >= logger.DEBUG:
            logger.debug(self, 'irrep_nelec = %s', noccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
        return mo_occ

    def _finalize_(self):
        hf.RHF._finalize_(self)

        # sort MOs wrt orbital energies, it should be done last.
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ>0])
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0])
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ>0][o_sort], \
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ>0][:,o_sort], \
                                      self.mo_coeff[:,self.mo_occ==0][:,v_sort]))
        nocc = len(o_sort)
        self.mo_occ[:nocc] = 2
        self.mo_occ[nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             self.hf_energy, self.mo_energy,
                             self.mo_coeff, self.mo_occ)

    def analyze(self, verbose=logger.DEBUG):
        return analyze(self, verbose)

    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None, s=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if s is None: s = self.get_ovlp()
        return get_irrep_nelec(mol, mo_coeff, mo_occ, s)


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
        e = numpy.hstack(es)
        idx = numpy.argsort(e)
        self.mo_energy = e[idx]
        self.mo_coeff = so2ao_mo_coeff(self.mol.symm_orb, cs)[:,idx]
        self.mo_occ = numpy.zeros_like(self.mo_energy)
        self.mo_occ[0] = 1
        self.hf_energy = self.mo_energy[0] + self.mol.energy_nuc()
        return self.hf_energy


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
        logger.info(self, '%s with symmetry adapted basis',
                    self.__class__.__name__)
#TODO: improve the sainity check
        float_irname = []
        fix_na = 0
        fix_nb = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if irname in self.irrep_nelec:
                if isinstance(self.irrep_nelec[irname], (int, numpy.integer)):
                    nb = self.irrep_nelec[irname] // 2
                    fix_na += self.irrep_nelec[irname] - nb
                    fix_nb += nb
                else:
                    fix_na += self.irrep_nelec[irname][0]
                    fix_nb += self.irrep_nelec[irname][1]
            else:
                float_irname.append(irname)
        float_irname = set(float_irname)
        if fix_na+fix_nb > 0:
            logger.info(self, 'fix %d electrons in irreps: %s',
                        fix_na+fix_nb, str(self.irrep_nelec.items()))
            if ((fix_na+fix_nb > self.mol.nelectron) or
                (fix_na>self.nelec[0]) or (fix_nb>self.nelec[1]) or
                (fix_na+self.nelec[1]>self.mol.nelectron) or
                (fix_nb+self.nelec[0]>self.mol.nelectron)):
                logger.error(self, 'electron number error in irrep_nelec %s',
                             self.irrep_nelec.items())
                raise ValueError('irrep_nelec')
        if float_irname:
            logger.info(self, '%d free electrons in irreps %s',
                        self.mol.nelectron-fix_na-fix_nb,
                        ' '.join(float_irname))
        elif fix_na+fix_nb != self.mol.nelectron:
            logger.error(self, 'electron number error in irrep_nelec %s',
                         self.irrep_nelec.items())
            raise ValueError('irrep_nelec')

    def build_(self, mol=None):
        # specify alpha,beta for same irreps
        na = 0
        nb = 0
        for x in self.irrep_nelec.values():
            if isinstance(x, (int, numpy.integer)):
                v = x // 2
                na += x - v
                nb += v
            else:
                na += x[0]
                nb += x[1]
        nopen = self.mol.spin
        assert(na >= nb and nopen >= na-nb)
        for irname in self.irrep_nelec.keys():
            if irname not in self.mol.irrep_name:
                logger.warn(self, '!! No irrep %s', irname)
        return hf.RHF.build_(self, mol)

    # same to RHF.eig
    def eig(self, h, s):
        ncore = (self.mol.nelectron-self.mol.spin) // 2
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

    def get_fock_(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None,
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
            level_shift_factor = self.level_shift_factor
        if damp_factor is None:
            damp_factor = self.damp_factor
        self._focka_ao = h1e + vhf[0]
        fockb_ao = h1e + vhf[1]
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        dmsf = dm[0]+dm[1]
        mo_space = scipy.linalg.eigh(-dmsf, s1e, type=2)[1]
        fa = reduce(numpy.dot, (mo_space.T, self._focka_ao, mo_space))
        fb = reduce(numpy.dot, (mo_space.T, fockb_ao, mo_space))
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

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nirrep = mol.symm_orb.__len__()
        float_idx = []
        neleca_fix = 0
        nelecb_fix = 0
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nelec:
                if isinstance(self.irrep_nelec[irname], (int, numpy.integer)):
                    ncore = self.irrep_nelec[irname] // 2
                    nocc = self.irrep_nelec[irname] - ncore
                else:
                    ncore = self.irrep_nelec[irname][1]
                    nocc = self.irrep_nelec[irname][0]
                mo_occ[p0:p0+ncore] = 2
                mo_occ[p0+ncore:p0+nocc] = 1
                neleca_fix += nocc
                nelecb_fix += ncore
            else:
                float_idx.append(range(p0,p0+nso))
            p0 += nso

        mo_energy = mo_energy.copy()  # Roothan Fock eigenvalue + alpha energy
        nelec_float = mol.nelectron - neleca_fix - nelecb_fix
        assert(nelec_float >= 0)
        if len(float_idx) > 0:
            float_idx = numpy.hstack(float_idx)
            nopen = mol.spin - (neleca_fix - nelecb_fix)
            ncore = (nelec_float - nopen)//2
            ecore = mo_energy[float_idx]
            core_sort = numpy.argsort(ecore)
            core_idx = float_idx[core_sort][:ncore]
            open_idx = float_idx[core_sort][ncore:]
            if mo_coeff is None:
                open_mo_energy = mo_energy
            else:
                open_mo_energy = numpy.einsum('ki,ki->i', mo_coeff,
                                              self._focka_ao.dot(mo_coeff))
            eopen = open_mo_energy[open_idx]
            mo_energy[open_idx] = eopen
            open_sort = numpy.argsort(eopen)
            open_idx = open_idx[open_sort]
            mo_occ[core_idx] = 2
            mo_occ[open_idx[:nopen]] = 1

        viridx = mo_occ==0
        if self.verbose < logger.INFO or viridx.sum() == 0:
            return mo_occ
        ehomo = max(mo_energy[mo_occ>0])
        elumo = min(mo_energy[viridx])
        ndoccs = []
        nsoccs = []
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]

            ndoccs.append((mo_occ[p0:p0+nso]==2).sum())
            nsoccs.append((mo_occ[p0:p0+nso]==1).sum())
            if ehomo in mo_energy[p0:p0+nso]:
                irhomo = irname
            if elumo in mo_energy[p0:p0+nso]:
                irlumo = irname
            p0 += nso

        # to help self.eigh compute orbital energy
        self._irrep_doccs = ndoccs
        self._irrep_soccs = nsoccs

        logger.info(self, 'HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                    irhomo, ehomo, irlumo, elumo)
        if self.verbose >= logger.DEBUG:
            logger.debug(self, 'double occ irrep_nelec = %s', ndoccs)
            logger.debug(self, 'single occ irrep_nelec = %s', nsoccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
            p0 = 0
            for ir in range(nirrep):
                irname = mol.irrep_name[ir]
                nso = mol.symm_orb[ir].shape[1]
                logger.debug2(self, 'core_mo_energy of %s = %s',
                              irname, mo_energy[p0:p0+nso])
                logger.debug2(self, 'open_mo_energy of %s = %s',
                              irname, open_mo_energy[p0:p0+nso])
                p0 += nso
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

    def _finalize_(self):
        rohf.ROHF._finalize_(self)

        # sort MOs wrt orbital energies, it should be done last.
        c_sort = numpy.argsort(self.mo_energy[self.mo_occ==2])
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ==1])
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0])
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ==2][c_sort],
                                       self.mo_energy[self.mo_occ==1][o_sort],
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ==2][:,c_sort],
                                      self.mo_coeff[:,self.mo_occ==1][:,o_sort],
                                      self.mo_coeff[:,self.mo_occ==0][:,v_sort]))
        ncore = len(c_sort)
        nocc = ncore + len(o_sort)
        self.mo_occ[:ncore] = 2
        self.mo_occ[ncore:nocc] = 1
        self.mo_occ[nocc:] = 0
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             self.hf_energy, self.mo_energy,
                             self.mo_coeff, self.mo_occ)

    def analyze(self, verbose=logger.DEBUG):
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
        for k, j in enumerate(orbsym):
            if j in irorbcnt:
                irorbcnt[j] += 1
            else:
                irorbcnt[j] = 1
            log.info('MO #%d (%s #%d), energy= %.15g occ= %g',
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
            log.debug(' ** MO coefficients **')
            dump_mat.dump_rec(mol.stdout, mo_coeff, label, molabel, start=1)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        return self.mulliken_meta(mol, dm, s=ovlp_ao, verbose=verbose)

    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import uhf_symm
        if mol is None: mol = self.mol
        if mo_coeff is None: mo_coeff = (self.mo_coeff,self.mo_coeff)
        if mo_occ is None: mo_occ = ((self.mo_occ>0), (self.mo_occ==2))
        return uhf_symm.get_irrep_nelec(mol, mo_coeff, mo_occ)


def _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, title=''):
    nirrep = mol.symm_orb.__len__()
    p0 = 0
    for ir in range(nirrep):
        irname = mol.irrep_name[ir]
        nso = mol.symm_orb[ir].shape[1]
        nocc = (mo_occ[p0:p0+nso]>0).sum()
        if nocc == 0:
            logger.debug(mol, '%s%s nocc = 0', title, irname)
        elif nocc == nso:
            logger.debug(mol, '%s%s nocc = %d  HOMO = %.12g',
                         title, irname, nocc, mo_energy[p0+nocc-1])
        else:
            logger.debug(mol, '%s%s nocc = %d  HOMO = %.12g  LUMO = %.12g',
                         title, irname,
                         nocc, mo_energy[p0+nocc-1], mo_energy[p0+nocc])
            if mo_energy[p0+nocc-1]+1e-3 > elumo:
                logger.warn(mol, '!! %s%s HOMO %.12g > system LUMO %.12g',
                            title, irname, mo_energy[p0+nocc-1], elumo)
            if mo_energy[p0+nocc] < ehomo+1e-3:
                logger.warn(mol, '!! %s%s LUMO %.12g < system HOMO %.12g',
                            title, irname, mo_energy[p0+nocc], ehomo)
        logger.debug(mol, '   mo_energy = %s', mo_energy[p0:p0+nso])
        p0 += nso



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
