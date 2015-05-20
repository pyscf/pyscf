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
import pyscf.symm
from pyscf.scf import hf



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
    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                       mo_coeff)
    orbsym = numpy.array(orbsym)
    tot_sym = 0
    noccs = [sum(orbsym[mo_occ>0]==ir) for ir in mol.irrep_id]
    log.info('total symmetry = %s',
             pyscf.symm.irrep_name(mol.groupname, tot_sym))
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
        label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
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
    return mf.mulliken_pop(mol, dm, mf.get_ovlp(), log)

def get_irrep_nelec(mol, mo_coeff, mo_occ):
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
    orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                       mo_coeff)
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
        h = pyscf.symm.symmetrize_matrix(h, self.mol.symm_orb)
        s = pyscf.symm.symmetrize_matrix(s, self.mol.symm_orb)
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
        logger.info(self, 'HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                    irhomo, ehomo, irlumo, elumo)
        if self.verbose >= logger.DEBUG:
            logger.debug(self, 'irrep_nelec = %s', noccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
        return mo_occ

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())
        mol = self.mol
        self.build(mol)
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_coeff, self.mo_occ \
                = hf.kernel(self, self.conv_tol, dm0=dm0)

        logger.timer(self, 'SCF', *cput0)
        self.dump_energy(self.hf_energy, self.converged)

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

# analyze at last, in terms of the ordered orbital energies
        #if self.verbose >= logger.INFO:
        #    self.analyze(self.verbose)
        return self.hf_energy

    def analyze(self, verbose=logger.DEBUG):
        return analyze(self, verbose)

    def get_irrep_nelec(self, mol=None, mo_coeff=None, mo_occ=None):
        if mol is None: mol = self.mol
        if mo_occ is None: mo_occ = self.mo_occ
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return get_irrep_nelec(mol, mo_coeff, mo_occ)


class ROHF(hf.ROHF):
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
        hf.ROHF.__init__(self, mol)
        self.irrep_nelec = {}
# use _irrep_doccs and _irrep_soccs help self.eig to compute orbital energy,
# do not overwrite them
        self._irrep_doccs = []
        self._irrep_soccs = []
# The _core_mo_energy is the orbital energy to help get_occ find doubly
# occupied core orbitals
        self._core_mo_energy = None
        self._open_mo_energy = None
        self._keys = self._keys.union(['irrep_nelec'])

    def dump_flags(self):
        hf.ROHF.dump_flags(self)
        logger.info(self, '%s with symmetry adapted basis',
                    self.__class__.__name__)
#TODO: improve the sainity check
        float_irname = []
        fix_na = 0
        fix_nb = 0
        nelectron_alpha = (self.mol.nelectron+self.mol.spin) // 2
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if irname in self.irrep_nelec:
                fix_na += self.irrep_nelec[irname][0]
                fix_nb += self.irrep_nelec[irname][1]
            else:
                float_irname.append(irname)
        float_irname = set(float_irname)
        if fix_na+fix_nb > 0:
            logger.info(self, 'fix %d electrons in irreps: %s',
                        fix_na+fix_nb, str(self.irrep_nelec.items()))
            if ((fix_na+fix_nb > self.mol.nelectron) or
                (fix_na>nelectron_alpha) or
                (fix_nb+nelectron_alpha>self.mol.nelectron)):
                logger.error(self, 'electron number error in irrep_nelec %s',
                             self.irrep_nelec.items())
                raise ValueError('irrep_nelec')
        if float_irname:
            logger.info(self, '%d free electrons in irreps %s',
                        self.mol.nelectron-fix_na-fix_nb,
                        ' '.join(float_irname))
        elif fix_na+fix_nb != self.mol.nelectron:
            logger.error(self, 'electron number error in irrep_nelec %d',
                         self.irrep_nelec.items())
            raise ValueError('irrep_nelec')

    def build_(self, mol=None):
        # specify alpha,beta for same irreps
        na = sum([x[0] for x in self.irrep_nelec.values()])
        nb = sum([x[1] for x in self.irrep_nelec.values()])
        nopen = self.mol.spin
        assert(na >= nb and nopen >= na-nb)
        for irname in self.irrep_nelec.keys():
            if irname not in self.mol.irrep_name:
                logger.warn(self, '!! No irrep %s', irname)
        return hf.RHF.build_(self, mol)

    # same to RHF.eig
    def eig(self, h, s):
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        feff, fa, fb = h
        nirrep = self.mol.symm_orb.__len__()
        fa = pyscf.symm.symmetrize_matrix(fa, self.mol.symm_orb)
        h = pyscf.symm.symmetrize_matrix(feff, self.mol.symm_orb)
        s = pyscf.symm.symmetrize_matrix(s, self.mol.symm_orb)
        cs = []
        es = []
        ecore = []
        eopen = []
        for ir in range(nirrep):
            e, c = hf.SCF.eig(self, h[ir], s[ir])
            ecore.append(e.copy())
            eopen.append(numpy.einsum('ik,ik->k', c, numpy.dot(fa[ir], c)))
            if len(self._irrep_doccs) > 0:
                ncore = self._irrep_doccs[ir]
                ea = eopen[ir][ncore:]
                idx = ea.argsort()
                e[ncore:] = ea[idx]
                c[:,ncore:] = c[:,ncore:][:,idx]
            elif self.mol.irrep_name[ir] in self.irrep_nelec:
                ncore = self.irrep_nelec[self.mol.irrep_name[ir]][1]
                ea = eopen[ir][ncore:]
                idx = ea.argsort()
                e[ncore:] = ea[idx]
                c[:,ncore:] = c[:,ncore:][:,idx]
            cs.append(c)
            es.append(e)
        self._core_mo_energy = numpy.hstack(ecore)
        self._open_mo_energy = numpy.hstack(eopen)
        e = numpy.hstack(es)
        c = so2ao_mo_coeff(self.mol.symm_orb, cs)
        return e, c

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
# Roothaan's effective fock
# http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html
#          |  closed     open    virtual
#  ----------------------------------------
#  closed  |    Fc        Fb       Fc
#  open    |    Fb        Fc       Fa
#  virtual |    Fc        Fa       Fc
# Fc = (Fa+Fb)/2
        fa0 = h1e + vhf[0]
        fb0 = h1e + vhf[1]
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        dmsf = dm[0]+dm[1]
        sds = -reduce(numpy.dot, (s1e, dmsf, s1e))
        _, mo_space = scipy.linalg.eigh(sds, s1e)
        fa = reduce(numpy.dot, (mo_space.T, fa0, mo_space))
        fb = reduce(numpy.dot, (mo_space.T, fb0, mo_space))
        feff = (fa + fb) * .5
        feff[:ncore,ncore:nocc] = fb[:ncore,ncore:nocc]
        feff[ncore:nocc,:ncore] = fb[ncore:nocc,:ncore]
        feff[nocc:,ncore:nocc] = fa[nocc:,ncore:nocc]
        feff[ncore:nocc,nocc:] = fa[ncore:nocc,nocc:]
        cinv = numpy.dot(mo_space.T, s1e)
        f = reduce(numpy.dot, (cinv.T, feff, cinv))

        if 0 <= cycle < self.diis_start_cycle-1:
            f = hf.damping(s1e, dmsf*.5, f, self.damp_factor)
            f = hf.level_shift(s1e, dmsf*.5, f, self.level_shift_factor)
        elif 0 <= cycle:
            # decay the level_shift_factor
            fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
            f = hf.level_shift(s1e, dmsf*.5, f, fac)
        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dmsf, f)
        return (f, fa0, fb0)

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
                ncore = self.irrep_nelec[irname][1]
                nocc = self.irrep_nelec[irname][0]
                mo_occ[p0:p0+ncore] = 2
                mo_occ[p0+ncore:p0+nocc] = 1
                neleca_fix += nocc
                nelecb_fix += ncore
            else:
                float_idx.append(range(p0,p0+nso))
            p0 += nso

        nelec_float = mol.nelectron - neleca_fix - nelecb_fix
        assert(nelec_float >= 0)
        if len(float_idx) > 0:
            float_idx = numpy.hstack(float_idx)
            nopen = mol.spin - (neleca_fix - nelecb_fix)
            ncore = (nelec_float - nopen)//2
            ecore = self._core_mo_energy[float_idx]
            core_sort = numpy.argsort(ecore)
            core_idx = float_idx[core_sort][:ncore]
            open_idx = float_idx[core_sort][ncore:]
            eopen = self._open_mo_energy[open_idx]
            open_sort = numpy.argsort(eopen)
            open_idx = open_idx[open_sort]
            mo_occ[core_idx] = 2
            mo_occ[open_idx[:nopen]] = 1

        ehomo = max(mo_energy[mo_occ>0])
        elumo = min(mo_energy[mo_occ==0])
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

        logger.info(self, 'HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                    irhomo, ehomo, irlumo, elumo)
        if self.verbose >= logger.DEBUG:
            logger.debug(self, 'double occ irrep_nelec = %s', ndoccs)
            logger.debug(self, 'single occ irrep_nelec = %s', nsoccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
            p0 = 0
            for ir in range(nirrep):
                irname = mol.irrep_name[ir]
                nso = mol.symm_orb[ir].shape[1]
                logger.debug1(self, '_core_mo_energy of %s = %s',
                              irname, self._core_mo_energy[p0:p0+nso])
                logger.debug1(self, '_open_mo_energy of %s = %s',
                              irname, self._open_mo_energy[p0:p0+nso])
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

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())
        mol = self.mol
        self.build(mol)
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_coeff, self.mo_occ \
                = hf.kernel(self, self.conv_tol, dm0=dm0)

        logger.timer(self, 'SCF', *cput0)
        self.dump_energy(self.hf_energy, self.converged)

        # sort MOs wrt orbital energies, it should be done last.
        o_sort = numpy.argsort(self.mo_energy[self.mo_occ>0])
        v_sort = numpy.argsort(self.mo_energy[self.mo_occ==0])
        self.mo_energy = numpy.hstack((self.mo_energy[self.mo_occ>0][o_sort], \
                                       self.mo_energy[self.mo_occ==0][v_sort]))
        self.mo_coeff = numpy.hstack((self.mo_coeff[:,self.mo_occ>0][:,o_sort], \
                                      self.mo_coeff[:,self.mo_occ==0][:,v_sort]))
        nocc = len(o_sort)
        self.mo_occ[:nocc] = self.mo_occ[self.mo_occ>0][o_sort]
        self.mo_occ[nocc:] = 0

        #if self.verbose >= logger.INFO:
        #    self.analyze(self.verbose)
        return self.hf_energy

    def analyze(self, verbose=logger.DEBUG):
        from pyscf.tools import dump_mat
        mo_energy = self.mo_energy
        mo_occ = self.mo_occ
        mo_coeff = self.mo_coeff
        log = logger.Logger(self.stdout, verbose)
        mol = self.mol
        nirrep = len(mol.irrep_id)
        orbsym = pyscf.symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                           mo_coeff)
        orbsym = numpy.array(orbsym)
        tot_sym = 0
        ndoccs = []
        nsoccs = []
        for k,ir in enumerate(mol.irrep_id):
            ndoccs.append(sum(orbsym[mo_occ==2] == ir))
            nsoccs.append(sum(orbsym[mo_occ==1] == ir))
            if nsoccs[k] % 2:
                tot_sym ^= ir
        if mol.groupname in ('Dooh', 'Coov'):
            log.info('TODO: total symmetry for %s', mol.groupname)
        else:
            log.info('total symmetry = %s',
                     pyscf.symm.irrep_name(mol.groupname, tot_sym))
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
            label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
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
        return self.mulliken_pop(mol, dm, self.get_ovlp(), verbose)

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
            logger.debug(mol, '%s%s nocc = %d, HOMO = %.12g,',
                         title, irname, nocc, mo_energy[p0+nocc-1])
        else:
            logger.debug(mol, '%s%s nocc = %d, HOMO = %.12g, LUMO = %.12g,',
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
    #method.irrep_nelec['B2u'] = 2
    energy = method.scf()
    print(energy)
