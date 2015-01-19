#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.lib.logger as log
from pyscf.lib import logger
import pyscf.symm
from pyscf.scf import diis
from pyscf.scf import hf
from pyscf.scf import _vhf

'''
'''


# mo_coeff, mo_occ, mo_energy are all in nosymm representation

def analyze(mf, verbose=logger.DEBUG):
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
    log.info('total symmetry = %s', \
             pyscf.symm.irrep_name(mol.groupname, tot_sym))
    log.info('occupancy for each irrep:  ' + (' %4s'*nirrep), \
             *mol.irrep_name)
    log.info('double occ                 ' + (' %4d'*nirrep), \
             *noccs)
    log.info('**** MO energy ****')
    irorbcnt = [0] * 8
    irname_full = [0] * 8
    for k,ir in enumerate(mol.irrep_id):
        irname_full[ir] = mol.irrep_name[k]
    for k, j in enumerate(orbsym):
        irorbcnt[j] += 1
        log.info('MO #%d (%s #%d), energy= %.15g occ= %g', \
                 k+1, irname_full[j], irorbcnt[j], mo_energy[k], mo_occ[k])

    if verbose >= logger.DEBUG:
        label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
        molabel = []
        irorbcnt = [0] * 8
        for k, j in enumerate(orbsym):
            irorbcnt[j] += 1
            molabel.append('#%-d(%s #%d)' % (k+1, irname_full[j], irorbcnt[j]))
        log.debug(' ** MO coefficients **')
        dump_mat.dump_rec(mol.stdout, mo_coeff, label, molabel, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_pop(mol, dm, mf.get_ovlp(), verbose)

def so2ao_mo_coeff(so, irrep_mo_coeff):
    return numpy.hstack([numpy.dot(so[ir],irrep_mo_coeff[ir]) \
                         for ir in range(so.__len__())])


class RHF(hf.RHF):
    '''RHF'''
    def __init__(self, mol):
        hf.RHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nocc = {} # {'ir_name':int,...}
        self._keys = self._keys.union(['irrep_nocc'])

    def dump_flags(self):
        hf.RHF.dump_flags(self)
        log.info(self, '%s with symmetry adapted basis', self.__doc__)
        float_irname = []
        fix_ne = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if irname in self.irrep_nocc:
                fix_ne += self.irrep_nocc[irname]
            else:
                float_irname.append(irname)
        if fix_ne > 0:
            log.info(self, 'fix %d electrons in irreps %s', \
                     fix_ne, self.irrep_nocc.items())
            if fix_ne > self.mol.nelectron:
                log.error(self, 'number of electrons error in irrep_nocc %s', \
                          self.irrep_nocc.items())
                raise ValueError('irrep_nocc')
        if float_irname:
            log.info(self, '%d free electrons in irreps %s', \
                     self.mol.nelectron-fix_ne, ' '.join(float_irname))
        elif fix_ne != self.mol.nelectron:
            log.error(self, 'number of electrons error in irrep_nocc %s', \
                      self.irrep_nocc.items())
            raise ValueError('irrep_nocc')

    def build_(self, mol=None):
        for irname in self.irrep_nocc.keys():
            if irname not in self.mol.irrep_name:
                log.warn(self, '!! No irrep %s', irname)
        return hf.RHF.build_(self, mol)

    def eig(self, h, s):
        nirrep = self.mol.symm_orb.__len__()
        h = pyscf.symm.symmetrize_matrix(h, self.mol.symm_orb)
        s = pyscf.symm.symmetrize_matrix(s, self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = scipy.linalg.eigh(h[ir], s[ir])
            cs.append(c)
            es.append(e)
        e = numpy.hstack(es)
        c = so2ao_mo_coeff(self.mol.symm_orb, cs)
        return e, c

    def get_occ(self, mo_energy, mo_coeff=None):
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nirrep = mol.symm_orb.__len__()
        mo_e_left = []
        noccs = []
        nelec_fix = 0
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nocc:
                n = self.irrep_nocc[irname]
                mo_occ[p0:p0+n//2] = 2
                nelec_fix += n
                noccs.append(n)
            else:
                noccs.append(0)
                mo_e_left.append(mo_energy[p0:p0+nso])
            p0 += nso
        nelec_float = mol.nelectron - nelec_fix
        assert(nelec_float >= 0)
        if nelec_float > 0:
            mo_e_left = sorted(numpy.hstack(mo_e_left))
            elumo_float = mo_e_left[nelec_float//2]

        ehomo, irhomo = (-1e9, None)
        elumo, irlumo = ( 1e9, None)
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nocc:
                nocc = self.irrep_nocc[irname] // 2
            else:
                nocc = int((mo_energy[p0:p0+nso]<elumo_float).sum())
                mo_occ[p0:p0+nocc] = 2
                noccs[ir] = nocc
            if nocc > 0 and mo_energy[p0+nocc-1] > ehomo:
                ehomo, irhomo = mo_energy[p0+nocc-1], irname
            if nocc < nso and mo_energy[p0+nocc] < elumo:
                elumo, irlumo = mo_energy[p0+nocc], irname
            p0 += nso
        log.info(self, 'HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                 irhomo, ehomo, irlumo, elumo)
        if self.verbose >= logger.DEBUG:
            log.debug(self, 'irrep_nocc = %s', noccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
        return mo_occ

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())
        mol = self.mol
        self.build(mol)
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.kernel(self, self.conv_tol, init_dm=dm0)

        log.timer(self, 'SCF', *cput0)
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
        if self.verbose >= logger.INFO:
            self.analyze(self.verbose)
        return self.hf_energy

    def analyze(self, verbose=logger.DEBUG):
        return analyze(self, verbose)


class ROHF(hf.ROHF):
    '''ROHF'''
    def __init__(self, mol):
        hf.ROHF.__init__(self, mol)
        self.irrep_nocc_alpha = {}
        self.irrep_nocc_beta = {}
# use _irrep_doccs and _irrep_soccs help self.eig to compute orbital energy,
# do not overwrite them
        self._irrep_doccs = []
        self._irrep_soccs = []
# The _core_mo_energy is the orbital energy to help get_occ find doubly
# occupied core orbitals
        self._core_mo_energy = None
        self._open_mo_energy = None
        self._keys = self._keys.union(['irrep_nocc_alpha','irrep_nocc_beta',
                                       '_irrep_doccs', '_irrep_soccs',
                                       '_core_mo_energy', '_open_mo_energy'])

    def dump_flags(self):
        hf.ROHF.dump_flags(self)
        log.info(self, '%s with symmetry adapted basis', self.__doc__)
        float_irname = []
        fix_na = 0
        fix_nb = 0
#FIXME        for ir in range(self.mol.symm_orb.__len__()):
#FIXME            irname = self.mol.irrep_name[ir]
#FIXME            if irname in self.irrep_nocc_alpha:
#FIXME                fix_na += self.irrep_nocc_alpha[irname]
#FIXME            else:
#FIXME                float_irname.append(irname)
#FIXME            if irname in self.irrep_nocc_beta:
#FIXME                fix_nb += self.irrep_nocc_beta[irname]
#FIXME            else:
#FIXME                float_irname.append(irname)
#FIXME        float_irname = set(float_irname)
#FIXME        if fix_na+fix_nb > 0:
#FIXME            log.info(self, 'fix %d electrons in irreps:\n' \
#FIXME                     '   alpha %s,\n   beta  %s', \
#FIXME                     fix_na+fix_nb, self.irrep_nocc_alpha.items(), \
#FIXME                     self.irrep_nocc_beta.items())
#FIXME            if fix_na+fix_nb > self.mol.nelectron \
#FIXME               or ((fix_na>self.nelectron_alpha) or \
#FIXME                   (fix_nb+self.nelectron_alpha>self.mol.nelectron)):
#FIXME                log.error(self, 'number of electrons error in irrep_nocc\n' \
#FIXME                        '   alpha %s,\n   beta  %s', \
#FIXME                        self.irrep_nocc_alpha.items(), \
#FIXME                        self.irrep_nocc_beta.items())
#FIXME                raise ValueError('irrep_nocc')
#FIXME        if float_irname:
#FIXME            log.info(self, '%d free electrons in irreps %s', \
#FIXME                     self.mol.nelectron-fix_na-fix_nb,
#FIXME                     ' '.join(float_irname))
#FIXME        elif fix_na+fix_nb != self.mol.nelectron:
#FIXME            log.error(self, 'number of electrons error in irrep_nocc \n' \
#FIXME                    '   alpha %s,\n   beta  %s', \
#FIXME                    self.irrep_nocc_alpha.items(), \
#FIXME                    self.irrep_nocc_beta.items())
#FIXME            raise ValueError('irrep_nocc')

    def build_(self, mol=None):
        # specify alpha,beta for same irreps
        assert(set(self.irrep_nocc_alpha.keys()) == \
               set(self.irrep_nocc_beta.keys()))
        na = sum(self.irrep_nocc_alpha.values())
        nb = sum(self.irrep_nocc_beta.values())
        nopen = self.mol.spin
        assert(na >= nb and nopen >= na-nb)
        for irname in self.irrep_nocc_alpha.keys():
            if irname not in self.mol.irrep_name:
                log.warn(self, '!! No irrep %s', irname)
        for irname in self.irrep_nocc_beta.keys():
            if irname not in self.mol.irrep_name:
                log.warn(self, '!! No irrep %s', irname)
        return hf.RHF.build_(self, mol)

#TODO:    def dump_flags(self):
#TODO:        pass

    # same to RHF.eig
    def eig(self, h, s):
        ncore = (self.mol.nelectron-self.mol.spin) // 2
        nopen = self.mol.spin
        nocc = ncore + nopen
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
            e, c = scipy.linalg.eigh(h[ir], s[ir])
            ecore.append(e.copy())
            eopen.append(numpy.einsum('ik,ik->k', c, numpy.dot(fa[ir], c)))
            if len(self._irrep_doccs) > 0:
                ncore = self._irrep_doccs[ir]
                ea = eopen[ir][ncore:]
                idx = ea.argsort()
                e[ncore:] = ea[idx]
                c[:,ncore:] = c[:,ncore:][:,idx]
            elif self.mol.irrep_name[ir] in self.irrep_nocc_beta:
                ncore = self.irrep_nocc_beta[self.mol.irrep_name[ir]]
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

    def get_occ(self, mo_energy, mo_coeff=None):
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
            if irname in self.irrep_nocc_alpha:
                ncore = self.irrep_nocc_beta[irname]
                nocc = self.irrep_nocc_alpha[irname]
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

        log.info(self, 'HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                 irhomo, ehomo, irlumo, elumo)
        if self.verbose >= logger.DEBUG:
            log.debug(self, 'double occ irrep_nocc = %s', ndoccs)
            log.debug(self, 'single occ irrep_nocc = %s', nsoccs)
            _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
            p0 = 0
            for ir in range(nirrep):
                irname = mol.irrep_name[ir]
                nso = mol.symm_orb[ir].shape[1]
                log.debug1(self, '_core_mo_energy of %s = %s',
                           irname, self._core_mo_energy[p0:p0+nso])
                log.debug1(self, '_open_mo_energy of %s = %s',
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
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.kernel(self, self.conv_tol, init_dm=dm0)

        log.timer(self, 'SCF', *cput0)
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

        if self.verbose >= logger.INFO:
            self.analyze(self.verbose)
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
        log.info('total symmetry = %s', \
                 pyscf.symm.irrep_name(mol.groupname, tot_sym))
        log.info('occupancy for each irrep:  ' + (' %4s'*nirrep), \
                 *mol.irrep_name)
        log.info('double occ                 ' + (' %4d'*nirrep), \
                 *ndoccs)
        log.info('single occ                 ' + (' %4d'*nirrep), \
                 *nsoccs)
        log.info('**** MO energy ****')
        irorbcnt = [0] * 8
        irname_full = [0] * 8
        for k,ir in enumerate(mol.irrep_id):
            irname_full[ir] = mol.irrep_name[k]
        for k, j in enumerate(orbsym):
            irorbcnt[j] += 1
            log.info('MO #%d (%s #%d), energy= %.15g occ= %g', \
                     k+1, irname_full[j], irorbcnt[j], mo_energy[k], mo_occ[k])

        if verbose >= logger.DEBUG:
            label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
            molabel = []
            irorbcnt = [0] * 8
            for k, j in enumerate(orbsym):
                irorbcnt[j] += 1
                molabel.append('#%-d(%s #%d)' % (k+1, irname_full[j], irorbcnt[j]))
            log.debug(' ** MO coefficients **')
            dump_mat.dump_rec(mol.stdout, mo_coeff, label, molabel, start=1)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        return self.mulliken_pop(mol, dm, self.get_ovlp(), verbose)


def _dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, title=''):
    nirrep = mol.symm_orb.__len__()
    p0 = 0
    for ir in range(nirrep):
        irname = mol.irrep_name[ir]
        nso = mol.symm_orb[ir].shape[1]
        nocc = (mo_occ[p0:p0+nso]>0).sum()
        if nocc == 0:
            log.debug(mol, '%s%s nocc = 0', title, irname)
        elif nocc == nso:
            log.debug(mol, '%s%s nocc = %d, HOMO = %.12g,', \
                      title, irname, \
                      nocc, mo_energy[p0+nocc-1])
        else:
            log.debug(mol, '%s%s nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                      title, irname, \
                      nocc, mo_energy[p0+nocc-1], mo_energy[p0+nocc])
            if mo_energy[p0+nocc-1]+1e-3 > elumo:
                log.warn(mol, '!! %s%s HOMO %.12g > system LUMO %.12g', \
                         title, irname, mo_energy[p0+nocc-1], elumo)
            if mo_energy[p0+nocc] < ehomo+1e-3:
                log.warn(mol, '!! %s%s LUMO %.12g < system HOMO %.12g', \
                         title, irname, mo_energy[p0+nocc], ehomo)
        log.debug(mol, '   mo_energy = %s', mo_energy[p0:p0+nso])
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
    #method.irrep_nocc['B2u'] = 2
    energy = method.scf()
    print(energy)
