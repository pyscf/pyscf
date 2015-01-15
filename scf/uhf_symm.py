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
import pyscf.lib.parameters as param
import pyscf.symm
from pyscf.scf import diis
from pyscf.scf import hf
from pyscf.scf import hf_symm
from pyscf.scf import uhf
from pyscf.scf import _vhf

'''
'''

def analyze(mf, mo_energy=None, mo_occ=None, mo_coeff=None):
    from pyscf.tools import dump_mat
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None: mo_occ = mf.mo_occ
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    mol = mf.mol
    nirrep = mol.symm_orb.__len__()
    if mf.verbose >= param.VERBOSE_INFO:
        tot_sym = 0
        noccsa = []
        noccsb = []
        irlabels = []
        irorbcnt = []
        p0 = 0
        for ir in range(nirrep):
            nso = mol.symm_orb[ir].shape[1]
            nocca = int(mo_occ[0][p0:p0+nso].sum())
            noccb = int(mo_occ[1][p0:p0+nso].sum())
            if (nocca+noccb) % 2:
                tot_sym ^= mol.irrep_id[ir]
            noccsa.append(nocca)
            noccsb.append(noccb)
            irlabels.extend([mol.irrep_name[ir]]*nso)
            irorbcnt.extend(range(nso))
            p0 += nso
        log.info(mf, 'total symmetry = %s', \
                 pyscf.symm.irrep_name(mol.groupname, tot_sym))
        log.info(mf, 'alpha occupancy for each irrep:  '+(' %4s'*nirrep), \
                 *mol.irrep_name)
        log.info(mf, '                                 '+(' %4d'*nirrep), \
                 *noccsa)
        log.info(mf, 'beta  occupancy for each irrep:  '+(' %4s'*nirrep), \
                 *mol.irrep_name)
        log.info(mf, '                                 '+(' %4d'*nirrep), \
                 *noccsb)

    ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                            mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
    log.info(mf, 'multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)

    if mf.verbose >= param.VERBOSE_INFO:
        log.info(mf, '**** MO energy ****')
        idxa = numpy.argsort(mo_energy[0])
        for k, j in enumerate(idxa):
            log.info(mf, 'alpha MO #%d (%s #%d), energy= %.15g occ= %g', \
                     k+1, irlabels[j], irorbcnt[j]+1,
                     mo_energy[0][j], mo_occ[0][j])
        idxb = numpy.argsort(mo_energy[1])
        for k, j in enumerate(idxb):
            log.info(mf, 'beta  MO #%d (%s #%d), energy= %.15g occ= %g', \
                     k+1, irlabels[j], irorbcnt[j]+1,
                     mo_energy[1][j], mo_occ[1][j])

    if mf.verbose >= param.VERBOSE_DEBUG:
        label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
        molabel = []
        for k, j in enumerate(idxa):
            molabel.append('#%-d(%s #%d)' % (k+1, irlabels[j],
                                              irorbcnt[j]+1))
        log.debug(mf, ' ** alpha MO coefficients **')
        dump_mat.dump_rec(mol.stdout, mo_coeff[0][:,idxa], label, molabel, start=1)
        molabel = []
        for k, j in enumerate(idxb):
            molabel.append('#%-d(%s #%d)' % (k+1, irlabels[j],
                                              irorbcnt[j]+1))
        log.debug(mf, ' ** beta MO coefficients **')
        dump_mat.dump_rec(mol.stdout, mo_coeff[1][:,idxb], label, molabel, start=1)

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return mf.mulliken_pop(mol, dm, mf.get_ovlp())

def map_rhf_to_uhf(rhf):
    return uhf.map_rhf_to_uhf(rhf)


class UHF(uhf.UHF):
    '''UHF'''
    def __init__(self, mol):
        uhf.UHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nocc_alpha = {}
        self.irrep_nocc_beta = {}
        self._keys = self._keys.union(['_eri', 'irrep_nocc_alpha','irrep_nocc_beta'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        log.info(self, '%s with symmetry adapted basis', self.__doc__)
        float_irname = []
        fix_na = 0
        fix_nb = 0
        for ir in range(self.mol.symm_orb.__len__()):
            irname = self.mol.irrep_name[ir]
            if irname in self.irrep_nocc_alpha:
                fix_na += self.irrep_nocc_alpha[irname]
            else:
                float_irname.append(irname)
            if irname in self.irrep_nocc_beta:
                fix_nb += self.irrep_nocc_beta[irname]
            else:
                float_irname.append(irname)
        float_irname = set(float_irname)
        if fix_na+fix_nb > 0:
            log.info(self, 'fix %d electrons in irreps:\n' \
                     '   alpha %s,\n   beta  %s', \
                     fix_na+fix_nb, self.irrep_nocc_alpha.items(), \
                     self.irrep_nocc_beta.items())
            if fix_na+fix_nb > self.mol.nelectron \
               or ((fix_na>self.nelectron_alpha) or \
                   (fix_nb+self.nelectron_alpha>self.mol.nelectron)):
                log.error(self, 'number of electrons error in irrep_nocc\n' \
                        '   alpha %s,\n   beta  %s', \
                        self.irrep_nocc_alpha.items(), \
                        self.irrep_nocc_beta.items())
                raise ValueError('irrep_nocc')
        if float_irname:
            log.info(self, '%d free electrons in irreps %s', \
                     self.mol.nelectron-fix_na-fix_nb,
                     ' '.join(float_irname))
        elif fix_na+fix_nb != self.mol.nelectron:
            log.error(self, 'number of electrons error in irrep_nocc \n' \
                    '   alpha %s,\n   beta  %s', \
                    self.irrep_nocc_alpha.items(), \
                    self.irrep_nocc_beta.items())
            raise ValueError('irrep_nocc')

    def build_(self, mol=None):
        for irname in self.irrep_nocc_alpha.keys():
            if irname not in self.mol.irrep_name:
                log.warn(self, '!! No irrep %s', irname)
        for irname in self.irrep_nocc_beta.keys():
            if irname not in self.mol.irrep_name:
                log.warn(self, '!! No irrep %s', irname)
        return uhf.UHF.build_(self, mol)

    def eig(self, h, s):
        nirrep = self.mol.symm_orb.__len__()
        s = pyscf.symm.symmetrize_matrix(s, self.mol.symm_orb)
        ha = pyscf.symm.symmetrize_matrix(h[0], self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = scipy.linalg.eigh(ha[ir], s[ir])
            cs.append(c)
            es.append(e)
        ea = numpy.hstack(es)
        ca = hf_symm.so2ao_mo_coeff(self.mol.symm_orb, cs)

        hb = pyscf.symm.symmetrize_matrix(h[1], self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = scipy.linalg.eigh(hb[ir], s[ir])
            cs.append(c)
            es.append(e)
        eb = numpy.hstack(es)
        cb = hf_symm.so2ao_mo_coeff(self.mol.symm_orb, cs)
        return numpy.array((ea,eb)), (ca,cb)

    def get_occ(self, mo_energy, mo_coeff=None):
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nirrep = mol.symm_orb.__len__()
        idx_ea_left = []
        idx_eb_left = []
        neleca_fix = nelecb_fix = 0
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nocc_alpha:
                n = self.irrep_nocc_alpha[irname]
                mo_occ[0][p0:p0+n] = 1
                neleca_fix += n
            else:
                idx_ea_left.append(range(p0,p0+nso))
            if irname in self.irrep_nocc_beta:
                n = self.irrep_nocc_beta[irname]
                mo_occ[1][p0:p0+n] = 1
                nelecb_fix += n
            else:
                idx_eb_left.append(range(p0,p0+nso))
            p0 += nso

        neleca_float = self.nelectron_alpha - neleca_fix
        nelecb_float = mol.nelectron - self.nelectron_alpha - nelecb_fix
        assert(neleca_float >= 0)
        assert(nelecb_float >= 0)
        if len(idx_ea_left) > 0:
            idx_ea_left = numpy.hstack(idx_ea_left)
            ea_left = mo_energy[0][idx_ea_left]
            ea_sort = numpy.argsort(ea_left)
            occ_idx = idx_ea_left[ea_sort][:neleca_float]
            mo_occ[0][occ_idx] = 1
        if len(idx_eb_left) > 0:
            idx_eb_left = numpy.hstack(idx_eb_left)
            eb_left = mo_energy[1][idx_eb_left]
            eb_sort = numpy.argsort(eb_left)
            occ_idx = idx_eb_left[eb_sort][:nelecb_float]
            mo_occ[1][occ_idx] = 1

        ehomoa = max(mo_energy[0][mo_occ[0]>0 ])
        elumoa = min(mo_energy[0][mo_occ[0]==0])
        ehomob = max(mo_energy[1][mo_occ[1]>0 ])
        elumob = min(mo_energy[1][mo_occ[1]==0])
        noccsa = []
        noccsb = []
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]

            noccsa.append(int(mo_occ[0][p0:p0+nso].sum()))
            noccsb.append(int(mo_occ[1][p0:p0+nso].sum()))
            if ehomoa in mo_energy[0][p0:p0+nso]:
                irhomoa = irname
            if elumoa in mo_energy[0][p0:p0+nso]:
                irlumoa = irname
            if ehomob in mo_energy[1][p0:p0+nso]:
                irhomob = irname
            if elumob in mo_energy[1][p0:p0+nso]:
                irlumob = irname
            p0 += nso

        log.info(self, 'alpha HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                 irhomoa, ehomoa, irlumoa, elumoa)
        log.info(self, 'beta  HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                 irhomob, ehomob, irlumob, elumob)
        if self.verbose >= param.VERBOSE_DEBUG:
            ehomo = max(ehomoa,ehomob)
            elumo = min(elumoa,elumob)
            log.debug(self, 'alpha irrep_nocc = %s', noccsa)
            log.debug(self, 'beta  irrep_nocc = %s', noccsb)
            hf_symm._dump_mo_energy(mol, mo_energy[0], mo_occ[0], ehomo, elumo, 'alpha-')
            hf_symm._dump_mo_energy(mol, mo_energy[1], mo_occ[1], ehomo, elumo, 'beta-')
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
        self.analyze(self.mo_energy, self.mo_occ, self.mo_coeff)

        ea = numpy.hstack(self.mo_energy[0])
        eb = numpy.hstack(self.mo_energy[0])
        oa_sort = numpy.argsort(ea[self.mo_occ[0]>0])
        va_sort = numpy.argsort(ea[self.mo_occ[0]==0])
        ob_sort = numpy.argsort(eb[self.mo_occ[1]>0])
        vb_sort = numpy.argsort(eb[self.mo_occ[1]==0])
        self.mo_energy = (numpy.hstack((ea[self.mo_occ[0]>0 ][oa_sort], \
                                        ea[self.mo_occ[0]==0][va_sort])), \
                          numpy.hstack((eb[self.mo_occ[1]>0 ][ob_sort], \
                                        eb[self.mo_occ[1]==0][vb_sort])))
        ca = self.mo_coeff[0]
        cb = self.mo_coeff[1]
        self.mo_coeff = (numpy.hstack((ca[:,self.mo_occ[0]>0 ][:,oa_sort], \
                                       ca[:,self.mo_occ[0]==0][:,va_sort])), \
                         numpy.hstack((cb[:,self.mo_occ[1]>0 ][:,ob_sort], \
                                       cb[:,self.mo_occ[1]==0][:,vb_sort])))
        nocc_a = int(self.mo_occ[0].sum())
        nocc_b = int(self.mo_occ[1].sum())
        self.mo_occ[0][:nocc_a] = 1
        self.mo_occ[0][nocc_a:] = 0
        self.mo_occ[1][:nocc_b] = 1
        self.mo_occ[1][nocc_b:] = 0
        return self.hf_energy

    def analyze(self, mo_energy=None, mo_occ=None, mo_coeff=None):
        return analyze(self, mo_energy, mo_occ, mo_coeff)


