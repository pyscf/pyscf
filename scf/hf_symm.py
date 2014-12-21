#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Hartree-Fock
'''

import os
import time
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import pyscf.symm
import diis
import hf
import _vhf



class RHF(hf.RHF):
    '''RHF'''
    def __init__(self, mol):
        hf.RHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nocc = {} # {'ir_name':int,...}
        self._keys = self._keys | set(['_eri', 'irrep_nocc'])

    def dump_flags(self):
        hf.RHF.dump_flags(self)
        log.info(self, '%s with symmetry adpated basis', self.__doc__)
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

    def set_occ(self, mo_energy, mo_coeff=None):
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
                mo_occ[p0:p0+n/2] = 2
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
            elumo_float = mo_e_left[nelec_float/2]

        ehomo, irhomo = (-1e9, None)
        elumo, irlumo = ( 1e9, None)
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nocc:
                nocc = self.irrep_nocc[irname] / 2
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
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'irrep_nocc = %s', noccs)
            dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
        return mo_occ

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())
        mol = self.mol
        self.build(mol)
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(mol, self, self.conv_threshold, init_dm=dm0)

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.converged)
        self.analyze_scf_result(mol, self.mo_energy,
                                self.mo_occ, self.mo_coeff)

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
        return etot

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        nirrep = mol.symm_orb.__len__()
        if self.verbose >= param.VERBOSE_INFO:
            tot_sym = 0
            noccs = []
            irlabels = []
            irorbcnt = []
            p0 = 0
            for ir in range(nirrep):
                nso = mol.symm_orb[ir].shape[1]
                nocc = int(mo_occ[p0:p0+nso].sum())
                #if nocc % 2:
                #    tot_sym ^= mol.irrep_id[ir]
                noccs.append(nocc)
                irlabels.extend([mol.irrep_name[ir]]*nso)
                irorbcnt.extend(range(nso))
                p0 += nso
            log.info(self, 'total symmetry = %s', \
                     pyscf.symm.irrep_name(mol.groupname, tot_sym))
            log.info(self, 'occupancy for each irrep:  ' + (' %4s'*nirrep), \
                     *mol.irrep_name)
            log.info(self, 'double occ                 ' + (' %4d'*nirrep), \
                     *noccs)
            log.info(self, '**** MO energy ****')
            idx = numpy.argsort(mo_energy)
            for k, j in enumerate(idx):
                log.info(self, 'MO #%d (%s #%d), energy= %.15g occ= %g', \
                         k+1, irlabels[j], irorbcnt[j]+1,
                         mo_energy[j], mo_occ[j])

        if self.verbose >= param.VERBOSE_DEBUG:
            import pyscf.tools.dump_mat as dump_mat
            label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
            molabel = []
            for k, j in enumerate(idx):
                molabel.append('#%-d(%s #%d)' % (k+1, irlabels[j],
                                                 irorbcnt[j]+1))
            log.debug(self, ' ** MO coefficients **')
            dump_mat.dump_rec(mol.stdout, mo_coeff, label, molabel, start=1)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        self.mulliken_pop(mol, dm, self.get_ovlp())


class UHF(hf.UHF):
    '''UHF'''
    def __init__(self, mol):
        hf.UHF.__init__(self, mol)
        # number of electrons for each irreps
        self.irrep_nocc_alpha = {}
        self.irrep_nocc_beta = {}
        self._keys = self._keys | set(['_eri', 'irrep_nocc_alpha','irrep_nocc_beta'])

    def dump_flags(self):
        hf.SCF.dump_flags(self)
        log.info(self, '%s with symmetry adpated basis', self.__doc__)
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
        return hf.UHF.build_(self, mol)

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
        ca = so2ao_mo_coeff(self.mol.symm_orb, cs)

        hb = pyscf.symm.symmetrize_matrix(h[1], self.mol.symm_orb)
        cs = []
        es = []
        for ir in range(nirrep):
            e, c = scipy.linalg.eigh(hb[ir], s[ir])
            cs.append(c)
            es.append(e)
        eb = numpy.hstack(es)
        cb = so2ao_mo_coeff(self.mol.symm_orb, cs)
        return numpy.array((ea,eb)), (ca,cb)

    def set_occ(self, mo_energy, mo_coeff=None):
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
            dump_mo_energy(mol, mo_energy[0], mo_occ[0], ehomo, elumo, 'alpha-')
            dump_mo_energy(mol, mo_energy[1], mo_occ[1], ehomo, elumo, 'beta-')
        return mo_occ

    def scf(self, dm0=None):
        cput0 = (time.clock(), time.time())
        mol = self.mol
        self.build(mol)
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(mol, self, self.conv_threshold, init_dm=dm0)

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.converged)
        self.analyze_scf_result(mol, self.mo_energy,
                                self.mo_occ, self.mo_coeff)

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
        return etot

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        nirrep = mol.symm_orb.__len__()
        if self.verbose >= param.VERBOSE_INFO:
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
            log.info(self, 'total symmetry = %s', \
                     pyscf.symm.irrep_name(mol.groupname, tot_sym))
            log.info(self, 'alpha occupancy for each irrep:  '+(' %4s'*nirrep), \
                     *mol.irrep_name)
            log.info(self, '                                 '+(' %4d'*nirrep), \
                     *noccsa)
            log.info(self, 'beta  occupancy for each irrep:  '+(' %4s'*nirrep), \
                     *mol.irrep_name)
            log.info(self, '                                 '+(' %4d'*nirrep), \
                     *noccsb)

        ss, s = hf.spin_square(mol, mo_coeff[0][:,mo_occ[0]>0],
                               mo_coeff[1][:,mo_occ[1]>0])
        log.info(self, 'multiplicity <S^2> = %.8g, 2S+1 = %.8g', ss, s)

        if self.verbose >= param.VERBOSE_INFO:
            log.info(self, '**** MO energy ****')
            idxa = numpy.argsort(mo_energy[0])
            for k, j in enumerate(idxa):
                log.info(self, 'alpha MO #%d (%s #%d), energy= %.15g occ= %g', \
                         k+1, irlabels[j], irorbcnt[j]+1,
                         mo_energy[0][j], mo_occ[0][j])
            idxb = numpy.argsort(mo_energy[1])
            for k, j in enumerate(idxb):
                log.info(self, 'beta  MO #%d (%s #%d), energy= %.15g occ= %g', \
                         k+1, irlabels[j], irorbcnt[j]+1,
                         mo_energy[1][j], mo_occ[1][j])

        if self.verbose >= param.VERBOSE_DEBUG:
            import pyscf.tools.dump_mat as dump_mat
            label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
            molabel = []
            for k, j in enumerate(idxa):
                molabel.append('#%-d(%s #%d)' % (k+1, irlabels[j],
                                                  irorbcnt[j]+1))
            log.debug(self, ' ** alpha MO coefficients **')
            dump_mat.dump_rec(mol.stdout, mo_coeff[0], label, molabel, start=1)
            molabel = []
            for k, j in enumerate(idxb):
                molabel.append('#%-d(%s #%d)' % (k+1, irlabels[j],
                                                  irorbcnt[j]+1))
            log.debug(self, ' ** beta MO coefficients **')
            dump_mat.dump_rec(mol.stdout, mo_coeff[1], label, molabel, start=1)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        self.mulliken_pop(mol, dm, self.get_ovlp())


def dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, title=''):
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

def so2ao_mo_coeff(so, irrep_mo_coeff):
    return numpy.hstack([numpy.dot(so[ir],irrep_mo_coeff[ir]) \
                         for ir in range(so.__len__())])

def map_rhf_to_uhf(mol, rhf):
    assert(isinstance(rhf, RHF))
    uhf = UHF(mol)
    uhf.verbose               = rhf.verbose
    uhf.mo_energy             = numpy.array((rhf.mo_energy,rhf.mo_energy))
    uhf.mo_coeff              = numpy.array((rhf.mo_coeff,rhf.mo_coeff))
    uhf.mo_occ                = numpy.array((rhf.mo_occ,rhf.mo_occ))
    uhf.hf_energy             = rhf.hf_energy
    uhf.diis_space            = rhf.diis_space
    uhf.diis_start_cycle      = rhf.diis_start_cycle
    uhf.damp_factor           = rhf.damp_factor
    uhf.level_shift_factor    = rhf.level_shift_factor
    uhf.converged             = rhf.converged
    uhf.direct_scf            = rhf.direct_scf
    uhf.direct_scf_threshold  = rhf.direct_scf_threshold

    uhf.chkfile               = rhf.chkfile
    uhf.stdout                = rhf.stdout
    uhf.conv_threshold        = rhf.conv_threshold
    uhf.max_cycle             = rhf.max_cycle
    return uhf


class ROHF(UHF):
    '''ROHF'''
    def __init__(self, mol):
        UHF.__init__(self, mol)
        self.DIIS = diis.SCF_DIIS
# use _irrep_doccs and _irrep_soccs help self.eig to compute orbital energy,
# do not overwrite them
        self._irrep_doccs = []
        self._irrep_soccs = []
# The _core_mo_energy is the orbital energy to help set_occ find doubly
# occupied core orbitals
        self._core_mo_energy = None
        self._open_mo_energy = None
        self._keys = self._keys | set(['_irrep_doccs', '_irrep_soccs',
                                       '_core_mo_energy', '_open_mo_energy'])

    def build_(self, mol=None):
        # specify alpha,beta for same irreps
        assert(set(self.irrep_nocc_alpha.keys()) == \
               set(self.irrep_nocc_beta.keys()))
        na = sum(self.irrep_nocc_alpha.values())
        nb = sum(self.irrep_nocc_beta.values())
        nopen = self.mol.spin
        assert(na >= nb and nopen >= na-nb)
        return UHF.build_(self, mol)

    # same to RHF.eig
    def eig(self, h, s):
        ncore = (self.mol.nelectron-self.mol.spin) / 2
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

    def make_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
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
        ncore = (self.mol.nelectron-self.mol.spin) / 2
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

    def set_occ(self, mo_energy, mo_coeff=None):
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
            ncore = (nelec_float - nopen)/2
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
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'double occ irrep_nocc = %s', ndoccs)
            log.debug(self, 'single occ irrep_nocc = %s', nsoccs)
            dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
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
                = hf.scf_cycle(mol, self, self.conv_threshold, init_dm=dm0)

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.converged)
        self.analyze_scf_result(mol, self.mo_energy,
                                self.mo_occ, self.mo_coeff)

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
        return etot

    def analyze_scf_result(self, mol, mo_energy, mo_occ, mo_coeff):
        nirrep = mol.symm_orb.__len__()
        if self.verbose >= param.VERBOSE_INFO:
            tot_sym = 0
            noccs = []
            irlabels = []
            irorbcnt = []
            ndoccs = []
            nsoccs = []
            p0 = 0
            for ir in range(nirrep):
                nso = mol.symm_orb[ir].shape[1]
                ndocc = (mo_occ[p0:p0+nso]==2).sum()
                nsocc = int(mo_occ[p0:p0+nso].sum()) - ndocc*2
                if nsocc % 2:
                    tot_sym ^= mol.irrep_id[ir]
                ndoccs.append(ndocc)
                nsoccs.append(nsocc)
                irlabels.extend([mol.irrep_name[ir]]*nso)
                irorbcnt.extend(range(nso))
                p0 += nso
            log.info(self, 'total symmetry = %s', \
                     pyscf.symm.irrep_name(mol.groupname, tot_sym))
            log.info(self, 'occupancy for each irrep:  ' + (' %4s'*nirrep), \
                     *mol.irrep_name)
            log.info(self, 'double occ                 ' + (' %4d'*nirrep), \
                     *ndoccs)
            log.info(self, 'single occ                 ' + (' %4d'*nirrep), \
                     *nsoccs)
            log.info(self, '**** MO energy ****')
            idx = numpy.argsort(mo_energy)
            for k, j in enumerate(idx):
                log.info(self, 'MO #%d (%s #%d), energy= %.15g occ= %g', \
                         k+1, irlabels[j], irorbcnt[j]+1,
                         mo_energy[j], mo_occ[j])

        if self.verbose >= param.VERBOSE_DEBUG:
            import pyscf.tools.dump_mat as dump_mat
            label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
            molabel = []
            for k, j in enumerate(idx):
                molabel.append('#%-d(%s #%d)' % (k+1, irlabels[j],
                                                 irorbcnt[j]+1))
            log.debug(self, ' ** MO coefficients **')
            dump_mat.dump_rec(mol.stdout, mo_coeff, label, molabel, start=1)

        dm = self.make_rdm1(mo_coeff, mo_occ)
        self.mulliken_pop(mol, dm, self.get_ovlp())



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
