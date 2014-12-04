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
        self._keys = self._keys | set(['irrep_nocc'])

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
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                      irhomo, ehomo, irlumo, elumo)
            log.debug(self, 'irrep_nocc = %s', noccs)
            dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
        return mo_occ

    def scf(self):
        cput0 = (time.clock(), time.time())
        self.build()
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(self.mol, self, self.conv_threshold)

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.converged)
        self.analyze_scf_result(self.mol, self.mo_energy,
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
        self._keys = self._keys | set(['irrep_nocc_alpha','irrep_nocc_beta'])

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
        mo_ea_left = []
        mo_eb_left = []
        noccsa = []
        noccsb = []
        neleca_fix = nelecb_fix = 0
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]
            if irname in self.irrep_nocc_alpha:
                n = self.irrep_nocc_alpha[irname]
                mo_occ[0][p0:p0+n] = 1
                neleca_fix += n
                noccsa.append(n)
            else:
                noccsa.append(0)
                mo_ea_left.append(mo_energy[0][p0:p0+nso])
            if irname in self.irrep_nocc_beta:
                n = self.irrep_nocc_beta[irname]
                mo_occ[1][p0:p0+n] = 1
                nelecb_fix += n
                noccsb.append(n)
            else:
                noccsb.append(0)
                mo_eb_left.append(mo_energy[1][p0:p0+nso])
            p0 += nso

        nelec_float = mol.nelectron - neleca_fix - nelecb_fix
        assert(nelec_float >= 0)
        if len(mo_ea_left) > 0:
            mo_ea_left = sorted(numpy.hstack(mo_ea_left))
        if len(mo_eb_left) > 0:
            mo_eb_left = sorted(numpy.hstack(mo_eb_left))

# determine how many alpha and beta electrons
        elumoa_float = 1e9
        elumob_float = 1e9
        neleca_float = 0
        nelecb_float = 0
        if nelec_float > 0:
            neleca_float = self.nelectron_alpha - neleca_fix
            nelecb_float = mol.nelectron - self.nelectron_alpha - nelecb_fix
            assert(neleca_float >= 0)
            assert(nelecb_float >= 0)
            if len(mo_ea_left) > 0:
                elumoa_float = mo_ea_left[neleca_float]
            if len(mo_eb_left) > 0:
                elumob_float = mo_eb_left[nelecb_float]

# determine how many alpha and beta occs for all irreps
        ehomoa, irhomoa = (-1e9, None)
        ehomob, irhomob = (-1e9, None)
        elumoa, irlumoa = ( 1e9, None)
        elumob, irlumob = ( 1e9, None)
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]

            if irname in self.irrep_nocc_alpha:
                nocc = self.irrep_nocc_alpha[irname]
            else:
                nocc = int((mo_energy[0][p0:p0+nso]<elumoa_float).sum())
                mo_occ[0][p0:p0+nocc] = 1
                noccsa[ir] = nocc
            if nocc > 0 and mo_energy[0][p0+nocc-1] > ehomoa:
                ehomoa, irhomoa = mo_energy[0][p0+nocc-1], irname
            if nocc < nso and mo_energy[0][p0+nocc] < elumoa:
                elumoa, irlumoa = mo_energy[0][p0+nocc], irname

            if irname in self.irrep_nocc_beta:
                nocc = self.irrep_nocc_beta[irname]
            else:
                nocc = int((mo_energy[1][p0:p0+nso]<elumob_float).sum())
                mo_occ[1][p0:p0+nocc] = 1
                noccsb[ir] = nocc
            if nocc > 0 and mo_energy[1][p0+nocc-1] > ehomob:
                ehomob, irhomob = mo_energy[1][p0+nocc-1], irname
            if nocc < nso and mo_energy[1][p0+nocc] < elumob:
                elumob, irlumob = mo_energy[1][p0+nocc], irname

            p0 += nso

        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'alpha HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                      irhomoa, ehomoa, irlumoa, elumoa)
            log.debug(self, 'beta  HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                      irhomob, ehomob, irlumob, elumob)
            ehomo = max(ehomoa,ehomob)
            elumo = min(elumoa,elumob)
            log.debug(self, 'alpha irrep_nocc = %s', noccsa)
            log.debug(self, 'beta  irrep_nocc = %s', noccsb)
            dump_mo_energy(mol, mo_energy[0], mo_occ[0], ehomo, elumo, 'alpha-')
            dump_mo_energy(mol, mo_energy[1], mo_occ[1], ehomo, elumo, 'beta-')
        return mo_occ

    def scf(self):
        cput0 = (time.clock(), time.time())
        self.build()
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(self.mol, self, self.conv_threshold)

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.converged)
        self.analyze_scf_result(self.mol, self.mo_energy,
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
            if mo_energy[p0+nocc-1] > elumo:
                log.warn(mol, '!! %s%s HOMO %.12g > system LUMO %.12g', \
                         title, irname, mo_energy[p0+nocc-1], elumo)
            if mo_energy[p0+nocc] < ehomo:
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
        self._mo_prev = None

    def build_(self, mol=None):
        # specify alpha,beta for same irreps
        assert(set(self.irrep_nocc_alpha.keys()) == \
               set(self.irrep_nocc_beta.keys()))
        na = sum(self.irrep_nocc_alpha.values())
        nb = sum(self.irrep_nocc_beta.values())
        nopen = self.mol.spin
        assert(na >= nb and nopen >= na-nb)
        return hf.SCF.build_(self, mol)

    def init_diis(self):
        diis_a = diis.SCF_DIIS(self)
        diis_a.space = self.diis_space
        #diis_a.start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                f = diis_a.update(s, d[0]+d[1], f)
            if cycle < self.diis_start_cycle-1:
                f = hf.damping(s, (d[0]+d[1])*.5, f, self.damp_factor)
                f = hf.level_shift(s, (d[0]+d[1])*.5, f, self.level_shift_factor)
            else:
                fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
                f = hf.level_shift(s, (d[0]+d[1])*.5, f, fac)
            return f
        return scf_diis

    # same to RHF.eig
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

    def make_fock(self, h1e, vhf):
# Roothaan's effective fock
# http://www-theor.ch.cam.ac.uk/people/ross/thesis/node15.html
#          |  closed     open    virtual
#  ----------------------------------------
#  closed  |    Fc        Fb       Fc
#  open    |    Fb        Fc       Fa
#  virtual |    Fc        Fa       Fc
# Fc = (Fa+Fb)/2
        fa = h1e + vhf[0]
        fb = h1e + vhf[1]
        ncore = (self.mol.nelectron-self.mol.spin) / 2
        nopen = self.mol.spin
        nocc = ncore + nopen
        s = self.get_ovlp(self.mol)

        if self._mo_prev is None:
            ftmp = (fa + fb) * .5
            _, mo_space = scipy.linalg.eigh(ftmp, s)
            ftmp = reduce(numpy.dot, (mo_space[:,ncore:].T, fa,
                                      mo_space[:,ncore:]))
            mo_space[:,ncore:] = numpy.dot(mo_space[:,ncore:],
                                           scipy.linalg.eigh(ftmp)[1])
        else:
            mo_space = self._mo_prev

        fa = reduce(numpy.dot, (mo_space.T, fa, mo_space))
        fb = reduce(numpy.dot, (mo_space.T, fb, mo_space))
        feff = (fa + fb) * .5
        feff[:ncore,ncore:nocc] = fb[:ncore,ncore:nocc]
        feff[ncore:nocc,:ncore] = fb[ncore:nocc,:ncore]
        feff[nocc:,ncore:nocc] = fa[nocc:,ncore:nocc]
        feff[ncore:nocc,nocc:] = fa[ncore:nocc,nocc:]
        cinv = numpy.dot(mo_space.T, s)
        return reduce(numpy.dot, (cinv.T, feff, cinv))

    def set_occ(self, mo_energy, mo_coeff=None):
        mol = self.mol
        mo_occ = numpy.zeros_like(mo_energy)
        nirrep = mol.symm_orb.__len__()
        mo_e_left = []
        ndoccs = []
        nsoccs = []
        neleca_fix = nelecb_fix = 0
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
                ndoccs.append(ncore)
                nsoccs.append(nocc-ncore)
            else:
                ndoccs.append(0)
                nsoccs.append(0)
                mo_e_left.append(mo_energy[p0:p0+nso])
            p0 += nso

        nelec_float = mol.nelectron - neleca_fix - nelecb_fix
        assert(nelec_float >= 0)
        if len(mo_e_left) > 0:
            mo_e_left = sorted(numpy.hstack(mo_e_left))
            nopen = self.mol.spin - (neleca_fix - nelecb_fix)
            ncore = (nelec_float - nopen)/2
            elumoa_float = mo_e_left[ncore+nopen]
            elumob_float = mo_e_left[ncore]
        else:
            elumoa_float = 1e9
            elumob_float = 1e9

        ehomo, irhomo = (-1e9, None)
        elumo, irlumo = ( 1e9, None)
        p0 = 0
        for ir in range(nirrep):
            irname = mol.irrep_name[ir]
            nso = mol.symm_orb[ir].shape[1]

            if irname in self.irrep_nocc_alpha:
                nocc = self.irrep_nocc_alpha[irname]
            else:
                ncore = int((mo_energy[p0:p0+nso]<elumob_float).sum())
                nocc = int((mo_energy[p0:p0+nso]<elumoa_float).sum())
                mo_occ[p0:p0+ncore] = 2
                mo_occ[p0+ncore:p0+nocc] = 1
                ndoccs[ir] = ncore
                nsoccs[ir] = nocc - ncore
            if nocc > 0 and mo_energy[p0+nocc-1] > ehomo:
                ehomo, irhomo = mo_energy[p0+nocc-1], irname
            if nocc < nso and mo_energy[p0+nocc] < elumo:
                elumo, irlumo = mo_energy[p0+nocc], irname
            p0 += nso

        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'HOMO (%s) = %.15g, LUMO (%s) = %.15g',
                      irhomo, ehomo, irlumo, elumo)
            log.debug(self, 'double occ irrep_nocc = %s', ndoccs)
            log.debug(self, 'single occ irrep_nocc = %s', nsoccs)
            dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo)
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

    def scf(self):
        cput0 = (time.clock(), time.time())
        self.build()
        self.dump_flags()
        self.converged, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = hf.scf_cycle(self.mol, self, self.conv_threshold)

        log.timer(self, 'SCF', *cput0)
        etot = self.dump_final_energy(self.hf_energy, self.converged)
        self.analyze_scf_result(self.mol, self.mo_energy,
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
