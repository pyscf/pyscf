#!/usr/bin/env python
# -*- coding: utf-8
#
# File: dhf_dkb.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Dirac Hartree-Fock
'''

__author__ = 'Qiming Sun <osirpt.sun@gmail.com>'
__version__ = '$ 0.1 $'

import ctypes
import numpy
import copy
#import scipy.linalg.flapack as lapack
from pyscf import gto
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import pycint
import hf
import dhf
import chkfile

_cint = hf._cint

def cdouble_to_cmplx(arr):
    return numpy.array(arr)[0::2] + numpy.array(arr)[1::2] * 1j

class UHF(dhf.UHF):
    def __init__(self, mol):
        dhf.UHF.__init__(self, mol)
        self.scf_threshold = 1e-8
        self.make_dkb_env(mol)
        self.init_guess_method = self._init_guess_from_rkb
        self.rkb_chkfile = None
        self.diis_start_cycle = 1

    def make_dkb_env(self, mol):
        '''update mol._atm, mol._bas, mol._env'''
        pmol = mol.copy()
        self.nbasL = mol.nbas
        for ib in range(mol.nbas):
            atom_id, angl, nprim, nctr, kappa, \
                    ptr_exp, ptr_coeff, gauge_method = pmol._bas[ib]
            if angl == 0:
                pmol._bas.append((atom_id, 1, nprim, nctr, 1, \
                                  ptr_exp, ptr_coeff, gauge_method))
            else:
                if kappa <= 0:
                    pmol._bas.append((atom_id, angl+1, nprim, nctr, angl+1, \
                                      ptr_exp, ptr_coeff, gauge_method))
                if kappa >= 0:
                    pmol._bas.append((atom_id, angl-1, nprim, nctr, -angl, \
                                      ptr_exp, ptr_coeff, gauge_method))
        pmol.nbas = pmol._bas.__len__()
        self.dkb_mol = pmol

    def init_guess_method(self, mol):
        if self._init_guess.lower() == '1e':
            self.init_guess_method = self._init_guess_by_1e
        elif self._init_guess.lower() == 'chkfile':
            self.init_guess_method = self._init_guess_by_chkfile
        elif self._init_guess.lower() == 'rkb':
            self.init_guess_method = self._init_guess_from_rkb
        else:
            raise KeyError('Unknown init guess.')

    def _init_guess_from_rkb(self, mol):
        mol.fout.flush()
        log.debug(self, 'DKB use RKB as initial guess')
        mrkb = dhf.UHF(mol)
        if self.rkb_chkfile is not None:
            mrkb.chkfile = self.rkb_chkfile
            mrkb.set_init_guess('chkfile')
        scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
                = mrkb.scf_cycle(mol, 1e-5, False)
        c = numpy.dot(self.project_rkb_to_dkb(mol), mo_coeff)
        dm = mrkb.calc_den_mat(c, mo_occ)
        log.debug(self, 'DKB initial guess with hf_energy = %.12g', hf_energy)
        mol.fout.flush()
        return hf_energy, dm

    def project_rkb_to_dkb(self, mol):
        '''|DKB> P, P = S_{DKB}^{-1} <DKB|RKB>
        <DKB|RKB> = [ <g|g>         <sp g|sp g>/4c^2 ]
                    [ -<sp f|g>/2c  <f|sp g>/2c      ]
        '''
        n2c = mol.num_2C_function()
        n4c = n2c * 2
        c = mol.light_speed

        s1e = numpy.zeros((n4c, n4c), numpy.complex)
        prd = numpy.zeros((n4c, n4c), numpy.complex)

        s = self.dkb_mol.intor_symmetric('cint1e_ovlp')
        t = self.dkb_mol.intor_symmetric('cint1e_spsp') * .5
        p = self.dkb_mol.intor_symmetric('cint1e_sp')

        s1e[:n2c,:n2c] = s[:n2c,:n2c] + t[:n2c,:n2c] * (.5/c**2)
        s1e[n2c:,n2c:] = s[n2c:,n2c:] + t[n2c:,n2c:] * (.5/c**2)
        prd[:n2c,:n2c] = s[:n2c,:n2c]
        prd[:n2c,n2c:] = t[:n2c,:n2c] * (.5/c**2)
        prd[n2c:,:n2c] = p[n2c:,:n2c] * (-.5/c)
        prd[n2c:,n2c:] = p[n2c:,:n2c] * (.5/c)
        return numpy.linalg.solve(s1e, prd)

    def _init_guess_by_chkfile(self, mol):
        try:
            chk_mol, scf_rec = chkfile.read_scf(self.chkfile)
        except IOError:
            log.warn(mol, 'Fail in reading from %s. Use RKB initial guess', \
                     self.chkfile)
            return self._init_guess_from_rkb(mol)

        if not mol.is_same_mol(chk_mol):
            #raise RuntimeError('input moleinfo is incompatible with chkfile')
            log.warn(mol, 'input moleinfo is incompatible with chkfile. ' \
                     'Use 1e initial guess')
            return self._init_guess_from_rkb(mol)

        #TODO the projection
        mo_coeff = scf_rec['mo_coeff']
        mo_energy = scf_rec['mo_energy']
        hf_energy = scf_rec['hf_energy']
        mo_occ = self.set_mo_occ(mo_energy, mo_coeff)
        dm = self.calc_den_mat(mo_coeff, mo_occ)
        return hf_energy, dm

    def _init_guess_by_rkb_chkfile(self, mol):
        try:
            chk_mol, scf_rec = chkfile.read_scf(self.rkb_chkfile)
        except IOError:
            log.warn(mol, 'Fail in reading RKB chkfile from %s. ' \
                     'Use RKB initial guess' \
                     % self.rkb_chkfile)
            return self._init_guess_from_rkb(mol)

        if not mol.is_same_mol(chk_mol):
            #raise RuntimeError('input moleinfo is incompatible with chkfile')
            log.warn(mol, 'input moleinfo is incompatible with chkfile. ' \
                     'Use 1e initial guess')
            return self._init_guess_from_rkb(mol)

        #TODO the projection
        mo_coeff = numpy.dot(self.project_rkb_to_dkb(mol), scf_rec['mo_coeff'])
        mo_occ = scf_rec['mo_occ']
        return scf_rec['hf_energy'], numpy.dot(mo_coeff*mo_occ, \
                                               mo_coeff.T.conj())

    def get_hcore(self, mol):
        n2c = mol.num_2C_function()
        n4c = n2c * 2
        c = mol.light_speed
        h1e = numpy.zeros((n4c, n4c), numpy.complex)

        s  = self.dkb_mol.intor_symmetric('cint1e_ovlp')
        t  = self.dkb_mol.intor_symmetric('cint1e_spsp') * .5
        vn = self.dkb_mol.intor_symmetric('cint1e_nuc')
        wn = self.dkb_mol.intor_symmetric('cint1e_spnucsp')
        pvn = self.dkb_mol.intor('cint1e_spnuc')
        ppp = self.dkb_mol.intor_symmetric('cint1e_spspsp')
        h1e[:n2c,:n2c] = vn[:n2c,:n2c] + t[:n2c,:n2c] + wn[:n2c,:n2c]*(.5/c)**2
        h1e[:n2c,n2c:] = (pvn[:n2c,n2c:] - pvn[n2c:,:n2c].T.conj()) * (.5/c) \
                - ppp[:n2c,n2c:] * (.25/c)
        h1e[n2c:,:n2c] = h1e[:n2c,n2c:].T.conj()
        h1e[n2c:,n2c:] = vn[n2c:,n2c:] - t[n2c:,n2c:] * 2 \
                + wn[n2c:,n2c:]*(.5/c)**2 - s[n2c:,n2c:]*(2*c**2)
        return h1e

    def get_ovlp(self, mol):
        n2c = mol.num_2C_function()
        n4c = n2c * 2
        c = mol.light_speed
        s = self.dkb_mol.intor_symmetric('cint1e_ovlp')
        t = self.dkb_mol.intor_symmetric('cint1e_spsp') * .5
        s1e = numpy.zeros((n4c, n4c), numpy.complex)
        s1e[:n2c,:n2c] = s[:n2c,:n2c] + t[:n2c,:n2c] * (.5/c**2)
        s1e[n2c:,n2c:] = s[n2c:,n2c:] + t[n2c:,n2c:] * (.5/c**2)
        return s1e

    def init_direct_scf(self, mol):
        if self.direct_scf:
            natm = lib.c_int_p(ctypes.c_int(self.dkb_mol._atm.__len__()))
            nbas = lib.c_int_p(ctypes.c_int(self.dkb_mol._bas.__len__()))
            atm = lib.c_int_arr(self.dkb_mol._atm)
            bas = lib.c_int_arr(self.dkb_mol._bas)
            env = lib.c_double_arr(self.dkb_mol._env)
            _cint.init_dkb_direct_scf_(atm, natm, bas, nbas, env)
            self.set_direct_scf_threshold(self.direct_scf_threshold)
        else:
            _cint.turnoff_direct_scf_()

    def del_direct_scf(self):
        _cint.del_dkb_direct_scf_()


    def get_coulomb_vj_vk(self, mol, dm, coulomb_allow):
        log.info(self, 'DKB Coulomb integral')
        vj, vk = hf.get_vj_vk(pycint.dkb_vhf_coul, self.dkb_mol, dm)
        #vj, vk = hf.get_vj_vk(pycint.dkb_vhf_coul_o02, self.dkb_mol, dm)
        return vj, vk

    def get_coulomb_vj_vk_screen(self, mol, dm, coulomb_allow):
        log.info(self, 'DKB Coulomb integral')
        vj, vk = hf.get_vj_vk(pycint.dkb_vhf_coul_direct, self.dkb_mol, dm)
        return vj, vk

    def scf_cycle(self, mol, scf_threshold=1e-9, dump_chk=True):
        log.info(self, 'start scf_cycle for DKB')
        return hf.SCF.scf_cycle(self, mol, scf_threshold, dump_chk)


if __name__ == '__main__':
    from pyscf.gto import basis
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dhf_dkb'

    mol.atom.extend([['He', (0.,0.,0.)], ])
# input even-tempered basis
    mol.etb = {
        'He': { 'max_l' : 1           # for even-tempered basis
              , 's'     : (4, 1, 1.8) # for etb:(num_basis, alpha, beta)
              , 'p'     : (1, 1, 1.8) # for etb: eta = alpha*beta**i
              , 'd'     : (0, 1, 1.8) #           for i in range num_basis
              , 'f'     : (0,0,0)
              , 'g'     : (0,0,0)}, }
# or input basis information directly
#    mol.basis = {
#        'He': [(0, 0, (1, 1)),
#               (0, 0, (3, 1)),
#               (1, 0, (1, 1)), ]}
    mol.build()

##############
# SCF result
    method = UHF(mol)
    energy = method.scf(mol)
    print energy
