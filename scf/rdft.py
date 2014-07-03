#!/usr/bin/env python
# -*- coding: utf-8
#
# File: rdft.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Relativistic Kohn-Sham
'''

__author__ = "Qiming Sun <osirpt.sun@gmail.com>"
__version__ = "$ 0.2 $"

import numpy
from pyscf import gto
from pyscf import lib
import pyscf.lib.logger as log
from pyscf.lib import pyvxc
from pyscf.lib import pycint
import dhf
import dft
from dft import *


__doc__ = '''Options:
self.chkfile = "/dev/shm/..."
self.fout = "..."
self.diis_space = 6
self.diis_start_cycle = 1
self.damp_factor = 1
self.level_shift_factor = 0
self.scf_threshold = 1e-10
self.max_scf_cycle = 50

self.init_guess(method)         # method = one of "atom", "1e", "chkfile"
self.potential(v, oob)          # v = one of "coulomb", "gaunt"
                                # oob = operator oriented basis level
                                #       1 sp|f> -> |f>
                                #       2 sp|f> -> sr|f>
self.xc_func(func_x,func_c)     # default is LSDA. or HF if not call this function
'''

class UKS(dhf.UHF, dft.RKS):
    ''' Unrestricted Kohn-Sham '''
    def __init__(self, mol):
        dhf.UHF.__init__(self, mol)
        self._ecoul = 0
        self._exc = 0
        self.func_x = XC_LDA_X
        self.func_c = XC_LDA_C_VWN_RPA
        log.debug(self, "initialize pyvxc")
        pyvxc.init_vxc(1, mol._atm, mol._bas, mol._env)

    def get_vj_vxc(self, mol, dm):
        n2c = mol.num_2C_function()
        dmll = dm[:n2c,:n2c]
        dmss = dm[n2c:,n2c:]
        self._exc, n, vxc = pyvxc.exc_vxc(self.func_x, self.func_c, \
                                          numpy.hstack((dmll, dmss)))
        vxcll,vxcss = numpy.hsplit(vxc, 2)
        log.debug(self, 'num electrons by numeric integration = %s', n)
        vj, vk = dhf.get_coulomb_hj_vk(mol, dm, self._coulomb_type)
        self._ecoul = lib.trace_ab(dm, vj) * .5
        vxc = numpy.zeros_like(vj)
        vxc[:n2c,:n2c] = vxcll
        vxc[n2c:,n2c:] = vxcss
        hyb = pyvxc.hybrid_coeff(self.func_x)
        if abs(hyb) > 1e-10:
            vk = vk * hyb
            self._exc -= lib.trace_ab(dm, vk) * .5
            vxc -= vk
        return vj, vxc

    def calc_tot_elec_energy(self, veff, dm, mo_energy, mo_occ):
        self._r_last_hf_e = self._r_hf_energy

        sum_mo_energy = numpy.dot(mo_energy, mo_occ)
        coul_dup = lib.trace_ab(dm, veff)
        log.debug(self, 'Ecoul = %s  Exc = %s', self._ecoul, self._exc)
        tot_e = sum_mo_energy - coul_dup + self._ecoul + self._exc
        self._r_hf_energy = tot_e
        return tot_e.real, self._ecoul.real, self._exc.real


class RKS(UKS, dhf.RHF):
    def __init__(self, mol):
        if mol.nelectron.__mod__(2) is not 0:
            raise ValueError("Invalid electron number %i." % mol.nelectron)
        UKS.__init__(self, mol)



if __name__ == "__main__":
    from pyscf.gto import basis
    mol = gto.Mole()
    mol.verbose = 7
    mol.output = "out_default"

    mol.atom.extend([["He", (0.,0.,0.)], ])
    mol.basis = {
        "He": basis.ccpvdz["He"]}
    mol.grids = { "He": (10, 14),}
    mol.build()

##############
# SCF result
    method = UKS(mol)
    method.init_guess("1e")
    method.xc_func(XC_LDA_X,XC_LDA_C_VWN_RPA)
    energy = method.scf(mol) #=-2.85210266828
    print energy
