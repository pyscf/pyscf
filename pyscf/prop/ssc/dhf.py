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

# JCP, 112, 3493
# JCC, 20, 1262

'''
4-component Dirac-Hartree-Fock spin-spin coupling (SSC) constants
'''


import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import tools
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.prop.ssc import rhf as rhf_ssc
from pyscf.prop.ssc.rhf import _write
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor

NUMINT_GRIDS = 30

def make_dia(sscobj, mol, dm0, nuc_pair=None, mb='RMB'):
    '''orbital diamagnetic term'''
    if nuc_pair is None:
        nuc_pair = sscobj.nuc_pair
    npair = len(nuc_pair)
    ssc_dia = numpy.zeros((npair,3,3))
    n4c = dm0.shape[0]
    n2c = n4c // 2

    if mb.upper() == 'RMB':
        for k, (ia, ja) in enumerate(nuc_pair):
            a01int = sa01sa01_integral(mol, mol.atom_coord(ia), mol.atom_coord(ja))
            h2 = numpy.zeros((3,3,n4c,n4c), numpy.complex128)
            h2[:,:,n2c:,:n2c] = a01int * .5
            h2[:,:,:n2c,n2c:] = a01int.conj().transpose(0,1,3,2) * .5
            ssc_dia[k] = numpy.einsum('xyij,ji->xy', h2, dm0).real

    elif mb.upper().startswith('ST'):  # Sternheim approximation
        for k, (ia, ja) in enumerate(nuc_pair):
            a01int = sa01sa01_integral(mol, mol.atom_coord(ia), mol.atom_coord(ja))
            ssc_dia[k] = numpy.einsum('xyij,ji', a01int, dm0[:n2c,:n2c]).real

    return ssc_dia * nist.ALPHA**4

def sa01sa01_integral(mol, orig1, orig2):
    '''vec{r}_A/r_A^3 times sigma vec{r}_B/r_B^3 times sigma'''
    with lib.temporary_env(mol, cart=True):
        s = rhf_ssc.dso_integral(mol, orig1, orig2)

    nao = s.shape[-1]
    gout = numpy.empty((3,3,4,nao,nao))
    gout[0,0,0] = -s[2,1] + s[1,2]
    gout[0,0,1] = 0
    gout[0,0,2] = 0
    gout[0,0,3] = +s[2,2] + s[1,1]
    gout[0,1,0] = +s[2,0]
    gout[0,1,1] = +s[1,2]
    gout[0,1,2] = +s[2,2]
    gout[0,1,3] = -s[1,0]
    gout[0,2,0] = -s[1,0]
    gout[0,2,1] = -s[1,1]
    gout[0,2,2] = -s[2,1]
    gout[0,2,3] = -s[2,0]
    gout[1,0,0] = -s[0,2]
    gout[1,0,1] = -s[2,1]
    gout[1,0,2] = -s[2,2]
    gout[1,0,3] = -s[0,1]
    gout[1,1,0] = 0
    gout[1,1,1] = -s[0,2] + s[2,0]
    gout[1,1,2] = 0
    gout[1,1,3] = +s[2,2] + s[0,0]
    gout[1,2,0] = +s[0,0]
    gout[1,2,1] = +s[0,1]
    gout[1,2,2] = +s[2,0]
    gout[1,2,3] = -s[2,1]
    gout[2,0,0] = +s[0,1]
    gout[2,0,1] = +s[1,1]
    gout[2,0,2] = +s[1,2]
    gout[2,0,3] = -s[0,2]
    gout[2,1,0] = -s[0,0]
    gout[2,1,1] = -s[1,0]
    gout[2,1,2] = -s[0,2]
    gout[2,1,3] = -s[1,2]
    gout[2,2,0] = 0
    gout[2,2,1] = 0
    gout[2,2,2] = -s[1,0] + s[0,1]
    gout[2,2,3] = +s[1,1] + s[0,0]

    c2s_a = []
    c2s_b = []
    for i in range(mol.nbas):
        l = mol.bas_angular(i)
        ca, cb = gto.mole.cart2j_kappa(mol.bas_kappa(i), l, 'sp')
        c2s_a.append(ca)
        c2s_b.append(cb)
    c2s_a = scipy.linalg.block_diag(*c2s_a)
    c2s_b = scipy.linalg.block_diag(*c2s_b)
    c2s_aT = c2s_a.conj().T
    c2s_bT = c2s_b.conj().T

    def cart2spinor(cmat):
        '''1 + 1j*sgima'''
        mx, my, mz, m1 = cmat
        smat  = c2s_aT.dot( m1 + mz*1j).dot(c2s_a)
        smat += c2s_aT.dot( my + mx*1j).dot(c2s_b)
        smat += c2s_bT.dot(-my + mx*1j).dot(c2s_a)
        smat += c2s_bT.dot( m1 - mz*1j).dot(c2s_b)
        return smat

    n2c = mol.nao_2c()
    out = numpy.empty((3,3,n2c,n2c), dtype=numpy.complex128)
    for i in range(3):
        for j in range(3):
            out[i,j] = cart2spinor(gout[i,j])
    return out


# Note mo10 is the imaginary part of MO^1
def make_para(sscobj, mol, mo1, mo_coeff, mo_occ, nuc_pair=None):
    if nuc_pair is None: nuc_pair = sscobj.nuc_pair
    if sscobj.mb.upper().startswith('ST'):
        nmo = mo_occ.size
        mo_coeff = mo_coeff[:,nmo//2:]
        mo_occ   = mo_occ[nmo//2:]

    nocc = numpy.count_nonzero(mo_occ> 0)
    nvir = numpy.count_nonzero(mo_occ==0)
    atm1lst = sorted(set([i for i,j in nuc_pair]))
    atm2lst = sorted(set([j for i,j in nuc_pair]))
    atm1dic = dict([(ia,k) for k,ia in enumerate(atm1lst)])
    atm2dic = dict([(ia,k) for k,ia in enumerate(atm2lst)])
    mo1 = mo1.reshape(len(atm1lst),3,nvir,nocc)
    h1 = make_h1(mol, mo_coeff, mo_occ, atm1lst)
    h1 = numpy.asarray(h1).reshape(len(atm1lst),3,nvir,nocc)

    para = []
    for i,j in nuc_pair:
        e = numpy.einsum('xij,yij->xy', h1[atm1dic[i]], mo1[atm2dic[j]].conj()) * 2
        para.append(e.real)
    return numpy.asarray(para) * nist.ALPHA**4

def make_h1(mol, mo_coeff, mo_occ, atmlst):
    orbo = mo_coeff[:,mo_occ> 0]
    orbv = mo_coeff[:,mo_occ==0]
    n4c = mo_coeff.shape[0]
    n2c = n4c // 2
    h1 = []
    for ia in atmlst:
        mol.set_rinv_origin(mol.atom_coord(ia))
        a01int = mol.intor('int1e_sa01sp_spinor', 3)
        h01 = numpy.zeros((n4c,n4c), numpy.complex128)
        for k in range(3):
            h01[:n2c,n2c:] = .5 * a01int[k]
            h01[n2c:,:n2c] = .5 * a01int[k].conj().T
            h1.append(orbv.conj().T.dot(h01).dot(orbo))
    return h1

def solve_mo1(sscobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              h1=None, s1=None, with_cphf=None):
    cput1 = (time.clock(), time.time())
    log = logger.Logger(sscobj.stdout, sscobj.verbose)
    if mo_energy is None: mo_energy = sscobj._scf.mo_energy
    if mo_coeff  is None: mo_coeff = sscobj._scf.mo_coeff
    if mo_occ    is None: mo_occ = sscobj._scf.mo_occ
    if with_cphf is None: with_cphf = sscobj.cphf

    mol = sscobj.mol
    if sscobj.mb.upper().startswith('ST'):  # Sternheim approximation
        nmo = mo_occ.size
        mo_energy = mo_energy[nmo//2:]
        mo_coeff = mo_coeff[:,nmo//2:]
        mo_occ = mo_occ[nmo//2:]

    if h1 is None:
        atmlst = sorted(set([j for i,j in sscobj.nuc_pair]))
        h1 = numpy.asarray(make_h1(mol, mo_coeff, mo_occ, atmlst))

    if with_cphf:
        if callable(with_cphf):
            vind = with_cphf
        else:
            vind = gen_vind(sscobj._scf, mo_coeff, mo_occ)
        mo1, mo_e1 = cphf.solve(vind, mo_energy, mo_occ, h1, None,
                                sscobj.max_cycle_cphf, sscobj.conv_tol,
                                verbose=log)
    else:
        e_ai = lib.direct_sum('i-a->ai', mo_energy[mo_occ>0], mo_energy[mo_occ==0])
        mo1 = h1 / e_ai
        mo_e1 = None

# Calculate RMB with approximation
# |MO1> = Z_RMB |i> + |p> bar{C}_{pi}^1 ~= |p> C_{pi}^1
# bar{C}_{pi}^1 ~= C_{pi}^1 - <p|Z_RMB|i>
    if sscobj.mb.upper() == 'RMB':
        orbo = mo_coeff[:,mo_occ> 0]
        orbv = mo_coeff[:,mo_occ==0]
        n4c = mo_coeff.shape[0]
        n2c = n4c // 2
        c = lib.param.LIGHT_SPEED
        orbvS_T = orbv[n2c:].conj().T
        for ia in atmlst:
            mol.set_rinv_origin(mol.atom_coord(ia))
            a01int = mol.intor('int1e_sa01sp_spinor', 3)
            for k in range(3):
                s1 = orbvS_T.dot(a01int[k].conj().T).dot(orbo[n2c:])
                mo1[ia*3+k] -= s1 * (.25/c**2)

    logger.timer(sscobj, 'solving mo1 eqn', *cput1)
    return mo1, mo_e1

def gen_vind(mf, mo_coeff, mo_occ):
    mol = mf.mol
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]
    nocc = orbo.shape[1]
    nvir = orbv.shape[1]
    def vind(mo1):
        #direct_scf_bak, mf.direct_scf = mf.direct_scf, False
        dm1 = [orbv.dot(x).dot(orbo.T.conj())
               for x in mo1.reshape(-1,nvir,nocc)]
        dm1 = numpy.asarray([d1+d1.conj().T for d1 in dm1])
        v1mo = numpy.asarray([orbv.conj().T.dot(x).dot(orbo)
                              for x in mf.get_veff(mol, dm1, hermi=1)])
        #mf.direct_scf = direct_scf_bak
        return v1mo.ravel()
    return vind


class SpinSpinCoupling(rhf_ssc.SpinSpinCoupling):
    def __init__(self, scf_method):
        self.mb = 'sternheim' # or RMB, RKB
        rhf_ssc.SpinSpinCoupling.__init__(self, scf_method)

    def dump_flags(self, verbose=None):
        rhf_ssc.SpinSpinCoupling.dump_flags(self, verbose)
        logger.info(self, 'mb = %s', self.mb)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()
        mol = self.mol

        dm0 = self._scf.make_rdm1()
        mo_coeff = self._scf.mo_coeff
        mo_occ = self._scf.mo_occ

        ssc_dia = self.make_dia(mol, dm0, mb=self.mb)

        if mo1 is None:
            mo1 = self.mo10 = self.solve_mo1()[0]
        ssc_para = self.make_para(mol, mo1, mo_coeff, mo_occ)
        e11 = ssc_para + ssc_dia
        logger.timer(self, 'spin-spin coupling', *cput0)

        if self.verbose > logger.QUIET:
            nuc_mag = .5 * (nist.E_MASS/nist.PROTON_MASS)  # e*hbar/2m
            au2Hz = nist.HARTREE2J / nist.PLANCK
            #logger.debug('Unit AU -> Hz %s', au2Hz*nuc_mag**2)
            iso_ssc = au2Hz * nuc_mag ** 2 * numpy.einsum('kii->k', e11) / 3
            natm = mol.natm
            ktensor = numpy.zeros((natm,natm))
            for k, (i, j) in enumerate(self.nuc_pair):
                ktensor[i,j] = ktensor[j,i] = iso_ssc[k]
                if self.verbose >= logger.DEBUG:
                    _write(self.stdout, e11[k],
                           '\nSSC E11 between %d %s and %d %s' \
                           % (i, self.mol.atom_symbol(i),
                              j, self.mol.atom_symbol(j)))
                    _write(self.stdout, ssc_dia [k], 'dia-magnetism')
                    _write(self.stdout, ssc_para[k], 'para-magnetism')

            gyro = [get_nuc_g_factor(mol.atom_symbol(ia)) for ia in range(natm)]
            jtensor = numpy.einsum('ij,i,j->ij', ktensor, gyro, gyro)
            label = ['%2d %-2s'%(ia, mol.atom_symbol(ia)) for ia in range(natm)]
            logger.note(self, 'Reduced spin-spin coupling constant K (Hz)')
            tools.dump_mat.dump_tri(self.stdout, ktensor, label)
            logger.info(self, '\nNuclear g factor %s', gyro)
            logger.note(self, 'Spin-spin coupling constant J (Hz)')
            tools.dump_mat.dump_tri(self.stdout, jtensor, label)
        return e11

    make_dia = make_dia
    make_para = make_para
    solve_mo1 = solve_mo1

SSC = SpinSpinCoupling

from pyscf import scf
scf.dhf.UHF.SSC = scf.dhf.UHF.SpinSpinCoupling = lib.class_as_method(SSC)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 3
    mol.output = None

    mol.atom.extend([
        [1   , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ])
    mol.nucmod = {'F': 2} # gaussian nuclear model
    mol.basis = {'H': '6-31g',
                 'F': '6-31g',}
    mol.build()

    rhf = scf.DHF(mol).run()
    ssc = rhf.SSC()
    ssc.cphf = True
    #ssc.mb = 'RKB' # 'RMB'
    jj = ssc.kernel()
    print(jj)
    print(lib.finger(jj)*1e8 - 0.12144116396441988)
