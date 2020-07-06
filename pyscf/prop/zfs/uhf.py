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

'''
Non-relativistic unrestricted Hartree-Fock zero-field splitting
(In testing)

Refs:
    JCP 134, 194113 (2011); DOI:10.1063/1.3590362
    PRB 60, 9566 (1999); DOI:10.1103/PhysRevB.60.9566
    JCP 127, 164112 (2007); 10.1063/1.2772857
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.scf import _response_functions  # noqa
from pyscf.prop.ssc.rhf import _dm1_mo2ao
from pyscf.data import nist


def koseki_charge(z):
    '''Koseki effective charge in SO correction

    Ref:
    JPC 96, 10768
    JPC, 99, 12764
    JPCA, 102, 10430
    '''
    # JPC 96, 10768
    if z <= 2:
        return z
    elif z <= 10:
        return z * (.3 + z * .05)
    elif z <= 18:
        return z * (1.05 - z * .0125)
    elif z <= 30:
        return z * ( 0.385 + 0.025 * (z - 18 - 2) ) # Jia: J. Phys. Chem. A 1998, 102, 10430
    elif z < 48:
        return z * ( 4.680 + 0.060 * (z - 36 - 2) )
    else:
        return z


def direct_spin_spin(zfsobj, mol, dm0, verbose=None):
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5
    nao = dma.shape[0]

    # Use QED g-factor or Dirac g-factor
    #g_fac = nist.G_ELECTRON**2/4  # QED
    g_fac = 1
    fac = g_fac * nist.ALPHA**2 / 8 / (effspin * (effspin - .5))

    hss = mol.intor('int2e_ip1ip2', comp=9).reshape(3,3,nao,nao,nao,nao)
    hss = hss + hss.transpose(0,1,3,2,4,5)
    hss = hss + hss.transpose(0,1,2,3,5,4)
    ej = numpy.einsum('xyijkl,ji,lk', hss, spindm, spindm)
    ek = numpy.einsum('xyijkl,jk,li', hss, spindm, spindm)
    dss = (ej - ek) * fac

# 2-electron Fermi contact term
# FC contribution is zero in mean-field calculations because of the 16-fold
# symmetry of the 4-index tensor.
# Generally, in a CI-like wfn, FC may have contributions to the direction
# spin-spin coupling.
    if 0:
        h_fc = mol.intor('int4c1e').reshape(nao,nao,nao,nao)
        ej = numpy.einsum('ijkl,ji,lk', h_fc, spindm, spindm)
        ek = numpy.einsum('ijkl,jk,li', h_fc, spindm, spindm)
        e_fc = (ej - ek) * fac * (4*numpy.pi/3)
        dss -= e_fc * numpy.eye(3)
    return dss

# Note mo1 is the imaginary part of MO^1
def make_soc(zfsobj, mol, mo_coeff, mo_occ):
    h1 = make_h1_soc(zfsobj, mol, mo_coeff, mo_occ)
    mo1 = solve_mo1(zfsobj, h1)
    h1aa, h1ab, h1ba, h1bb = h1
    mo1aa, mo1ab, mo1ba, mo1bb = mo1

    effspin = mol.spin * .5
    if 0: # Pederson-Khanna formula , PRB, 60, 9566
        fac = -.25 / effspin**2
        dso  = fac * numpy.einsum('xij,yij->xy', h1aa, mo1aa)
        dso += fac * numpy.einsum('xij,yij->xy', h1bb, mo1bb)
        dso -= fac * numpy.einsum('xij,yij->xy', h1ab, mo1ab)
        dso -= fac * numpy.einsum('xij,yij->xy', h1ba, mo1ba)
    elif 0: # Neese formula, see JCP, 127, 164112
        facy = -.25 / ((effspin-.5)*effspin)
        facz = -.25 / effspin**2
        facx = -.25 / ((effspin+.5)*(effspin+1))
        dso  = facz * numpy.einsum('xij,yij->xy', h1aa, mo1aa)
        dso += facz * numpy.einsum('xij,yij->xy', h1bb, mo1bb)
        dso -= facx * numpy.einsum('xij,yij->xy', h1ab, mo1ab)
        dso -= facy * numpy.einsum('xij,yij->xy', h1ba, mo1ba)
    else: # van Wullen formula, JCP, 134, 194113
        # Note the sign difference to van Wullen's paper, due to the
        # anti-symmetricity of the Hamiltonian
        fac = -.25 / (effspin*(effspin-.5))
        dso  = fac * numpy.einsum('xij,yij->xy', h1aa, mo1aa)
        dso += fac * numpy.einsum('xij,yij->xy', h1bb, mo1bb)
        dso -= fac * numpy.einsum('xij,yij->xy', h1ab, mo1ab)
        dso -= fac * numpy.einsum('xij,yij->xy', h1ba, mo1ba)

    dso *= nist.ALPHA ** 4 / 4
    return dso

def make_h1_soc(zfsobj, mol, mo_coeff, mo_occ):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:,~occidxa]
    orbvb = mo_coeff[1][:,~occidxb]
# hso1e is the imaginary part of [i sigma dot pV x p]
# JCP, 122, 034107 Eq (2) = 1/4c^2 hso1e
    if zfsobj.so_eff_charge:
        hso1e = 0
        for ia in range(mol.natm):
            mol.set_rinv_origin(mol.atom_coord(ia))
            #FIXME: when ECP is enabled
            Z = koseki_charge(mol.atom_charge(ia))
            hso1e += -Z * mol.intor('int1e_prinvxp', 3)
    else:
        hso1e = mol.intor('int1e_pnucxp', 3)
    h1aa = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orboa)) for x in hso1e])
    h1bb = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orbob)) for x in hso1e])
    h1ab = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orbob)) for x in hso1e])
    h1ba = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orboa)) for x in hso1e])

    if zfsobj.sso or zfsobj.soo:
        hso2e = make_soc2e(zfsobj, mo_coeff, mo_occ)
    else:
        hso2e = (0, 0, 0, 0)

    h1aa += hso2e[0]
    h1ab += hso2e[1]
    h1ba += hso2e[2]
    h1bb += hso2e[3]
    return h1aa, h1ab, h1ba, h1bb

# Using the approximation in JCP, 122, 034107
def make_soc2e(zfsobj, mo_coeff, mo_occ):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,~occidxa]
    orbvb = mo_coeff[1][:,~occidxb]
    dma = numpy.dot(orboa, orboa.T)
    dmb = numpy.dot(orbob, orbob.T)
    dm1 = dma + dmb
    nao = dma.shape[0]

# hso2e is the imaginary part of SSO
    hso2e = mol.intor('int2e_p1vxp1', 3).reshape(3,nao,nao,nao,nao)
    vj = numpy.einsum('yijkl,lk->yij', hso2e, dm1)
    vk = numpy.einsum('yijkl,jk->yil', hso2e, dm1)
    vk+= numpy.einsum('yijkl,li->ykj', hso2e, dm1)
    hso2e = vj - vk * 1.5

    haa = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orboa)) for x in hso2e])
    hab = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orbob)) for x in hso2e])
    hba = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orboa)) for x in hso2e])
    hbb = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orbob)) for x in hso2e])
    return haa, hab, hba, hbb

def solve_mo1(sscobj, h1):
    cput1 = (time.clock(), time.time())
    log = logger.Logger(sscobj.stdout, sscobj.verbose)

    mo_energy = sscobj._scf.mo_energy
    mo_coeff = sscobj._scf.mo_coeff
    mo_occ = sscobj._scf.mo_occ
    h1aa, h1ab, h1ba, h1bb = h1
    nset = len(h1aa)
    eai_aa = 1. / lib.direct_sum('a-i->ai', mo_energy[0][mo_occ[0]==0], mo_energy[0][mo_occ[0]>0])
    eai_ab = 1. / lib.direct_sum('a-i->ai', mo_energy[0][mo_occ[0]==0], mo_energy[1][mo_occ[1]>0])
    eai_ba = 1. / lib.direct_sum('a-i->ai', mo_energy[1][mo_occ[1]==0], mo_energy[0][mo_occ[0]>0])
    eai_bb = 1. / lib.direct_sum('a-i->ai', mo_energy[1][mo_occ[1]==0], mo_energy[1][mo_occ[1]>0])

    mo1 = (numpy.asarray(h1aa) * -eai_aa,
           numpy.asarray(h1ab) * -eai_ab,
           numpy.asarray(h1ba) * -eai_ba,
           numpy.asarray(h1bb) * -eai_bb)
    h1aa = h1ab = h1ba = h1bb = None
    if not sscobj.cphf:
        return mo1

    orboa = mo_coeff[0][:,mo_occ[0]> 0]
    orbva = mo_coeff[0][:,mo_occ[0]==0]
    orbob = mo_coeff[1][:,mo_occ[1]> 0]
    orbvb = mo_coeff[1][:,mo_occ[1]==0]
    nocca = orboa.shape[1]
    nvira = orbva.shape[1]
    noccb = orbob.shape[1]
    nvirb = orbvb.shape[1]
    p1 = nvira * nocca
    p2 = p1 + nvira * noccb
    p3 = p2 + nvirb * nocca
    def _split_mo1(mo1):
        mo1 = mo1.reshape(nset,-1)
        mo1aa = mo1[:,  :p1].reshape(nset,nvira,nocca)
        mo1ab = mo1[:,p1:p2].reshape(nset,nvira,noccb)
        mo1ba = mo1[:,p2:p3].reshape(nset,nvirb,nocca)
        mo1bb = mo1[:,p3:  ].reshape(nset,nvirb,noccb)
        return mo1aa, mo1ab, mo1ba, mo1bb

    mo1 = numpy.hstack((mo1[0].reshape(nset,-1),
                        mo1[1].reshape(nset,-1),
                        mo1[2].reshape(nset,-1),
                        mo1[3].reshape(nset,-1)))

    vresp = mf.gen_response(with_j=False, hermi=0)
    mo_va_oa = numpy.asarray(numpy.hstack((orbva,orboa)), order='F')
    mo_va_ob = numpy.asarray(numpy.hstack((orbva,orbob)), order='F')
    mo_vb_oa = numpy.asarray(numpy.hstack((orbvb,orboa)), order='F')
    mo_vb_ob = numpy.asarray(numpy.hstack((orbvb,orbob)), order='F')
    def vind(mo1):
        mo1aa, mo1ab, mo1ba, mo1bb = _split_mo1(mo1)
        dm1aa = _dm1_mo2ao(mo1aa, orbva, orboa)
        dm1ab = _dm1_mo2ao(mo1ab, orbva, orbob)
        dm1ba = _dm1_mo2ao(mo1ba, orbvb, orboa)
        dm1bb = _dm1_mo2ao(mo1bb, orbvb, orbob)
        # imaginary Hermitian
        dm1 = numpy.vstack([dm1aa-dm1aa.transpose(0,2,1),
                            dm1ab-dm1ba.transpose(0,2,1),
                            dm1ba-dm1ab.transpose(0,2,1),
                            dm1bb-dm1bb.transpose(0,2,1)])
        v1 = vresp(dm1)
        v1aa = _ao2mo.nr_e2(v1[      :nset  ], mo_va_oa, (0,nvira,nvira,nvira+nocca))
        v1ab = _ao2mo.nr_e2(v1[nset*1:nset*2], mo_va_ob, (0,nvira,nvira,nvira+noccb))
        v1ba = _ao2mo.nr_e2(v1[nset*2:nset*3], mo_vb_oa, (0,nvirb,nvirb,nvirb+nocca))
        v1bb = _ao2mo.nr_e2(v1[nset*3:      ], mo_vb_ob, (0,nvirb,nvirb,nvirb+noccb))
        v1aa = v1aa.reshape(nset,nvira,nocca)
        v1ab = v1ab.reshape(nset,nvira,noccb)
        v1ba = v1ba.reshape(nset,nvirb,nocca)
        v1bb = v1bb.reshape(nset,nvirb,noccb)
        v1aa *= eai_aa
        v1ab *= eai_ab
        v1ba *= eai_ba
        v1bb *= eai_bb
        v1mo = numpy.hstack((v1aa.reshape(nset,-1), v1ab.reshape(nset,-1),
                             v1ba.reshape(nset,-1), v1bb.reshape(nset,-1)))
        return v1mo.ravel()

    mo1 = lib.krylov(vind, mo1.ravel(), tol=1e-9, max_cycle=20, verbose=log)
    log.timer('solving FC CPHF eqn', *cput1)
    mo1 = _split_mo1(mo1)
    return mo1


class ZeroFieldSplitting(lib.StreamObject):
    '''dE = I dot gtensor dot s'''
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.verbose = scf_method.mol.verbose
        self.stdout = scf_method.mol.stdout
        self.chkfile = scf_method.chkfile
        self._scf = scf_method

        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9
        self.sso = False  # Two-electron spin-same-orbit coupling
        self.soo = False  # Two-electron spin-other-orbit coupling
        self.so_eff_charge = True

        self.mo10 = None
        self.mo_e10 = None
        self._keys = set(self.__dict__.keys())
        logger.warn(self, 'UHF-ZFS is an experimental feature. It is still in '
                    'testing\nFeatures and APIs may be changed in the future.')

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s (In testing) ********',
                 self.__class__, self._scf.__class__)
        log.info('with cphf = %s', self.cphf)
        if self.cphf:
            log.info('CPHF conv_tol = %g', self.conv_tol)
            log.info('CPHF max_cycle_cphf = %d', self.max_cycle_cphf)
        log.info('sso = %s (2e spin-same-orbit coupling)', self.sso)
        log.info('soo = %s (2e spin-other-orbit coupling)', self.soo)
        log.info('so_eff_charge = %s (1e SO effective charge)',
                 self.so_eff_charge)
        return self

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        mol = self.mol
        dm0 = self._scf.make_rdm1()

        zfs_ss = direct_spin_spin(self, mol, dm0)
        zfs_soc = make_soc(zfsobj, mol, self._scf.mo_coeff, self._scf.mo_occ)
        zfs_tensor = zfs_ss + zfs_soc
        zfs_diag = numpy.linalg.eigh(zfs_tensor)[0]
        dtrace = zfs_tensor.trace()
        zfs_diag -= dtrace / 3
        zidx = numpy.argmax(abs(zfs_diag))
        dvalue = zfs_diag[zidx] * 1.5
        tmp = zfs_diag + dvalue/3
        tmp[zidx] = 0
        evalue = abs(tmp).max()
        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        logger.debug(self, 'D trace = %s', dtrace)
        logger.note(self, 'Axial   parameter D = %s (cm^{-1})', dvalue*au2cm)
        logger.note(self, 'Rhombic parameter E = %s (cm^{-1})', evalue*au2cm)

        if self.verbose > logger.debug:
            self.stdout.write('\nZero-field splitting tensor\n')
            self.stdout.write('S_x %s\n' % zfs_tensor[0])
            self.stdout.write('S_y %s\n' % zfs_tensor[1])
            self.stdout.write('S_z %s\n' % zfs_tensor[2])
            self.stdout.flush()
        logger.timer(self, 'ZFS tensor', *cput0)
        return zfs_tensor

ZFS = ZeroFieldSplitting


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='Ne 0 0 0',
                basis='ccpvdz', spin=2, charge=-2, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    zfsobj = ZFS(mf)
    #zfsobj.cphf = False
    #zfsobj.sso = True
    #zfsobj.soo = True
    #zfsobj.so_eff_charge = False
    print(zfsobj.kernel())
