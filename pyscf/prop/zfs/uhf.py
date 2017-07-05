#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic unrestricted Hartree-Fock zero-field splitting

Refs:
    JCP, 127, 164112
'''

import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.gto import mole
from pyscf.dft import numint
from pyscf.prop.nmr import uhf as uhf_nmr
from pyscf.prop.gtensor.uhf import koseki_charge


def direct_spin_spin(zfsobj, mol, dm0, verbose=None):
    log = logger.new_logger(zfsobj, verbose)
    if isinstance(dm0, numpy.ndarray) and dm0.ndim == 2: # RHF DM
        return numpy.zeros((3,3))

    dma, dmb = dm0
    spindm = dma - dmb
    effspin = mol.spin * .5
    nao = dma.shape[0]

    s2 = effspin * (effspin * 2 - 1)
    fac = lib.param.G_ELECTRON**2 * lib.param.ALPHA**2 / s2 / 16

    hss = mol.intor('int2e_ip1ip2', comp=9).reshape(3,3,nao,nao,nao,nao)
    hss = hss - hss.transpose(0,1,3,2,4,5)
    hss = hss - hss.transpose(0,1,2,3,5,4)
    ej = numpy.einsum('xyijkl,ji,lk', hss, spindm, spindm)
    ek = numpy.einsum('xyijkl,jk,li', hss, spindm, spindm)
    ess = (ej - ek) * fac
    h_fc = mol.intor('int4c1e').reshape(nao,nao,nao,nao)
    ej = numpy.einsum('ijkl,ji,lk', h_fc, spindm, spindm)
    ek = numpy.einsum('ijkl,jk,li', h_fc, spindm, spindm)
    e_fc = (ej - ek) * fac * (8*numpy.pi/3)
    dipdip = ess + e_fc * numpy.eye(3)
    return dipdip

# Note mo1 is the imaginary part of MO^1
def soc_part(zfsobj, mol, mo1, mo_coeff, mo_occ):
    effspin = mol.spin * .5
    h1aa, h1bb, h1ab, h1ba = make_h1_mo(mol, mo_coeff, mo_occ)
    dkl = .25/effspin**2 * numpy.einsum('xij,yij->xy', h1aa, mo1[0])
    dkl+= .25/effspin**2 * numpy.einsum('xij,yij->xy', h1bb, mo1[1])
    dkl-= .25/effspin**2 * numpy.einsum('xij,yij->xy', h1ab, mo1[2])
    dkl-= .25/effspin**2 * numpy.einsum('xij,yij->xy', h1ba, mo1[3])
    dkl *= lib.param.ALPHA**2 / 2
    return dkl

def make_h1_mo(mol, mo_coeff, mo_occ):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,~occidxa]
    orbvb = mo_coeff[1][:,~occidxb]
# hso1e is the imaginary part of [i sigma dot pV x p]
# JCP, 122, 034107 Eq (2) = 1/4c^2 hso1e
    if zfsobj.with_so_eff_charge:
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

    if zfsobj.with_sso or zfsobj.with_soo:
        hso2e = make_soc2e_mo(zfsobj, mo_coeff, mo_occ)
    else:
        hso2e = (0, 0, 0, 0)

    h1aa += hso2e[0]
    h1bb += hso2e[1]
    h1ab += hso2e[2]
    h1ba += hso2e[3]
    return h1aa, h1bb, h1ab, h1ba

# Using SOMF approximation in this implementation
#TODO: full SSO+SOO without approximation
def make_soc2e_mo(zfsobj, mo_coeff, mo_occ):
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
    hbb = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orbob)) for x in hso2e])
    hab = numpy.asarray([reduce(numpy.dot, (orbva.T, x, orbob)) for x in hso2e])
    hba = numpy.asarray([reduce(numpy.dot, (orbvb.T, x, orboa)) for x in hso2e])
    return haa, hbb, hab, hba

def solve_mo1(zfsobj):
    log = logger.new_logger(zfsobj, zfsobj.verbose)
    mol = zfsobj.mol
    mf = zfsobj._scf
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ

    h1 = make_h1_mo(mol, mo_coeff, mo_occ)
    h1aa, h1bb, h1ab, h1ba = h1

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:,~occidxa]
    orbvb = mo_coeff[1][:,~occidxb]
    eai_aa = -1 / (mo_energy[0][viridxa,None] - mo_energy[0][occidxa])
    eai_bb = -1 / (mo_energy[1][viridxb,None] - mo_energy[1][occidxb])
    eai_ab = -1 / (mo_energy[0][viridxa,None] - mo_energy[1][occidxb])
    eai_ba = -1 / (mo_energy[1][viridxb,None] - mo_energy[0][occidxa])
    mo1_aa = h1aa * eai_aa
    mo1_bb = h1bb * eai_bb
    mo1_ab = h1ab * eai_ab
    mo1_ba = h1ba * eai_ba

    if zfsobj.cphf:
        nvira, nocca = eai_aa.shape
        nvirb, noccb = eai_bb.shape
        p0 = nvira * nocca
        p1 = p0 + nvirb * noccb
        p2 = p1 + nvira * noccb
        p3 = p2 + nvirb * nocca
        def vind(x):
            x = x.reshape(3,-1)
            mo1_aa = x[:,  :p0].reshape(3,nvira,nocca)
            mo1_bb = x[:,p0:p1].reshape(3,nvirb,noccb)
            mo1_ab = x[:,p1:p2].reshape(3,nvira,noccb)
            mo1_ba = x[:,p2:  ].reshape(3,nvirb,nocca)
            dm1 = numpy.asarray([reduce(numpy.dot, (orbva, x, orboa.T)) for x in mo1_aa] +
                                [reduce(numpy.dot, (orbvb, x, orbob.T)) for x in mo1_bb] +
                                [reduce(numpy.dot, (orbva, x, orbob.T)) for x in mo1_ab] +
                                [reduce(numpy.dot, (orbvb, x, orboa.T)) for x in mo1_ba])
            dm1 = dm1 - dm1.transpose(0,2,1)
            vj, vk = mf.get_jk(mol, dm1, hermi=2)
            v1ao = -vk
            mo1_aa = [reduce(numpy.dot, (orbva.T, v1ao[i], orboa)) for i in range(3)  ]
            mo1_bb = [reduce(numpy.dot, (orbvb.T, v1ao[i], orbob)) for i in range(3,6)]
            mo1_ab = [reduce(numpy.dot, (orbva.T, v1ao[i], orbob)) for i in range(6,9)]
            mo1_ba = [reduce(numpy.dot, (orbvb.T, v1ao[i], orboa)) for i in range(9,12)]
            mo1_aa = (numpy.asarray(mo1_aa) * eai_aa).reshape(3,-1)
            mo1_bb = (numpy.asarray(mo1_bb) * eai_bb).reshape(3,-1)
            mo1_ab = (numpy.asarray(mo1_ab) * eai_ab).reshape(3,-1)
            mo1_ba = (numpy.asarray(mo1_ba) * eai_ba).reshape(3,-1)
            return numpy.hstack((mo1_aa, mo1_bb, mo1_ab, mo1_ba)).ravel()

        mo1 = numpy.hstack((mo1_aa.reshape(3,-1), mo1_bb.reshape(3,-1),
                            mo1_ab.reshape(3,-1), mo1_ba.reshape(3,-1)))
        mo1 = lib.krylov(vind, mo1.ravel(), tol=1e-9, max_cycle=20, verbose=log)
        mo1 = mo1.reshape(3,-1)
        mo1_aa = mo1[:,  :p0].reshape(3,nvira,nocca)
        mo1_bb = mo1[:,p0:p1].reshape(3,nvirb,noccb)
        mo1_ab = mo1[:,p1:p2].reshape(3,nvira,noccb)
        mo1_ba = mo1[:,p2:  ].reshape(3,nvirb,nocca)

    return (mo1_aa, mo1_bb, mo1_ab, mo1_ba)


class ZeroFieldSplitting(uhf_nmr.NMR):
    '''dE = I dot gtensor dot s'''
    def __init__(self, scf_method):
        self.with_sso = False  # Two-electron spin-same-orbit coupling
        self.with_soo = False  # Two-electron spin-other-orbit coupling
        self.with_so_eff_charge = True
        uhf_nmr.NMR.__init__(self, scf_method)

    def dump_flags(self):
        uhf_nmr.NMR.dump_flags(self)
        logger.info(self, 'with_sso = %s (2e spin-same-orbit coupling)', self.with_sso)
        logger.info(self, 'with_soo = %s (2e spin-other-orbit coupling)', self.with_soo)
        logger.info(self, 'with_so_eff_charge = %s (1e SO effective charge)',
                    self.with_so_eff_charge)

    def kernel(self, mo1=None):
        cput0 = (time.clock(), time.time())
        self.check_sanity()
        self.dump_flags()

        mol = self.mol
        dm0 = self._scf.make_rdm1()

        zfs_ss = direct_spin_spin(self, mol, dm0)
        mo1 = solve_mo1(self)
        zfs_soc = soc_part(zfsobj, mol, mo1, self._scf.mo_coeff, self._scf.mo_occ)
        zfs_tensor = zfs_ss + zfs_soc

        logger.timer(self, 'ZFS tensor', *cput0)
        if self.verbose > logger.QUIET:
            self.stdout.write('\nZero-field splitting tensor\n')
            self.stdout.write('S_x %s\n' % zfs_tensor[0])
            self.stdout.write('S_y %s\n' % zfs_tensor[1])
            self.stdout.write('S_z %s\n' % zfs_tensor[2])
            self.stdout.flush()
        return zfs_tensor

ZFS = ZeroFieldSplitting


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='Ne 0 0 0',
                basis='ccpvdz', spin=2, charge=-2, verbose=3)
    mf = scf.UHF(mol)
    mf.kernel()
    zfsobj = ZFS(mf)
    zfsobj.with_sso = True
    zfsobj.with_soo = True
    zfsobj.with_so_eff_charge = False
    print(zfsobj.kernel())
