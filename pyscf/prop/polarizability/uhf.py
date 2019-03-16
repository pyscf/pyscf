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
Non-relativistic static and dynamic polarizability and hyper-polarizability tensor
(In testing)
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.soscf.newton_ah import _gen_uhf_response
from pyscf.prop.polarizability import rhf as rhf_polarizability


def dipole(mf):
    return mf.dip_moment(mf.mol, mf.make_rdm1())


# Note: polarizability and relevant properties are demanding on basis sets.
# ORCA recommends to use Sadlej basis for these properties.
def polarizability(polobj, with_cphf=True):
    from pyscf.prop.nmr import uhf as uhf_nmr
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    mo0a, mo0b = mo_coeff
    orboa = mo0a[:, occidxa]
    orbva = mo0a[:,~occidxa]
    orbob = mo0b[:, occidxb]
    orbvb = mo0b[:,~occidxb]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1a = lib.einsum('xpq,pi,qj->xij', int_r, mo0a.conj(), orboa)
    h1b = lib.einsum('xpq,pi,qj->xij', int_r, mo0b.conj(), orbob)
    s1a = numpy.zeros_like(h1a)
    s1b = numpy.zeros_like(h1b)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = ucphf.solve(vind, mo_energy, mo_occ, (h1a,h1b), (s1a,s1b),
                          polobj.max_cycle_cphf, polobj.conv_tol,
                          verbose=log)[0]
    else:
        mo1 = uhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, (h1a,h1b),
                                           (s1a,s1b))[0]

    e2 = numpy.einsum('xpi,ypi->xy', h1a, mo1[0])
    e2+= numpy.einsum('xpi,ypi->xy', h1b, mo1[1])
    e2 = -(e2 + e2.T)

    if mf.verbose >= logger.INFO:
        xx, yy, zz = e2.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug('Static polarizability tensor\n%s', e2)
    return e2


def hyper_polarizability(polobj, with_cphf=True):
    from pyscf.prop.nmr import uhf as uhf_nmr
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    mo0a, mo0b = mo_coeff
    orboa = mo0a[:, occidxa]
    orbva = mo0a[:,~occidxa]
    orbob = mo0b[:, occidxb]
    orbvb = mo0b[:,~occidxb]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1a = lib.einsum('xpq,pi,qj->xij', int_r, mo0a.conj(), orboa)
    h1b = lib.einsum('xpq,pi,qj->xij', int_r, mo0b.conj(), orbob)
    s1a = numpy.zeros_like(h1a)
    s1b = numpy.zeros_like(h1b)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1, e1 = ucphf.solve(vind, mo_energy, mo_occ, (h1a,h1b), (s1a,s1b),
                              polobj.max_cycle_cphf, polobj.conv_tol, verbose=log)
    else:
        mo1, e1 = uhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, (h1a,h1b),
                                               (s1a,s1b))
    mo1a = lib.einsum('xqi,pq->xpi', mo1[0], mo0a)
    mo1b = lib.einsum('xqi,pq->xpi', mo1[1], mo0b)

    dm1a = lib.einsum('xpi,qi->xpq', mo1a, orboa)
    dm1b = lib.einsum('xpi,qi->xpq', mo1b, orbob)
    dm1a = dm1a + dm1a.transpose(0,2,1)
    dm1b = dm1b + dm1b.transpose(0,2,1)
    vresp = _gen_uhf_response(mf, hermi=1)
    h1ao = int_r + vresp(numpy.stack((dm1a, dm1b)))
    s0 = mf.get_ovlp()
    e3  = lib.einsum('xpq,ypi,zqi->xyz', h1ao[0], mo1a, mo1a)
    e3 += lib.einsum('xpq,ypi,zqi->xyz', h1ao[1], mo1b, mo1b)
    e3 -= lib.einsum('pq,xpi,yqj,zij->xyz', s0, mo1a, mo1a, e1[0])
    e3 -= lib.einsum('pq,xpi,yqj,zij->xyz', s0, mo1b, mo1b, e1[1])
    e3 = (e3 + e3.transpose(1,2,0) + e3.transpose(2,0,1) +
          e3.transpose(0,2,1) + e3.transpose(1,0,2) + e3.transpose(2,1,0))
    e3 = -e3
    log.debug('Static hyper polarizability tensor\n%s', e3)
    return e3


# Solve the frequency-dependent CPHF problem
# [A-wI, B   ] [X] + [h1] = [0]
# [B   , A+wI] [Y]   [h1]   [0]
def ucphf_with_freq(mf, mo_energy, mo_occ, h1, freq=0,
                    max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    mo_ea, mo_eb = mo_energy

    # e_ai - freq may produce very small elements which can cause numerical
    # issue in krylov solver
    LEVEL_SHIF = 0.1
    e_ai_a = lib.direct_sum('a-i->ai', mo_ea[viridxa], mo_ea[occidxa]).ravel()
    e_ai_b = lib.direct_sum('a-i->ai', mo_eb[viridxb], mo_eb[occidxb]).ravel()
    diag = (e_ai_a - freq,
            e_ai_b - freq,
            e_ai_a + freq,
            e_ai_b + freq)
    diag[0][diag[0] < LEVEL_SHIF] += LEVEL_SHIF
    diag[1][diag[1] < LEVEL_SHIF] += LEVEL_SHIF
    diag[2][diag[2] < LEVEL_SHIF] += LEVEL_SHIF
    diag[3][diag[3] < LEVEL_SHIF] += LEVEL_SHIF

    mo0a, mo0b = mf.mo_coeff
    nao, nmoa = mo0a.shape
    nmob = mo0b.shape
    orbva = mo0a[:,viridxa]
    orbvb = mo0b[:,viridxb]
    orboa = mo0a[:,occidxa]
    orbob = mo0b[:,occidxb]
    nvira = orbva.shape[1]
    nvirb = orbvb.shape[1]
    nocca = orboa.shape[1]
    noccb = orbob.shape[1]
    h1a = h1[0].reshape(-1,nvira*nocca)
    h1b = h1[1].reshape(-1,nvirb*noccb)
    ncomp = h1a.shape[0]

    mo1base = numpy.hstack((-h1a/diag[0],
                            -h1b/diag[1],
                            -h1a/diag[2],
                            -h1b/diag[3]))

    offsets = numpy.cumsum((nocca*nvira, noccb*nvirb, nocca*nvira))
    vresp = _gen_uhf_response(mf, hermi=0)
    def vind(xys):
        nz = len(xys)
        dm1a = numpy.empty((nz,nao,nao))
        dm1b = numpy.empty((nz,nao,nao))
        for i in range(nz):
            xa, xb, ya, yb = numpy.split(xys[i], offsets)
            dmx = reduce(numpy.dot, (orbva, xa.reshape(nvira,nocca)  , orboa.T))
            dmy = reduce(numpy.dot, (orboa, ya.reshape(nvira,nocca).T, orbva.T))
            dm1a[i] = dmx + dmy  # AX + BY
            dmx = reduce(numpy.dot, (orbvb, xb.reshape(nvirb,noccb)  , orbob.T))
            dmy = reduce(numpy.dot, (orbob, yb.reshape(nvirb,noccb).T, orbvb.T))
            dm1b[i] = dmx + dmy  # AX + BY

        v1ao = vresp(numpy.stack((dm1a,dm1b)))
        v1voa = lib.einsum('xpq,pi,qj->xij', v1ao[0], orbva, orboa).reshape(nz,-1)
        v1vob = lib.einsum('xpq,pi,qj->xij', v1ao[1], orbvb, orbob).reshape(nz,-1)
        v1ova = lib.einsum('xpq,pi,qj->xji', v1ao[0], orboa, orbva).reshape(nz,-1)
        v1ovb = lib.einsum('xpq,pi,qj->xji', v1ao[1], orbob, orbvb).reshape(nz,-1)

        for i in range(nz):
            xa, xb, ya, yb = numpy.split(xys[i], offsets)
            v1voa[i] += (e_ai_a - freq - diag[0]) * xa
            v1voa[i] /= diag[0]
            v1vob[i] += (e_ai_b - freq - diag[1]) * xb
            v1vob[i] /= diag[1]
            v1ova[i] += (e_ai_a + freq - diag[2]) * ya
            v1ova[i] /= diag[2]
            v1ovb[i] += (e_ai_b + freq - diag[3]) * yb
            v1ovb[i] /= diag[3]
        v = numpy.hstack((v1voa, v1vob, v1ova, v1ovb))
        return v

    # FIXME: krylov solver is not accurate enough for many freqs. Using tight
    # tol and lindep could offer small help. A better linear equation solver
    # is needed.
    mo1 = lib.krylov(vind, mo1base, tol=tol, max_cycle=max_cycle,
                     hermi=hermi, lindep=1e-18, verbose=log)
    log.timer('krylov solver in CPHF', *t0)

    dm1a = numpy.empty((ncomp,nao,nao))
    dm1b = numpy.empty((ncomp,nao,nao))
    for i in range(ncomp):
        xa, xb, ya, yb = numpy.split(mo1[i], offsets)
        dmx = reduce(numpy.dot, (orbva, xa.reshape(nvira,nocca)  *2, orboa.T))
        dmy = reduce(numpy.dot, (orboa, ya.reshape(nvira,nocca).T*2, orbva.T))
        dm1a[i] = dmx + dmy
        dmx = reduce(numpy.dot, (orbvb, xb.reshape(nvirb,noccb)  *2, orbob.T))
        dmy = reduce(numpy.dot, (orbob, yb.reshape(nvirb,noccb).T*2, orbvb.T))
        dm1b[i] = dmx + dmy

    v1ao = vresp(numpy.stack((dm1a,dm1b)))
    mo_e1_a = lib.einsum('xpq,pi,qj->xij', v1ao[0], orboa, orboa)
    mo_e1_b = lib.einsum('xpq,pi,qj->xij', v1ao[1], orbob, orbob)
    mo_e1 = (mo_e1_a, mo_e1_b)
    xa, xb, ya, yb = numpy.split(mo1, offsets, axis=1)
    mo1 = (xa.reshape(ncomp,nvira,nocca),
           xb.reshape(ncomp,nvirb,noccb),
           ya.reshape(ncomp,nvira,nocca),
           yb.reshape(ncomp,nvirb,noccb))
    return mo1, mo_e1


def polarizability_with_freq(polobj, freq=None):
    from pyscf.prop.nmr import rhf as rhf_nmr
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    mo0a, mo0b = mo_coeff
    orboa = mo0a[:, occidxa]
    orbva = mo0a[:,~occidxa]
    orbob = mo0b[:, occidxb]
    orbvb = mo0b[:,~occidxb]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1a = lib.einsum('xpq,pi,qj->xij', int_r, orbva.conj(), orboa)
    h1b = lib.einsum('xpq,pi,qj->xij', int_r, orbvb.conj(), orbob)
    mo1 = ucphf_with_freq(mf, mo_energy, mo_occ, (h1a,h1b), freq,
                          polobj.max_cycle_cphf, polobj.conv_tol,
                          verbose=log)[0]

    # *-1 from the definition of dipole moment.
    e2 = -numpy.einsum('xpi,ypi->xy', h1a, mo1[0])
    e2 -= numpy.einsum('xpi,ypi->xy', h1b, mo1[1])
    e2 -= numpy.einsum('xpi,ypi->xy', h1a, mo1[2])
    e2 -= numpy.einsum('xpi,ypi->xy', h1b, mo1[3])

    log.debug('Polarizability tensor with freq %s', freq)
    log.debug('%s', e2)
    return e2


class Polarizability(lib.StreamObject):
    def __init__(self, mf):
        mol = mf.mol
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self._scf = mf

        self.cphf = True
        self.max_cycle_cphf = 20
        self.conv_tol = 1e-9

        self._keys = set(self.__dict__.keys())


    def gen_vind(self, mf, mo_coeff, mo_occ):
        '''Induced potential'''
        vresp = _gen_uhf_response(mf, hermi=1)
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        mo0a, mo0b = mo_coeff
        orboa = mo0a[:, occidxa]
        orbob = mo0b[:, occidxb]
        nocca = orboa.shape[1]
        noccb = orbob.shape[1]
        nmoa = mo0a.shape[1]
        nmob = mo0b.shape[1]
        def vind(mo1):
            mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
            mo1a = mo1[:,:nmoa*nocca].reshape(-1,nmoa,nocca)
            mo1b = mo1[:,nmoa*nocca:].reshape(-1,nmob,noccb)
            dm1a = lib.einsum('xai,pa,qi->xpq', mo1a, mo0a, orboa.conj())
            dm1b = lib.einsum('xai,pa,qi->xpq', mo1b, mo0b, orbob.conj())
            dm1a = dm1a + dm1a.transpose(0,2,1).conj()
            dm1b = dm1b + dm1b.transpose(0,2,1).conj()
            v1ao = vresp(numpy.stack((dm1a,dm1b)))
            v1a = lib.einsum('xpq,pi,qj->xij', v1ao[0], mo0a.conj(), orboa)
            v1b = lib.einsum('xpq,pi,qj->xij', v1ao[1], mo0b.conj(), orbob)
            v1mo = numpy.hstack((v1a.reshape(-1,nmoa*nocca),
                                 v1b.reshape(-1,nmob*noccb)))
            return v1mo.ravel()
        return vind


    polarizability = polarizability
    polarizability_with_freq = polarizability_with_freq

    hyper_polarizability = hyper_polarizability

from pyscf import scf
scf.uhf.UHF.Polarizability = lib.class_as_method(Polarizability)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    # Disagreement between analytical results and finite difference found for
    # linear molecule
    #mol.atom = '''h  ,  0.   0.   0.
    #              F  ,  0.   0.   .917'''

    mol.atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587'''
    mol.spin = 2
    mol.basis = '631g'
    mol.build()

    mf = scf.UHF(mol).run(conv_tol=1e-14)
    polar = mf.Polarizability().polarizability()
    hpol = mf.Polarizability().hyper_polarizability()
    print(polar)

    mf.verbose = 0
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
        mf.run(conv_tol=1e-14)
        return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0.0001, 0])
    e2 = apply_E([0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0, 0.0001])
    e2 = apply_E([0, 0,-0.0001])
    print((e1 - e2) / 0.0002)

    # Small discrepancy found between analytical derivatives and finite
    # differences
    print(hpol)
    def apply_E(E):
        mf.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
        mf.run(conv_tol=1e-14)
        return Polarizability(mf).polarizability()
    e1 = apply_E([ 0.0001, 0, 0])
    e2 = apply_E([-0.0001, 0, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0.0001, 0])
    e2 = apply_E([0,-0.0001, 0])
    print((e1 - e2) / 0.0002)
    e1 = apply_E([0, 0, 0.0001])
    e2 = apply_E([0, 0,-0.0001])
    print((e1 - e2) / 0.0002)

    print(Polarizability(mf).polarizability())
    print(Polarizability(mf).polarizability_with_freq(freq= 0.))

    print(Polarizability(mf).polarizability_with_freq(freq= 0.1))
    print(Polarizability(mf).polarizability_with_freq(freq=-0.1))

