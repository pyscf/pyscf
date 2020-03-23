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
'''


import time
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import cphf
from pyscf.scf import _response_functions  # noqa


def dipole(mf):
    return mf.dip_moment(mf.mol, mf.make_rdm1())


# Note: polarizability and relevant properties are demanding on basis sets.
# ORCA recommends to use Sadlej basis for these properties.
def polarizability(polobj, with_cphf=True):
    from pyscf.prop.nmr import rhf as rhf_nmr
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    #orbv = mo_coeff[:,~occidx]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo)
    s1 = numpy.zeros_like(h1)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,
                         polobj.max_cycle_cphf, polobj.conv_tol,
                         verbose=log)[0]
    else:
        mo1 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)[0]

    e2 = numpy.einsum('xpi,ypi->xy', h1, mo1)
    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 = (e2 + e2.T) * -2

    if mf.verbose >= logger.INFO:
        xx, yy, zz = e2.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug('Static polarizability tensor\n%s', e2)
    return e2


def hyper_polarizability(polobj, with_cphf=True):
    from pyscf.prop.nmr import rhf as rhf_nmr
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    #orbv = mo_coeff[:,~occidx]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1 = lib.einsum('xpq,pi,qj->xij', int_r, mo_coeff.conj(), orbo)
    s1 = numpy.zeros_like(h1)
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1, e1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,
                             polobj.max_cycle_cphf, polobj.conv_tol, verbose=log)
    else:
        mo1, e1 = rhf_nmr._solve_mo1_uncoupled(mo_energy, mo_occ, h1, s1)
    mo1 = lib.einsum('xqi,pq->xpi', mo1, mo_coeff)

    dm1 = lib.einsum('xpi,qi->xpq', mo1, orbo) * 2
    dm1 = dm1 + dm1.transpose(0,2,1)
    vresp = mf.gen_response(hermi=1)
    h1ao = int_r + vresp(dm1)
    # *2 for double occupancy
    e3  = lib.einsum('xpq,ypi,zqi->xyz', h1ao, mo1, mo1) * 2
    e3 -= lib.einsum('pq,xpi,yqj,zij->xyz', mf.get_ovlp(), mo1, mo1, e1) * 2
    e3 = (e3 + e3.transpose(1,2,0) + e3.transpose(2,0,1) +
          e3.transpose(0,2,1) + e3.transpose(1,0,2) + e3.transpose(2,1,0))
    e3 = -e3
    log.debug('Static hyper polarizability tensor\n%s', e3)
    return e3


# Solve the frequency-dependent CPHF problem
# [A-wI, B   ] [X] + [h1] = [0]
# [B   , A+wI] [Y]   [h1]   [0]

# TODO: new solver with Arnoldi iteration.
# The krylov solver in this implementation often fails. see
# https://github.com/pyscf/pyscf/issues/507
def __FIXME_cphf_with_freq(mf, mo_energy, mo_occ, h1, freq=0,
                   max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_ai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])

    # e_ai - freq may produce very small elements which can cause numerical
    # issue in krylov solver
    LEVEL_SHIF = 0.1
    diag = (e_ai - freq,
            e_ai + freq)
    diag[0][diag[0] < LEVEL_SHIF] += LEVEL_SHIF
    diag[1][diag[1] < LEVEL_SHIF] += LEVEL_SHIF

    nvir, nocc = e_ai.shape
    mo_coeff = mf.mo_coeff
    nao, nmo = mo_coeff.shape
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    h1 = h1.reshape(-1,nvir,nocc)
    ncomp = h1.shape[0]

    mo1base = numpy.stack((-h1/diag[0],
                           -h1/diag[1]), axis=1)
    mo1base = mo1base.reshape(ncomp,nocc*nvir*2)

    vresp = mf.gen_response(hermi=0)
    def vind(xys):
        nz = len(xys)
        dms = numpy.empty((nz,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            # *2 for double occupancy
            dmx = reduce(numpy.dot, (orbv, x  *2, orbo.T))
            dmy = reduce(numpy.dot, (orbo, y.T*2, orbv.T))
            dms[i] = dmx + dmy  # AX + BY

        v1ao = vresp(dms)
        v1vo = lib.einsum('xpq,pi,qj->xij', v1ao, orbv, orbo)  # ~c1
        v1ov = lib.einsum('xpq,pi,qj->xji', v1ao, orbo, orbv)  # ~c1^T

        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            v1vo[i] += (e_ai - freq - diag[0]) * x
            v1vo[i] /= diag[0]
            v1ov[i] += (e_ai + freq - diag[1]) * y
            v1ov[i] /= diag[1]
        v = numpy.stack((v1vo, v1ov), axis=1)
        return v.reshape(nz,-1)

    # FIXME: krylov solver is not accurate enough for many freqs. Using tight
    # tol and lindep could offer small help. A better linear equation solver
    # is needed.
    mo1 = lib.krylov(vind, mo1base, tol=tol, max_cycle=max_cycle,
                     hermi=hermi, lindep=1e-18, verbose=log)
    mo1 = mo1.reshape(-1,2,nvir,nocc)
    log.timer('krylov solver in CPHF', *t0)

    dms = numpy.empty((ncomp,nao,nao))
    for i in range(ncomp):
        x, y = mo1[i]
        dmx = reduce(numpy.dot, (orbv, x  *2, orbo.T))
        dmy = reduce(numpy.dot, (orbo, y.T*2, orbv.T))
        dms[i] = dmx + dmy
    mo_e1 = lib.einsum('xpq,pi,qj->xij', vresp(dms), orbo, orbo)
    mo1 = (mo1[:,0], mo1[:,1])
    return mo1, mo_e1

def cphf_with_freq(mf, mo_energy, mo_occ, h1, freq=0,
                   max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN):
    # lib.krylov often fails, newton_krylov solver from relatively new scipy
    # library is needed.
    from scipy.optimize import newton_krylov
    log = logger.new_logger(verbose=verbose)
    t0 = (time.clock(), time.time())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_ai = lib.direct_sum('a-i->ai', mo_energy[viridx], mo_energy[occidx])

    # e_ai - freq may produce very small elements which can cause numerical
    # issue in krylov solver
    LEVEL_SHIF = 0.1
    diag = (e_ai - freq,
            e_ai + freq)
    diag[0][diag[0] < LEVEL_SHIF] += LEVEL_SHIF
    diag[1][diag[1] < LEVEL_SHIF] += LEVEL_SHIF

    nvir, nocc = e_ai.shape
    mo_coeff = mf.mo_coeff
    nao, nmo = mo_coeff.shape
    orbv = mo_coeff[:,viridx]
    orbo = mo_coeff[:,occidx]
    h1 = h1.reshape(-1,nvir,nocc)
    ncomp = h1.shape[0]

    rhs = numpy.stack((-h1, -h1), axis=1)
    rhs = rhs.reshape(ncomp,nocc*nvir*2)
    mo1base = numpy.stack((-h1/diag[0],
                           -h1/diag[1]), axis=1)
    mo1base = mo1base.reshape(ncomp,nocc*nvir*2)

    vresp = mf.gen_response(hermi=0)
    def vind(xys):
        nz = len(xys)
        dms = numpy.empty((nz,nao,nao))
        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            # *2 for double occupancy
            dmx = reduce(numpy.dot, (orbv, x  *2, orbo.T))
            dmy = reduce(numpy.dot, (orbo, y.T*2, orbv.T))
            dms[i] = dmx + dmy  # AX + BY

        v1ao = vresp(dms)
        v1vo = lib.einsum('xpq,pi,qj->xij', v1ao, orbv, orbo)  # ~c1
        v1ov = lib.einsum('xpq,pi,qj->xji', v1ao, orbo, orbv)  # ~c1^T

        for i in range(nz):
            x, y = xys[i].reshape(2,nvir,nocc)
            v1vo[i] += (e_ai - freq) * x
            v1ov[i] += (e_ai + freq) * y
        v = numpy.stack((v1vo, v1ov), axis=1)
        return v.reshape(nz,-1) - rhs

    mo1 = newton_krylov(vind, mo1base, f_tol=tol)
    mo1 = mo1.reshape(-1,2,nvir,nocc)
    log.timer('krylov solver in CPHF', *t0)

    dms = numpy.empty((ncomp,nao,nao))
    for i in range(ncomp):
        x, y = mo1[i]
        dmx = reduce(numpy.dot, (orbv, x  *2, orbo.T))
        dmy = reduce(numpy.dot, (orbo, y.T*2, orbv.T))
        dms[i] = dmx + dmy
    mo_e1 = lib.einsum('xpq,pi,qj->xij', vresp(dms), orbo, orbo)
    mo1 = (mo1[:,0], mo1[:,1])
    return mo1, mo_e1


def polarizability_with_freq(polobj, freq=None):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    occidx = mo_occ > 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:,~occidx]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_orig(charge_center):
        int_r = mol.intor_symmetric('int1e_r', comp=3)

    h1 = lib.einsum('xpq,pi,qj->xij', int_r, orbv.conj(), orbo)
    mo1 = cphf_with_freq(mf, mo_energy, mo_occ, h1, freq,
                         polobj.max_cycle_cphf, polobj.conv_tol, verbose=log)[0]

    e2 =  numpy.einsum('xpi,ypi->xy', h1, mo1[0])
    e2 += numpy.einsum('xpi,ypi->xy', h1, mo1[1])

    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 *= -2
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
        vresp = mf.gen_response(hermi=1)
        occidx = mo_occ > 0
        orbo = mo_coeff[:, occidx]
        nocc = orbo.shape[1]
        nao, nmo = mo_coeff.shape
        def vind(mo1):
            dm1 = lib.einsum('xai,pa,qi->xpq', mo1.reshape(-1,nmo,nocc), mo_coeff,
                             orbo.conj())
            dm1 = (dm1 + dm1.transpose(0,2,1).conj()) * 2
            v1mo = lib.einsum('xpq,pi,qj->xij', vresp(dm1), mo_coeff.conj(), orbo)
            return v1mo.ravel()
        return vind


    polarizability = polarizability
    polarizability_with_freq = polarizability_with_freq

    hyper_polarizability = hyper_polarizability

from pyscf import scf
scf.hf.RHF.Polarizability = lib.class_as_method(Polarizability)

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = '''h  ,  0.   0.   0.
                  F  ,  0.   0.   .917'''
    mol.basis = '631g'
    mol.build()

    mf = scf.RHF(mol).run(conv_tol=1e-14)
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

    mol = gto.M(atom='''O      0.   0.       0.
                        H      0.  -0.757    0.587
                        H      0.   0.757    0.587''',
                basis='6-31g')
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    print(Polarizability(mf).polarizability())
    print(Polarizability(mf).polarizability_with_freq(freq= 0.))

    print(Polarizability(mf).polarizability_with_freq(freq= 0.1))
    print(Polarizability(mf).polarizability_with_freq(freq=-0.1))

