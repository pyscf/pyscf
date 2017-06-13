#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Wave Function Stability Analysis

Ref.
JCP, 66, 3045
JCP, 104, 9047
'''

import numpy
import scipy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo

# Real RHF
# Real UHF
# Real GHF

def rhf_stability(mf, internal=True, external=False, verbose=None):
    if internal:
        rhf_internal(mf, verbose)
    if external:
        rhf_external(mf, verbose)
    # TODO: return eigenvectors of modified Fock matrix
    return mf

def uhf_stability(mf, internal=True, external=False, verbose=None):
    if internal:
        uhf_internal(mf, verbose)
    if external:
        uhf_external(mf, verbose)
    # TODO: return eigenvectors of modified Fock matrix
    return mf

def rhf_internal(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nmo = mo_coeff.shape[1]
    nocc = numpy.count_nonzero(mo_occ)
    nvir = nmo - nocc

    eri_mo = ao2mo.full(mol, mo_coeff)
    eri_mo = ao2mo.restore(1, eri_mo, nmo)
    eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
    # A
    h = numpy.einsum('ckld->kcld', eri_mo[nocc:,:nocc,:nocc,nocc:]) * 2
    h-= numpy.einsum('cdlk->kcld', eri_mo[nocc:,nocc:,:nocc,:nocc])
    for a in range(nvir):
        for i in range(nocc):
            h[i,a,i,a] += eai[a,i]
    # B
    h+= numpy.einsum('ckdl->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc]) * 2
    h-= numpy.einsum('cldk->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc])

    nov = nocc * nvir
    e = scipy.linalg.eigh(h.reshape(nov,nov))[0]
    log.debug('rhf_internal: lowest eig = %g', e[0])
    if e[0] < -1e-5:
        log.note('RHF wavefunction has an internal instablity')
    else:
        log.note('RHF wavefunction is stable in the intenral stabliaty analysis')
    return e[0]

def rhf_external(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nmo = mo_coeff.shape[1]
    nocc = numpy.count_nonzero(mo_occ)
    nvir = nmo - nocc

    eri_mo = ao2mo.full(mol, mo_coeff)
    eri_mo = ao2mo.restore(1, eri_mo, nmo)
    eai = lib.direct_sum('a-i->ai', mo_energy[nocc:], mo_energy[:nocc])
    # A
    h = numpy.einsum('ckld->kcld', eri_mo[nocc:,:nocc,:nocc,nocc:]) * 2
    h-= numpy.einsum('cdlk->kcld', eri_mo[nocc:,nocc:,:nocc,:nocc])
    for a in range(nvir):
        for i in range(nocc):
            h[i,a,i,a] += eai[a,i]
    # B
    h-= numpy.einsum('ckdl->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc]) * 2
    h+= numpy.einsum('cldk->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc])

    nov = nocc * nvir
    e1 = scipy.linalg.eigh(h.reshape(nov,nov))[0]
    log.debug('rhf_external: lowest eig = %g', e1[0])
    if e1[0] < -1e-5:
        log.note('RHF wavefunction has an RHF real -> complex instablity')
    else:
        log.note('RHF wavefunction is stable in the RHF real -> complex stabliaty analysis')

    h =-numpy.einsum('cdlk->kcld', eri_mo[nocc:,nocc:,:nocc,:nocc])
    for a in range(nvir):
        for i in range(nocc):
            h[i,a,i,a] += eai[a,i]
    h-= numpy.einsum('cldk->kcld', eri_mo[nocc:,:nocc,nocc:,:nocc])
    e3 = scipy.linalg.eigh(h.reshape(nov,nov))[0]
    log.debug('rhf_external: lowest eig of H = %g', e3[0])
    if e3[0] < -1e-5:
        log.note('RHF wavefunction has an RHF -> UHF instablity.')
    else:
        log.note('RHF wavefunction is stable in the RHF -> UHF stabliaty analysis')
    return e1[0], e3[0]

def uhf_internal(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    mol = mf.mol
    mo_a, mo_b = mf.mo_coeff
    mo_ea, mo_eb = mf.mo_energy
    mo_occa, mo_occb = mf.mo_occ
    nmo = mo_a.shape[1]
    nocca = numpy.count_nonzero(mo_occa)
    noccb = numpy.count_nonzero(mo_occb)
    nvira = nmo - nocca
    nvirb = nmo - noccb

    eri_aa = ao2mo.restore(1, ao2mo.full(mol, mo_a), nmo)
    eri_ab = ao2mo.restore(1, ao2mo.general(mol, [mo_a,mo_a,mo_b,mo_b]), nmo)
    eri_bb = ao2mo.restore(1, ao2mo.full(mol, mo_b), nmo)
    # alpha -> alpha
    haa = numpy.einsum('aijb->iajb', eri_aa[nocca:,:nocca,:nocca,nocca:]) * 2
    haa-= numpy.einsum('abji->iajb', eri_aa[nocca:,nocca:,:nocca,:nocca])
    haa-= numpy.einsum('ajbi->iajb', eri_aa[nocca:,:nocca,nocca:,:nocca])
    for a in range(nvira):
        for i in range(nocca):
            haa[i,a,i,a] += mo_ea[nocca+a] - mo_ea[i]
    # beta -> beta
    hbb = numpy.einsum('aijb->iajb', eri_bb[noccb:,:noccb,:noccb,noccb:]) * 2
    hbb-= numpy.einsum('abji->iajb', eri_bb[noccb:,noccb:,:noccb,:noccb])
    hbb-= numpy.einsum('ajbi->iajb', eri_bb[noccb:,:noccb,noccb:,:noccb])
    for a in range(nvirb):
        for i in range(noccb):
            hbb[i,a,i,a] += mo_eb[noccb+a] - mo_eb[i]
    # (alpha -> alpha, beta -> beta)
    hab = numpy.einsum('aijb->iajb', eri_ab[nocca:,:nocca,:noccb,noccb:]) * 2

    nova = nocca * nvira
    novb = noccb * nvirb
    hall = numpy.empty((nova+novb,nova+novb))
    hall[:nova,:nova] = haa.reshape(nova,nova)
    hall[nova:,nova:] = hbb.reshape(novb,novb)
    hall[:nova,nova:] = hab.reshape(nova,novb)
    hall[nova:,:nova] = hab.reshape(nova,novb).T
    e = scipy.linalg.eigh(hall)[0]
    log.debug('uhf_internal: lowest eig of H = %g', e[0])
    if e[0] < -1e-5:
        log.note('UHF wavefunction has an internal instablity')
    else:
        log.note('UHF wavefunction is stable in the intenral stabliaty analysis')
    return e[0]

def uhf_external(mf, verbose=None):
    log = logger.new_logger(mf, verbose)
    mol = mf.mol
    mo_a, mo_b = mf.mo_coeff
    mo_ea, mo_eb = mf.mo_energy
    mo_occa, mo_occb = mf.mo_occ
    nmo = mo_a.shape[1]
    nocca = numpy.count_nonzero(mo_occa)
    noccb = numpy.count_nonzero(mo_occb)
    nvira = nmo - nocca
    nvirb = nmo - noccb

    eri_aa = ao2mo.restore(1, ao2mo.full(mol, mo_a), nmo)
    eri_ab = ao2mo.restore(1, ao2mo.general(mol, [mo_a,mo_a,mo_b,mo_b]), nmo)
    eri_bb = ao2mo.restore(1, ao2mo.full(mol, mo_b), nmo)
    # alpha -> alpha
    haa =-numpy.einsum('abji->iajb', eri_aa[nocca:,nocca:,:nocca,:nocca])
    haa+= numpy.einsum('ajbi->iajb', eri_aa[nocca:,:nocca,nocca:,:nocca])
    for a in range(nvira):
        for i in range(nocca):
            haa[i,a,i,a] += mo_ea[nocca+a] - mo_ea[i]
    # beta -> beta
    hbb =-numpy.einsum('abji->iajb', eri_bb[noccb:,noccb:,:noccb,:noccb])
    hbb+= numpy.einsum('ajbi->iajb', eri_bb[noccb:,:noccb,noccb:,:noccb])
    for a in range(nvirb):
        for i in range(noccb):
            hbb[i,a,i,a] += mo_eb[noccb+a] - mo_eb[i]

    nova = nocca * nvira
    novb = noccb * nvirb
    hall = numpy.zeros((nova+novb,nova+novb))
    hall[:nova,:nova] = haa.reshape(nova,nova)
    hall[nova:,nova:] = hbb.reshape(novb,novb)
    e1 = scipy.linalg.eigh(hall)[0]
    log.debug('uhf_external: lowest eig of H = %g', e1[0])
    if e1[0] < -1e-5:
        log.note('UHF wavefunction has an UHF real -> complex instablity')
    else:
        log.note('UHF wavefunction is stable in the UHF real -> complex stabliaty analysis')

    h11 =-numpy.einsum('abji->iajb', eri_ab[nocca:,nocca:,:noccb,:noccb])
    for a in range(nvira):
        for i in range(noccb):
            h11[i,a,i,a] += mo_eb[nocca+a] - mo_eb[i]
    h22 =-numpy.einsum('jiab->iajb', eri_ab[:nocca,:nocca,noccb:,noccb:])
    for a in range(nvirb):
        for i in range(nocca):
            h22[i,a,i,a] += mo_ea[noccb+a] - mo_eb[i]
    h12 =-numpy.einsum('ajbi->iajb', eri_ab[nocca:,:nocca,noccb:,:noccb])
    h21 =-numpy.einsum('biaj->iajb', eri_ab[nocca:,:nocca,noccb:,:noccb])

    n1 = noccb * nvira
    n2 = nocca * nvirb
    hall = numpy.empty((n1+n2,n1+n2))
    hall[:n1,:n1] = h11.reshape(n1,n1)
    hall[n1:,n1:] = h22.reshape(n2,n2)
    hall[:n1,n1:] = h12.reshape(n1,n2)
    hall[n1:,:n1] = h21.reshape(n2,n1)
    e3 = scipy.linalg.eigh(hall)[0]
    log.debug('uhf_external: lowest eig of H = %g', e3[0])
    if e3[0] < -1e-5:
        log.note('UHF wavefunction has an UHF -> GHF instablity.')
    else:
        log.note('UHF wavefunction is stable in the UHF -> GHF stabliaty analysis')
    return e1[0], e3[0]


if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(atom='O 0 0 0; O 0 0 1.2222', basis='631g*')
    mf = scf.RHF(mol).run()
    rhf_stability(mf, True, True, verbose=5)

    mf = scf.UHF(mol).run()
    uhf_stability(mf, True, True, verbose=5)

    mol.spin = 2
    mf = scf.UHF(mol).run()
    uhf_stability(mf, True, True, verbose=5)

    mol = gto.M(atom='''
O1
O2  1  1.2227
O3  1  1.2227  2  114.0451
                ''', basis = '631g*')
    mf = scf.RHF(mol).run()
    rhf_stability(mf, True, True, verbose=5)

    mf = scf.UHF(mol).run()
    uhf_stability(mf, True, True, verbose=5)
