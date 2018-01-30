#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import numpy
import numpy as np

from pyscf import lib
from pyscf.lib import logger

from pyscf.pbc.lib import kpts_helper 
from pyscf.pbc.cc.kccsd_rhf import get_nocc, get_nmo

'''
kpoint-adapted and spin-adapted MP2
t2[i,j,a,b] = <ij|ab> / D_ij^ab

t2 and eris are never stored in full, only a partial 
eri of size (nkpts,nocc,nocc,nvir,nvir)
'''

def kernel(mp, mo_energy, mo_coeff, verbose=logger.NOTE):
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    nkpts = mp.nkpts

    eia = numpy.zeros((nocc,nvir))
    eijab = numpy.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    emp2 = 0.
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)
    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]
            orbo_i = mo_coeff[ki][:,:nocc]
            orbo_j = mo_coeff[kj][:,:nocc]
            orbv_a = mo_coeff[ka][:,nocc:]
            orbv_b = mo_coeff[kb][:,nocc:]
            oovv_ij[ka] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                            (mp.kpts[ki],mp.kpts[ka],mp.kpts[kj],mp.kpts[kb]), 
                            compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]
            eia = mo_energy[ki][:nocc].reshape(-1,1) - mo_energy[ka][nocc:]
            ejb = mo_energy[kj][:nocc].reshape(-1,1) - mo_energy[kb][nocc:]
            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2_ijab = np.conj(oovv_ij[ka]/eijab)
            woovv = 2*oovv_ij[ka] - oovv_ij[kb].transpose(0,1,3,2)
            emp2 += np.einsum('ijab,ijab', t2_ijab, woovv).real

    emp2 /= nkpts

    return emp2, None 


class KMP2(lib.StreamObject):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.nkpts = len(self.kpts)
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_energy = mf.mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.emp2 = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())

    nocc = property(get_nocc)
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    nmo = property(get_nmo)
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo

    def kernel(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log = logger.Logger(self.stdout, self.verbose)
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError

        self.emp2, self.t2 = \
                kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
        logger.log(self, 'KMP2 energy = %.15g', self.emp2)
        self.e_corr = self.emp2
        return self.emp2, self.t2


def _mem_usage(nkpts, nocc, nvir):
    nmo = nocc + nvir
    basic = (nkpts*nocc**2*nvir**2*2)*16 / 1e6
    incore = outcore = basic 
    return incore, outcore, basic


if __name__ == '__main__':
    from pyscf.pbc import gto, scf, mp 

    cell = gto.Cell()
    cell.atom='''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 5
    cell.build()

    # Running HF and MP2 with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
    ehf = kmf.kernel()

    mymp = mp.KMP2(kmf)
    emp2, t2 = mymp.kernel()
    print(emp2 - -0.204721432828996)

