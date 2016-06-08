#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Integral transformation with analytic Fourier transformation
'''

import time
import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger


def get_eri(pwdf, kpts=None, compact=False):
    cell = pwdf.cell
    if kpts is None:
        kptijkl = numpy.zeros((4,3))
    elif numpy.shape(kpts) == (3,):
        kptijkl = numpy.vstack([kpts]*4)
    else:
        kptijkl = numpy.reshape(kpts, (4,3))

    kpti, kptj, kptk, kptl = kptijkl
    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2

####################
# gamma point, the integral is real and with s4 symmetry
    if abs(kptijkl).sum() < 1e-9:
        eriR = numpy.zeros((nao_pair,nao_pair))
        max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
        trilidx = numpy.tril_indices(nao)
        for pqkR, pqkI, coulG \
                in pwdf.pw_loop(cell, pwdf.gs, kptijkl[:2], max_memory):
            pqkR = numpy.asarray(pqkR.reshape(nao,nao,-1)[trilidx], order='C')
            pqkI = numpy.asarray(pqkI.reshape(nao,nao,-1)[trilidx], order='C')
            coulG = numpy.sqrt(coulG)
            pqkR *= coulG
            pqkI *= coulG
            lib.dot(pqkR, pqkR.T, 1, eriR, 1)
            lib.dot(pqkI, pqkI.T, 1, eriR, 1)
        pqkR = LkR = pqkI = LkI = coulG = None
        return eriR

####################
# (kpt) i == j == k == l != 0
# (kpt) i == l && j == k && i != j && j != k  =>
#
# complex integrals, N^4 elements
    elif (abs(kpti-kptl).sum() < 1e-9) and (abs(kptj-kptk).sum() < 1e-9):
        eriR = numpy.zeros((nao**2,nao**2))
        eriI = numpy.zeros((nao**2,nao**2))
        max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .8
        for pqkR, pqkI, coulG \
                in pwdf.pw_loop(cell, pwdf.gs, kptijkl[:2], max_memory):
            coulG = numpy.sqrt(coulG)
            pqkR *= coulG
            pqkI *= coulG
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            lib.dot(pqkR, pqkR.T, 1, eriR, 1)
            lib.dot(pqkI, pqkI.T, 1, eriR, 1)
            lib.dot(pqkI, pqkR.T, 1, eriI, 1)
            lib.dot(pqkR, pqkI.T,-1, eriI, 1)
        return (eriR.reshape((nao,)*4).transpose(0,1,3,2) +
                eriI.reshape((nao,)*4).transpose(0,1,3,2)*1j).reshape(nao**2,-1)

####################
# aosym = s1, complex integrals
#
# If kpti == kptj, (kptl-kptk)*a has to be multiples of 2pi because of the wave
# vector symmetry.  k is a fraction of reciprocal basis, 0 < k/b < 1, by definition.
# So  kptl/b - kptk/b  must be -1 < k/b < 1.  =>  kptl == kptk
#
    else:
        eriR = numpy.zeros((nao**2,nao**2))
        eriI = numpy.zeros((nao**2,nao**2))
        max_memory = (pwdf.max_memory - lib.current_memory()[0]) * .4
        for (pqkR, pqkI, coulG), (rskR, rskI, coulG1) in \
                lib.izip(pwdf.pw_loop(cell, pwdf.gs, kptijkl[:2], max_memory),
                         pwdf.pw_loop(cell, pwdf.gs,-kptijkl[2:], max_memory)):
            pqkR *= coulG
            pqkI *= coulG
# rho_pq(G+k_pq) * conj(rho_rs(G-k_rs))
            lib.dot(pqkR, rskR.T, 1, eriR, 1)
            lib.dot(pqkI, rskI.T, 1, eriR, 1)
            lib.dot(pqkI, rskR.T, 1, eriI, 1)
            lib.dot(pqkR, rskI.T,-1, eriI, 1)
        return (eriR+eriI*1j)


def general(pwdf, mo_coeffs, kpts=None, compact=False):
    from pyscf.pbc.df import xdf_ao2mo
    return xdf_ao2mo.general(pwdf, mo_coeffs, kpts, compact)


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    from pyscf.pbc.df import pwdf

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.h = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)

    nao = cell.nao_nr()
    numpy.random.seed(1)
    kpts = numpy.random.random((4,3))
    #kpts[3] = -numpy.einsum('ij->j', kpts[:3])
    kpts[3] = kpts[0]
    kpts[2] = kpts[1]
    with_df = pwdf.PWDF(cell)
    with_df.kpts = kpts
    mo =(numpy.random.random((nao,nao)) +
         numpy.random.random((nao,nao))*1j)
    eri = with_df.get_eri(kpts).reshape((nao,)*4)
    eri0 = numpy.einsum('pjkl,pi->ijkl', eri , mo.conj())
    eri0 = numpy.einsum('ipkl,pj->ijkl', eri0, mo       )
    eri0 = numpy.einsum('ijpl,pk->ijkl', eri0, mo.conj())
    eri0 = numpy.einsum('ijkp,pl->ijkl', eri0, mo       )
    eri1 = with_df.ao2mo(mo, kpts)
    print abs(eri1-eri0).sum()
