#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import ctypes
import _ctypes
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
import pyscf.gto
from pyscf.scf import _vhf

libri = pyscf.lib.load_library('libri')
def _fpointer(name):
    return ctypes.c_void_p(_ctypes.dlsym(libri._handle, name))

def format_aux_basis(mol, auxbasis='weigend'):
    pmol = pyscf.gto.Mole()
    pmol.verbose = 0
    pmol.atom = mol.atom
    pmol.basis = auxbasis
    pmol.spin = mol.spin
    pmol.charge = mol.charge
    pmol.build(False, False)
    pmol.verbose = mol.verbose
    logger.debug(mol, 'aux basis %s, num shells = %d, num cGTO = %d',
                 auxbasis, pmol.nbas, pmol.nao_nr())
    return pmol


# (ij|L)
def aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1', comp=1, hermi=0):
    assert(aosym in ('s1', 's2ij'))
    atm, bas, env = \
            pyscf.gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                    auxmol._atm, auxmol._bas, auxmol._env)
    c_atm = numpy.array(atm, dtype=numpy.int32)
    c_bas = numpy.array(bas, dtype=numpy.int32)
    c_env = numpy.array(env)
    natm = ctypes.c_int(mol.natm)
    nbas = ctypes.c_int(mol.nbas)

    nao = mol.nao_nr()
    naoaux = auxmol.nao_nr()
    if aosym == 's1':
        eri = numpy.empty((nao*nao,naoaux))
        fill = _fpointer('RIfill_s1_auxe2')
    else:
        eri = numpy.empty((nao*(nao+1)//2,naoaux))
        fill = _fpointer('RIfill_s2ij_auxe2')
    fintor = _fpointer(intor)
    cintopt = _vhf.make_cintopt(c_atm, c_bas, c_env, intor)
    libri.RInr_3c2e_auxe2_drv(fintor, fill,
                              eri.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_int(0), ctypes.c_int(mol.nbas),
                              ctypes.c_int(mol.nbas), ctypes.c_int(auxmol.nbas),
                              ctypes.c_int(1), cintopt,
                              c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                              c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                              c_env.ctypes.data_as(ctypes.c_void_p))
    libri.CINTdel_optimizer(ctypes.byref(cintopt))
    return eri


# (L|ij)
def aux_e1(mol, auxmol, intor='cint3c2e_sph', aosym='s1', comp=1, hermi=0):
    eri = aux_e2(mol, auxmol, intor, aosym, comp, hermi)
    naux = eri.shape[1]
    return pyscf.lib.transpose(eri.reshape(-1,naux))


def fill_2c2e(mol, auxmol, intor='cint2c2e_sph'):
    c_atm = numpy.array(auxmol._atm, dtype=numpy.int32)
    c_bas = numpy.array(auxmol._bas, dtype=numpy.int32)
    c_env = numpy.array(auxmol._env)
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    naoaux = auxmol.nao_nr()
    eri = numpy.empty((naoaux,naoaux))
    libri.RInr_fill2c2e_sph(eri.ctypes.data_as(ctypes.c_void_p),
                            ctypes.c_int(0), ctypes.c_int(auxmol.nbas),
                            c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                            c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                            c_env.ctypes.data_as(ctypes.c_void_p))
    return eri


def cholesky_eri(mol, auxbasis='weigend', verbose=0):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    t0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    auxmol = format_aux_basis(mol, auxbasis)

    j2c = fill_2c2e(mol, auxmol, intor='cint2c2e_sph')
    log.debug('size of aux basis %d', j2c.shape[0])
    t1 = log.timer('2c2e', *t0)
    low = scipy.linalg.cholesky(j2c, lower=True)
    j2c = None
    t1 = log.timer('Cholesky 2c2e', *t1)

    j3c = aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s2ij')
    t1 = log.timer('3c2e', *t1)
    cderi = scipy.linalg.solve_triangular(low, j3c.T, lower=True,
                                          overwrite_b=True)
    j3c = None
    # solve_triangular return cderi in Fortran order
    cderi = pyscf.lib.transpose(cderi.T)
    log.timer('cholesky_eri', *t0)
    return cderi



if __name__ == '__main__':
    from pyscf import scf
    from pyscf import ao2mo
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom.extend([
        ["H", (0,  0, 0  )],
        ["H", (0,  0, 1  )],
    ])
    mol.basis = 'cc-pvdz'
    mol.build()

    auxmol = format_aux_basis(mol)
    j3c = aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1')
    nao = mol.nao_nr()
    naoaux = auxmol.nao_nr()
    j3c = j3c.reshape(nao,nao,naoaux)

    atm, bas, env = \
            pyscf.gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                    auxmol._atm, auxmol._bas, auxmol._env)
    eri0 = numpy.empty((nao,nao,naoaux))
    libri.CINTcgto_spheric.restype = ctypes.c_int
    pi = 0
    for i in range(mol.nbas):
        pj = 0
        for j in range(mol.nbas):
            pk = 0
            for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                shls = (i, j, k)
                buf = pyscf.gto.moleintor.getints_by_shell('cint3c2e_sph',
                                                           shls, atm, bas, env)
                di, dj, dk = buf.shape
                eri0[pi:pi+di,pj:pj+dj,pk:pk+dk] = buf
                pk += dk
            pj += dj
        pi += di
    print(numpy.allclose(eri0, j3c))

    j2c = fill_2c2e(mol, auxmol)
    eri0 = numpy.empty_like(j2c)
    pi = 0
    for i in range(mol.nbas, len(bas)):
        pj = 0
        for j in range(mol.nbas, len(bas)):
            shls = (i, j)
            buf = pyscf.gto.moleintor.getints_by_shell('cint2c2e_sph',
                                                       shls, atm, bas, env)
            di, dj = buf.shape
            eri0[pi:pi+di,pj:pj+dj] = buf
            pj += dj
        pi += di
    print(numpy.allclose(eri0, j2c))

    j3c = aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s2ij')
    cderi = cholesky_eri(mol)
    eri0 = numpy.einsum('pi,pk->ik', cderi, cderi)
    eri1 = numpy.einsum('ik,kl->il', j3c, numpy.linalg.inv(j2c))
    eri1 = numpy.einsum('ip,kp->ik', eri1, j3c)
    print(numpy.allclose(eri1, eri0))
    eri0 = pyscf.ao2mo.restore(1, eri0, nao)

    mf = scf.RHF(mol)
    ehf0 = mf.scf()

    nao = mf.mo_energy.size
    eri1 = ao2mo.restore(1, mf._eri, nao)
    print(numpy.linalg.norm(eri1-eri0))

    mf._eri = ao2mo.restore(8, eri0, nao)
    ehf1 = mf.scf()

    mf = scf.density_fit(scf.RHF(mol))
    ehf2 = mf.scf()
    print(ehf0, ehf1, ehf2)
