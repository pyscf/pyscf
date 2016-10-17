#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import copy
import time
import ctypes
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.df import _ri
from pyscf.df import addons

libri = pyscf.lib.load_library('libri')

def format_aux_basis(mol, auxbasis='weigend+etb'):
    '''Generate a fake Mole object which uses the density fitting auxbasis as
    the basis sets
    '''
    pmol = copy.copy(mol)  # just need shallow copy

    if auxbasis == 'weigend+etb':
        pmol._basis = pmol.format_basis(addons.aug_etb_for_dfbasis(mol))
    elif isinstance(auxbasis, str):
        uniq_atoms = set([a[0] for a in mol._atom])
        pmol._basis = pmol.format_basis(dict([(a, auxbasis)
                                              for a in uniq_atoms]))
    else:
        pmol._basis = pmol.format_basis(auxbasis)
    pmol._atm, pmol._bas, pmol._env = \
            pmol.make_env(mol._atom, pmol._basis, mol._env[:gto.PTR_ENV_START])
    pmol._built = True
    logger.debug(mol, 'aux basis %s, num shells = %d, num cGTO = %d',
                 auxbasis, pmol.nbas, pmol.nao_nr())
    return pmol


# (ij|L)
def aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1', comp=1, out=None):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.
    '''
    atm, bas, env, ao_loc = _env_and_aoloc(intor, mol, auxmol)
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
    return _ri.nr_auxe2(intor, atm, bas, env, shls_slice, ao_loc,
                        aosym, comp, out=out)

# (L|ij)
def aux_e1(mol, auxmol, intor='cint3c2e_sph', aosym='s1', comp=1, out=None):
    '''3-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    if comp == 1:
        out = aux_e2(mol, auxmol, intor, aosym, comp, out).T
    else:
        out = aux_e2(mol, auxmol, intor, aosym, comp, out).transpose(0,2,1)
    return out


def fill_2c2e(mol, auxmol, intor='cint2c2e_sph', comp=1, hermi=1, out=None):
    '''2-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.
    '''
    return auxmol.intor(intor, comp=comp, hermi=hermi, out=out)


# Note the temporary memory usage is about twice as large as the return cderi
# array
def cholesky_eri(mol, auxbasis='weigend+etb', auxmol=None, verbose=0):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    t0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    if auxmol is None:
        auxmol = format_aux_basis(mol, auxbasis)

    j2c = fill_2c2e(mol, auxmol, intor='cint2c2e_sph')
    log.debug('size of aux basis %d', j2c.shape[0])
    t1 = log.timer('2c2e', *t0)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
    except scipy.linalg.LinAlgError
        j2c[numpy.diag_indices(j2c.shape[1])] += 1e-14
        low = scipy.linalg.cholesky(j2c, lower=True)
    j2c = None
    t1 = log.timer('Cholesky 2c2e', *t1)

    j3c = aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s2ij')
    j3cT = j3c.T
    t1 = log.timer('3c2e', *t1)
    cderi = scipy.linalg.solve_triangular(low, j3c.T, lower=True,
                                          overwrite_b=True)
    j3c = None
    if cderi.flags.f_contiguous:
        cderi = pyscf.lib.transpose(cderi.T)
    log.timer('cholesky_eri', *t0)
    return cderi


def _env_and_aoloc(intor, mol, auxmol):
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    if 'ssc' in intor:
        ao_loc = mol.ao_loc_nr()
        nao = ao_loc[-1]
        ao_loc = numpy.hstack((ao_loc[:-1], nao+auxmol.ao_loc_nr(cart=True)))
    else:
        ao_loc = gto.moleintor.make_loc(bas, intor)
    return atm, bas, env, ao_loc


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import ao2mo
    mol = gto.Mole()
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
            gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                              auxmol._atm, auxmol._bas, auxmol._env)
    eri0 = numpy.empty((nao,nao,naoaux))
    pi = 0
    for i in range(mol.nbas):
        pj = 0
        for j in range(mol.nbas):
            pk = 0
            for k in range(mol.nbas, mol.nbas+auxmol.nbas):
                shls = (i, j, k)
                buf = gto.moleintor.getints_by_shell('cint3c2e_sph',
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
            buf = gto.moleintor.getints_by_shell('cint2c2e_sph',
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
