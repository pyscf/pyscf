#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import time
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.df import addons
from pyscf import __config__


MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)  # 2GB
LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-12)


# This funciton is aliased for backward compatibility.
format_aux_basis = addons.make_auxmol


def aux_e2(mol, auxmol, intor='int3c2e', aosym='s1', comp=None, out=None,
           cintopt=None):
    '''3-center AO integrals (ij|L), where L is the auxiliary basis.

    Kwargs:
        cintopt : Libcint-3.14 and newer version support to compute int3c2e
            without the opt for the 3rd index.  It can be precomputed to
            reduce the overhead of cintopt initialization repeatedly.

            cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
    '''
    from pyscf.gto.moleintor import getints, make_cintopt
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)

    # Extract the call of the two lines below
    #  pmol = gto.mole.conc_mol(mol, auxmol)
    #  return pmol.intor(intor, comp, aosym=aosym, shls_slice=shls_slice, out=out)
    intor = mol._add_suffix(intor)
    hermi = 0
    ao_loc = None
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    return getints(intor, atm, bas, env, shls_slice, comp, hermi, aosym,
                   ao_loc, cintopt, out)

def aux_e1(mol, auxmol, intor='int3c2e', aosym='s1', comp=None, out=None):
    '''3-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.

    Note aux_e1 is basically analogous to aux_e2 function. It can be viewed as
    the version of transposed aux_e2 tensor:
    if comp == 1:
        aux_e1 = aux_e2().T
    else:
        aux_e1 = aux_e2().transpose(0,2,1)

    The same arguments as function aux_e2 can be input to aux_e1.
    '''
    out = aux_e2(mol, auxmol, intor, aosym, comp, out)
    if out.ndim == 2:  # comp == 1
        out = out.T
    else:
        out = out.transpose(0,2,1)
    return out


def fill_2c2e(mol, auxmol, intor='int2c2e', comp=None, hermi=1, out=None):
    '''2-center 2-electron AO integrals for auxiliary basis (auxmol)
    '''
    return auxmol.intor(intor, comp=comp, hermi=hermi, out=out)


# Note the temporary memory usage is about twice as large as the return cderi
# array
def cholesky_eri(mol, auxbasis='weigend+etb', auxmol=None,
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 max_memory=MAX_MEMORY, verbose=0, fauxe2=aux_e2):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    from pyscf.df.outcore import _guess_shell_ranges
    assert(comp == 1)
    t0 = (time.clock(), time.time())
    log = logger.new_logger(mol, verbose)
    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    j2c = auxmol.intor(int2c, hermi=1)
    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        tag = 'cd'
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        idx = w > LINEAR_DEP_THR
        low = (v[:,idx] / numpy.sqrt(w[idx]))
        v = None
        tag = 'eig'
    j2c = None
    naoaux, naux = low.shape
    log.debug('size of aux basis %d', naux)
    t1 = log.timer_debug1('2c2e', *t0)

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = ao_loc[mol.nbas]

    if aosym == 's1':
        nao_pair = nao * nao
    else:
        nao_pair = nao * (nao+1) // 2

    cderi = numpy.empty((naux, nao_pair))

    max_words = max_memory*.98e6/8 - low.size - cderi.size
    # Divide by 3 because scipy.linalg.solve may create a temporary copy for
    # ints and return another copy for results
    buflen = min(max(int(max_words/naoaux/comp/3), 8), nao_pair)
    shranges = _guess_shell_ranges(mol, buflen, aosym)
    log.debug1('shranges = %s', shranges)

    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    bufs1 = numpy.empty((comp*max([x[2] for x in shranges]),naoaux))
    bufs2 = numpy.empty_like(bufs1)

    p1 = 0
    for istep, sh_range in enumerate(shranges):
        log.debug('int3c2e [%d/%d], AO [%d:%d], nrow = %d', \
                  istep+1, len(shranges), *sh_range)
        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        ints = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                       aosym, ao_loc, cintopt, out=bufs1)

        if ints.ndim == 3 and ints.flags.f_contiguous:
            ints = lib.transpose(ints.T, axes=(0,2,1), out=bufs2).reshape(naoaux,-1)
            bufs1, bufs2 = bufs2, bufs1
        else:
            ints = ints.reshape((-1,naoaux)).T

        p0, p1 = p1, p1 + nrow
        if tag == 'cd':
            if ints.flags.c_contiguous:
                ints = lib.transpose(ints, out=bufs2).T
                bufs1, bufs2 = bufs2, bufs1
            dat = scipy.linalg.solve_triangular(low, ints, lower=True,
                                                overwrite_b=True, check_finite=False)
            if dat.flags.f_contiguous:
                dat = lib.transpose(dat.T, out=bufs2)
            cderi[:,p0:p1] = dat
        else:
            dat = numpy.ndarray((naux, ints.shape[1]), buffer=bufs2)
            cderi[:,p0:p1] = lib.dot(low.T, ints, c=dat)
        dat = ints = None

    log.timer('cholesky_eri', *t0)
    return cderi

# Debug version of cholesky_eri. Note the temporary memory usage is about
# twice as large as the return cderi array
def cholesky_eri_debug(mol, auxbasis='weigend+etb', auxmol=None,
                       int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                       verbose=0, fauxe2=aux_e2):
    '''
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
    '''
    assert(comp == 1)
    t0 = (time.clock(), time.time())
    log = logger.new_logger(mol, verbose)
    if auxmol is None:
        auxmol = addons.make_auxmol(mol, auxbasis)

    j2c = auxmol.intor(int2c, hermi=1)
    naux = j2c.shape[0]
    log.debug('size of aux basis %d', naux)
    t1 = log.timer('2c2e', *t0)

    j3c = fauxe2(mol, auxmol, intor=int3c, aosym=aosym).reshape(-1,naux)
    t1 = log.timer('3c2e', *t1)

    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        j2c = None
        t1 = log.timer('Cholesky 2c2e', *t1)
        cderi = scipy.linalg.solve_triangular(low, j3c.T, lower=True,
                                              overwrite_b=True)
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        idx = w > LINEAR_DEP_THR
        v = (v[:,idx] / numpy.sqrt(w[idx]))
        cderi = lib.dot(v.T, j3c.T)

    j3c = None
    if cderi.flags.f_contiguous:
        cderi = lib.transpose(cderi.T)
    log.timer('cholesky_eri', *t0)
    return cderi


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
    j3c = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s1')
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
                buf = gto.moleintor.getints_by_shell('int3c2e_sph',
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
            buf = gto.moleintor.getints_by_shell('int2c2e_sph',
                                                 shls, atm, bas, env)
            di, dj = buf.shape
            eri0[pi:pi+di,pj:pj+dj] = buf
            pj += dj
        pi += di
    print(numpy.allclose(eri0, j2c))

    j3c = aux_e2(mol, auxmol, intor='int3c2e_sph', aosym='s2ij')
    cderi = cholesky_eri(mol, auxmol=auxmol)
    eri0 = numpy.einsum('pi,pk->ik', cderi, cderi)
    eri1 = numpy.einsum('ik,kl->il', j3c, numpy.linalg.inv(j2c))
    eri1 = numpy.einsum('ip,kp->ik', eri1, j3c)
    print(abs(eri1 - eri0).max())
    eri0 = ao2mo.restore(1, eri0, nao)

    mf = scf.RHF(mol)
    ehf0 = mf.scf()

    nao = mf.mo_energy.size
    eri1 = ao2mo.restore(1, mf._eri, nao)
    print(abs(eri1-eri0).max() - 0.0022142583265513105)

    mf._eri = ao2mo.restore(8, eri0, nao)
    ehf1 = mf.scf()

    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    ehf2 = mf.scf()

    mf = mf.density_fit(auxbasis='weigend')
    ehf3 = mf.scf()
    print(ehf0, ehf1, ehf2, ehf3)
