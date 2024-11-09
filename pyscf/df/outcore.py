#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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


import tempfile
import numpy
import scipy.linalg
import h5py
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf.df.incore import _eig_decompose, LINEAR_DEP_THR
from pyscf.df.addons import make_auxmol
from pyscf import __config__

MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)  # 2GB

#
# for auxe1 (P|ij)
#

def cholesky_eri(mol, erifile, auxbasis='weigend+etb', dataname='j3c', tmpdir=None,
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 max_memory=MAX_MEMORY, auxmol=None, verbose=logger.NOTE):
    '''3-index density-fitting tensor.
    '''
    assert (aosym in ('s1', 's2ij'))
    assert (comp == 1)
    log = logger.new_logger(mol, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())

    if auxmol is None:
        auxmol = make_auxmol(mol, auxbasis)

    if tmpdir is None:
        tmpdir = lib.param.TMPDIR
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    cholesky_eri_b(mol, swapfile.name, auxbasis, dataname,
                   int3c, aosym, int2c, comp, max_memory, auxmol, verbose=log)
    fswap = h5py.File(swapfile.name, 'r')
    time1 = log.timer('generate (ij|L) 1 pass', *time0)

    # Cannot let naoaux = auxmol.nao_nr() if auxbasis has linear dependence
    nao = mol.nao_nr()
    if aosym == 's1':
        nao_pair = nao * nao
    else:
        nao_pair = nao * (nao+1) // 2

    feri = _create_h5file(erifile, dataname)
    if comp == 1:
        naoaux = fswap['%s/0'%dataname].shape[0]
        h5d_eri = feri.create_dataset(dataname, (naoaux,nao_pair), 'f8')
    else:
        naoaux = fswap['%s/0'%dataname].shape[1]
        h5d_eri = feri.create_dataset(dataname, (comp,naoaux,nao_pair), 'f8')

    iolen = min(max(int(max_memory*.45e6/8/nao_pair), 28), naoaux)
    totstep = (naoaux+iolen-1)//iolen
    def load(row_slice):
        row0, row1 = row_slice
        return _load_from_h5g(fswap[dataname], row0, row1)

    ti0 = time1
    slices = list(lib.prange(0, naoaux, iolen))
    for istep, dat in enumerate(lib.map_with_prefetch(load, slices)):
        row0, row1 = slices[istep]
        nrow = row1 - row0
        if comp == 1:
            h5d_eri[row0:row1] = dat
        else:
            h5d_eri[:,row0:row1] = dat
        dat = None
        ti0 = log.timer('step 2 [%d/%d], [%d:%d], row = %d'%
                        (istep+1, totstep, row0, row1, nrow), *ti0)

    # A bug in NFS / HDF5 may cause .close() not to
    # flush properly, hanging the calculation. Flush manually
    fswap.flush()
    feri.flush()

    fswap.close()
    feri.close()
    log.timer('cholesky_eri', *time0)
    return erifile

def cholesky_eri_b(mol, erifile, auxbasis='weigend+etb', dataname='j3c',
                   int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                   max_memory=MAX_MEMORY, auxmol=None, decompose_j2c='CD',
                   lindep=LINEAR_DEP_THR, verbose=logger.NOTE):
    '''3-center 2-electron DF tensor. Similar to cholesky_eri while this
    function stores DF tensor in blocks.

    Args:
        dataname: string
            Dataset label of the DF tensor in HDF5 file.
        decompose_j2c: string
            The method to decompose the metric defined by int2c. It can be set
            to CD (cholesky decomposition) or ED (eigenvalue decomposition).
        lindep : float
            The threshold to discard linearly dependent basis when decompose_j2c
            is set to ED.
    '''
    assert (aosym in ('s1', 's2ij'))
    log = logger.new_logger(mol, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())

    if auxmol is None:
        auxmol = make_auxmol(mol, auxbasis)
    j2c = auxmol.intor(int2c, hermi=1)
    log.debug('size of aux basis %d', j2c.shape[0])
    time1 = log.timer('2c2e', *time0)
    decompose_j2c = decompose_j2c.upper()
    if decompose_j2c != 'CD':
        low = _eig_decompose(mol, j2c, lindep)
    else:
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
            decompose_j2c = 'CD'
        except scipy.linalg.LinAlgError:
            low = _eig_decompose(mol, j2c, lindep)
            decompose_j2c = 'ED'
    j2c = None
    naoaux, naux = low.shape
    time1 = log.timer('Cholesky 2c2e', *time1)

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = int(ao_loc[mol.nbas])
    naoaux = int(ao_loc[-1] - nao)
    if aosym == 's1':
        nao_pair = nao * nao
        buflen = min(max(int(max_memory*.24e6/8/naoaux/comp), 1), nao_pair)
        shranges = _guess_shell_ranges(mol, buflen, 's1')
    else:
        nao_pair = nao * (nao+1) // 2
        buflen = min(max(int(max_memory*.24e6/8/naoaux/comp), 1), nao_pair)
        shranges = _guess_shell_ranges(mol, buflen, 's2ij')
    log.debug('erifile %.8g MB, IO buf size %.8g MB',
              naoaux*nao_pair*8/1e6, comp*buflen*naoaux*8/1e6)
    log.debug1('shranges = %s', shranges)
    # TODO: Libcint-3.14 and newer version support to compute int3c2e without
    # the opt for the 3rd index.
    #if '3c2e' in int3c:
    #    cintopt = gto.moleintor.make_cintopt(atm, mol._bas, env, int3c)
    #else:
    #    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    bufs1 = numpy.empty((comp*max([x[2] for x in shranges]),naoaux))
    bufs2 = numpy.empty_like(bufs1)

    def transform(b):
        if b.ndim == 3 and b.flags.f_contiguous:
            b = lib.transpose(b.T, axes=(0,2,1)).reshape(naoaux,-1)
        else:
            b = b.reshape((-1,naoaux)).T
        if decompose_j2c != 'CD':
            return lib.dot(low, b)

        if b.flags.c_contiguous:
            trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, b))
            return trsm(1.0, low, b.T, lower=True, trans_a = 1, side = 1,
                     overwrite_b=True).T
        else:
            return scipy.linalg.solve_triangular(low, b, lower=True,
                                             overwrite_b=True, check_finite=False)

    def process(sh_range):
        nonlocal bufs1, bufs2
        bufs2, bufs1 = bufs1, bufs2
        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        ints = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                       aosym, ao_loc, cintopt, out=bufs1)
        if comp == 1:
            dat = transform(ints)
        else:
            dat = [transform(x) for x in ints]
        return dat

    feri = _create_h5file(erifile, dataname)

    for istep, dat in enumerate(lib.map_with_prefetch(process, shranges)):
        sh_range = shranges[istep]
        label = '%s/%d'%(dataname,istep)
        if comp == 1:
            feri[label] = dat
        else:
            shape = (len(dat),) + dat[0].shape
            fdat = feri.create_dataset(label, shape, dat[0].dtype.char)
            for i, b in enumerate(dat):
                fdat[i] = b
        dat = None
        log.debug('int3c2e [%d/%d], AO [%d:%d], nrow = %d',
                  istep+1, len(shranges), *sh_range)
        time1 = log.timer('gen CD eri [%d/%d]' % (istep+1,len(shranges)), *time1)
    bufs1 = None
    bufs2 = None
    feri.flush()
    feri.close()
    return erifile


def general(mol, mo_coeffs, erifile, auxbasis='weigend+etb', dataname='eri_mo', tmpdir=None,
            int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
            max_memory=MAX_MEMORY, verbose=0, compact=True):
    ''' Transform ij of (ij|L) to MOs.
    '''
    assert (aosym in ('s1', 's2ij'))
    time0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)

    if tmpdir is None:
        tmpdir = lib.param.TMPDIR
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    cholesky_eri_b(mol, swapfile.name, auxbasis, dataname,
                   int3c, aosym, int2c, comp, max_memory, verbose=log)
    fswap = h5py.File(swapfile.name, 'r')
    time1 = log.timer('AO->MO eri transformation 1 pass', *time0)

    nao = mo_coeffs[0].shape[0]
    if aosym == 's1':
        nao_pair = nao * nao
        aosym_as_nr_e2 = 's1'
    else:
        nao_pair = nao * (nao+1) // 2
        aosym_as_nr_e2 = 's2kl'

    ijmosym, nij_pair, moij, ijshape = \
            ao2mo.incore._conc_mos(mo_coeffs[0], mo_coeffs[1],
                                   compact and aosym != 's1')

    naoaux = fswap['%s/0'%dataname].shape[-2]
    feri = _create_h5file(erifile, dataname)
    if comp == 1:
        h5d_eri = feri.create_dataset(dataname, (naoaux,nij_pair), 'f8')
    else:
        h5d_eri = feri.create_dataset(dataname, (comp,naoaux,nij_pair), 'f8')

    def load(row_slice):
        row0, row1 = row_slice
        return _load_from_h5g(fswap[dataname], row0, row1)

    iolen = min(max(int(max_memory*.45e6/8/(nao_pair+nij_pair)), 28), naoaux)
    totstep = (naoaux+iolen-1)//iolen
    ti0 = time1
    slices = list(lib.prange(0, naoaux, iolen))
    for istep, dat in enumerate(lib.map_with_prefetch(load, slices)):
        row0, row1 = slices[istep]
        nrow = row1 - row0
        if comp == 1:
            dat = _ao2mo.nr_e2(dat, moij, ijshape, aosym_as_nr_e2, ijmosym)
            h5d_eri[row0:row1] = dat
        else:
            dat = _ao2mo.nr_e2(dat.reshape(comp*nrow, nao_pair),
                               moij, ijshape, aosym_as_nr_e2, ijmosym)
            h5d_eri[:,row0:row1] = dat.reshape(comp, nrow, nij_pair)
        dat = None
        log.debug('step 2 [%d/%d], [%d:%d], row = %d',
                  istep+1, totstep, row0, row1, nrow)
        ti0 = log.timer('step 2 [%d/%d], [%d:%d], row = %d'%
                        (istep+1, totstep, row0, row1, nrow), *ti0)

    fswap.close()
    feri.close()
    log.timer('AO->MO CD eri transformation 2 pass', *time1)
    log.timer('AO->MO CD eri transformation', *time0)
    return erifile

def _guess_shell_ranges(mol, buflen, aosym, start=0, stop=None):
    from pyscf.ao2mo.outcore import balance_partition
    ao_loc_long = mol.ao_loc_nr().astype(numpy.int64)
    if 's2' in aosym:
        return balance_partition(ao_loc_long*(ao_loc_long+1)//2, buflen, start, stop)
    else:
        nao = ao_loc_long[-1]
        return balance_partition(ao_loc_long*nao, buflen, start, stop)

def _create_h5file(erifile, dataname):
    if isinstance(getattr(erifile, 'name', None), str):
        # The TemporaryFile and H5Tmpfile
        erifile = erifile.name

    if h5py.is_hdf5(erifile):
        feri = lib.H5FileWrap(erifile, 'a')
        if dataname in feri:
            del (feri[dataname])
    else:
        feri = lib.H5FileWrap(erifile, 'w')
    return feri

del (MAX_MEMORY)


if __name__ == '__main__':
    from pyscf.df import incore
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvtz'
    mol.build()

    cderi0 = incore.cholesky_eri(mol)
    cholesky_eri(mol, 'cderi.dat')
    with h5py.File('cderi.dat', 'r') as feri:
        print(numpy.allclose(feri['j3c'], cderi0))

    cholesky_eri(mol, 'cderi.dat', max_memory=.5)
    with h5py.File('cderi.dat', 'r') as feri:
        print(numpy.allclose(feri['j3c'], cderi0))

    general(mol, (numpy.eye(mol.nao_nr()),)*2, 'cderi.dat',
            max_memory=.2, verbose=6)
    with h5py.File('cderi.dat', 'r') as feri:
        print(numpy.allclose(feri['j3c'], cderi0))
