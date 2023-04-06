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


import numpy
import h5py
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo import incore
from pyscf import __config__

IOBLK_SIZE = getattr(__config__, 'ao2mo_outcore_ioblk_size', 256)  # 256 MB
IOBUF_WORDS = getattr(__config__, 'ao2mo_outcore_iobuf_words', 1e8)  # 800 MB
IOBUF_ROW_MIN = getattr(__config__, 'ao2mo_outcore_row_min', 160)
MAX_MEMORY = getattr(__config__, 'ao2mo_outcore_max_memory', 2000)  # 2GB


def full(mol, mo_coeff, erifile, dataname='eri_mo',
         intor='int2e', aosym='s4', comp=None,
         max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE, verbose=logger.WARN,
         compact=True):
    r'''Transfer arbitrary spherical AO integrals to MO integrals for given orbitals

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeff : ndarray
            Transform (ij|kl) with the same set of orbitals.
        erifile : str or h5py File or h5py Group object
            To store the transformed integrals, in HDF5 format.

    Kwargs:
        dataname : str
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        None

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     f5 = h5py.File(h5file, 'r')
    ...     print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    ...     f5.close()
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 55)
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    general(mol, (mo_coeff,)*4, erifile, dataname,
            intor, aosym, comp, max_memory, ioblk_size, verbose, compact)
    return erifile

def general(mol, mo_coeffs, erifile, dataname='eri_mo',
            intor='int2e', aosym='s4', comp=None,
            max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE, verbose=logger.WARN,
            compact=True):
    r'''For the given four sets of orbitals, transfer arbitrary spherical AO
    integrals to MO integrals on the fly.

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
        erifile : str or h5py File or h5py Group object
            To store the transformed integrals, in HDF5 format.

    Kwargs
        dataname : str
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        None

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     f5 = h5py.File(h5file, 'r')
    ...     print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    ...     f5.close()
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> mo2 = numpy.random.random((mol.nao_nr(), 8))
    >>> mo3 = numpy.random.random((mol.nao_nr(), 6))
    >>> mo4 = numpy.random.random((mol.nao_nr(), 4))
    >>> ao2mo.outcore.general(mol, (mo1,mo2,mo3,mo4), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 24)
    >>> ao2mo.outcore.general(mol, (mo1,mo2,mo3,mo3), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 21)
    >>> ao2mo.outcore.general(mol, (mo1,mo2,mo3,mo3), 'oh2.h5', compact=False)
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 36)
    >>> ao2mo.outcore.general(mol, (mo1,mo1,mo2,mo2), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (55, 36)
    >>> ao2mo.outcore.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', dataname='new')
    >>> view('oh2.h5', 'new')
    dataset ['eri_mo', 'new'], shape (55, 55)
    >>> ao2mo.outcore.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)
    >>> ao2mo.outcore.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if any(c.dtype == numpy.complex128 for c in mo_coeffs):
        raise NotImplementedError('Integral transformation for complex orbitals')

    time_0pass = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]

    intor, comp = gto.moleintor._get_intor_and_comp(mol._add_suffix(intor), comp)
    assert (nao == mol.nao_nr('_cart' in intor))

    aosym = _stand_sym_code(aosym)
    if aosym in ('s4', 's2kl'):
        nao_pair = nao * (nao+1) // 2
    else:
        nao_pair = nao * nao

    if (compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1]) and
        aosym in ('s4', 's2ij')):
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        nij_pair = nmoi*nmoj

    klmosym, nkl_pair, mokl, klshape = \
            incore._conc_mos(mo_coeffs[2], mo_coeffs[3],
                             compact and aosym in ('s4', 's2kl'))

#    if nij_pair > nkl_pair:
#        log.warn('low efficiency for AO to MO trans!')

    if isinstance(erifile, str):
        if h5py.is_hdf5(erifile):
            feri = h5py.File(erifile, 'a')
            if dataname in feri:
                del (feri[dataname])
        else:
            feri = h5py.File(erifile, 'w')
    else:
        assert (isinstance(erifile, h5py.Group))
        feri = erifile

    if comp == 1:
        chunks = (nmoj, nmol)
        shape = (nij_pair, nkl_pair)
    else:
        chunks = (1, nmoj, nmol)
        shape = (comp, nij_pair, nkl_pair)

    if nij_pair == 0 or nkl_pair == 0:
        feri.create_dataset(dataname, shape, 'f8')
        if isinstance(erifile, str):
            feri.close()
        return erifile
    else:
        h5d_eri = feri.create_dataset(dataname, shape, 'f8', chunks=chunks)

    log.debug('MO integrals %s are saved in %s/%s', intor, erifile, dataname)
    log.debug('num. MO ints = %.8g, required disk %.8g MB',
              float(nij_pair)*nkl_pair*comp, nij_pair*nkl_pair*comp*8/1e6)

# transform e1
    fswap = lib.H5TmpFile()
    half_e1(mol, mo_coeffs, fswap, intor, aosym, comp, max_memory, ioblk_size,
            log, compact)

    time_1pass = log.timer('AO->MO transformation for %s 1 pass'%intor,
                           *time_0pass)

    def load(icomp, row0, row1, buf):
        if icomp+1 < comp:
            icomp += 1
        else:  # move to next row-block
            row0, row1 = row1, min(nij_pair, row1+iobuflen)
            icomp = 0
        if row0 < row1:
            _load_from_h5g(fswap['%d'%icomp], row0, row1, buf)

    def save(icomp, row0, row1, buf):
        if comp == 1:
            h5d_eri[row0:row1] = buf[:row1-row0]
        else:
            h5d_eri[icomp,row0:row1] = buf[:row1-row0]

    ioblk_size = max(max_memory*.1, ioblk_size)
    iobuflen = guess_e2bufsize(ioblk_size, nij_pair, max(nao_pair,nkl_pair))[0]
    buf = numpy.empty((iobuflen,nao_pair))
    buf_prefetch = numpy.empty_like(buf)
    outbuf = numpy.empty((iobuflen,nkl_pair))
    buf_write = numpy.empty_like(outbuf)

    log.debug('step2: kl-pair (ao %d, mo %d), mem %.8g MB, ioblock %.8g MB',
              nao_pair, nkl_pair, iobuflen*nao_pair*8/1e6,
              iobuflen*nkl_pair*8/1e6)

    #klaoblks = len(fswap['0'])
    ijmoblks = int(numpy.ceil(float(nij_pair)/iobuflen)) * comp
    ao_loc = mol.ao_loc_nr('_cart' in intor)
    ti0 = time_1pass
    istep = 0
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as async_write:
            _load_from_h5g(fswap['0'], 0, min(nij_pair, iobuflen), buf_prefetch)

            for row0, row1 in prange(0, nij_pair, iobuflen):
                nrow = row1 - row0

                for icomp in range(comp):
                    istep += 1
                    log.debug1('step 2 [%d/%d], [%d,%d:%d], row = %d',
                               istep, ijmoblks, icomp, row0, row1, nrow)

                    buf, buf_prefetch = buf_prefetch, buf
                    prefetch(icomp, row0, row1, buf_prefetch)
                    _ao2mo.nr_e2(buf[:nrow], mokl, klshape, aosym, klmosym,
                                 ao_loc=ao_loc, out=outbuf)
                    async_write(icomp, row0, row1, outbuf)
                    outbuf, buf_write = buf_write, outbuf  # avoid flushing writing buffer

                    ti1 = (logger.process_clock(), logger.perf_counter())
                    log.debug1('step 2 [%d/%d] CPU time: %9.2f, Wall time: %9.2f',
                               istep, ijmoblks, ti1[0]-ti0[0], ti1[1]-ti0[1])
                    ti0 = ti1

    fswap = None
    if isinstance(erifile, str):
        feri.close()

    log.timer('AO->MO transformation for %s 2 pass'%intor, *time_1pass)
    log.timer('AO->MO transformation for %s '%intor, *time_0pass)
    return erifile


# swapfile will be overwritten if exists.
def half_e1(mol, mo_coeffs, swapfile,
            intor='int2e', aosym='s4', comp=1,
            max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE, verbose=logger.WARN,
            compact=True, ao2mopt=None):
    r'''Half transform arbitrary spherical AO integrals to MO integrals
    for the given two sets of orbitals

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeff : ndarray
            Transform (ij|kl) with the same set of orbitals.
        swapfile : str or h5py File or h5py Group object
            To store the transformed integrals, in HDF5 format.  The transformed
            integrals are saved in blocks.

    Kwargs
        intor : str
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        verbose : int
            Print level
        max_memory : float or int
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals
        ao2mopt : :class:`AO2MOpt` object
            Precomputed data to improve perfomance

    Returns:
        None

    '''
    if any(c.dtype == numpy.complex128 for c in mo_coeffs):
        raise NotImplementedError('Integral transformation for complex orbitals')

    intor = mol._add_suffix(intor)
    time0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)

    nao = mo_coeffs[0].shape[0]
    aosym = _stand_sym_code(aosym)
    if aosym in ('s4', 's2ij'):
        nao_pair = nao * (nao+1) // 2
    else:
        nao_pair = nao * nao

    ijmosym, nij_pair, moij, ijshape = \
            incore._conc_mos(mo_coeffs[0], mo_coeffs[1],
                             compact and aosym in ('s4', 's2ij'))

    e1buflen, mem_words, iobuf_words, ioblk_words = \
            guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp)
    ioblk_size = ioblk_words * 8/1e6
# The buffer to hold AO integrals in C code, see line (@)
    aobuflen = max(int((mem_words - 2*comp*e1buflen*nij_pair) // (nao_pair*comp)),
                   IOBUF_ROW_MIN)
    ao_loc = mol.ao_loc_nr('_cart' in intor)
    shranges = guess_shell_ranges(mol, (aosym in ('s4', 's2kl')), e1buflen,
                                  aobuflen, ao_loc)
    if ao2mopt is None:
        if intor == 'int2e_cart' or intor == 'int2e_sph':
            ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                     'CVHFsetnr_direct_scf')
        else:
            ao2mopt = _ao2mo.AO2MOpt(mol, intor)

    if isinstance(swapfile, h5py.Group):
        fswap = swapfile
    else:
        fswap = lib.H5TmpFile(swapfile)
    for icomp in range(comp):
        fswap.create_group(str(icomp)) # for h5py old version

    log.debug('step1: tmpfile %s  %.8g MB', fswap.filename, nij_pair*nao_pair*8/1e6)
    log.debug('step1: (ij,kl) = (%d,%d), mem cache %.8g MB, iobuf %.8g MB',
              nij_pair, nao_pair, mem_words*8/1e6, iobuf_words*8/1e6)
    nstep = len(shranges)
    e1buflen = max([x[2] for x in shranges])

    e2buflen, chunks = guess_e2bufsize(ioblk_size, nij_pair, e1buflen)
    def save(istep, iobuf):
        for icomp in range(comp):
            _transpose_to_h5g(fswap, '%d/%d'%(icomp,istep), iobuf[icomp],
                              e2buflen, None)

    # transform e1
    ti0 = log.timer('Initializing ao2mo.outcore.half_e1', *time0)
    with lib.call_in_background(save) as async_write:
        buf1 = numpy.empty((comp*e1buflen,nao_pair))
        buf2 = numpy.empty((comp*e1buflen,nij_pair))
        buf_write = numpy.empty_like(buf2)
        fill = _ao2mo.nr_e1fill
        f_e1 = _ao2mo.nr_e1
        for istep,sh_range in enumerate(shranges):
            log.debug1('step 1 [%d/%d], AO [%d:%d], len(buf) = %d',
                       istep+1, nstep, *(sh_range[:3]))
            buflen = sh_range[2]
            iobuf = numpy.ndarray((comp,buflen,nij_pair), buffer=buf2)
            nmic = len(sh_range[3])
            p1 = 0
            for imic, aoshs in enumerate(sh_range[3]):
                log.debug2('      fill iobuf micro [%d/%d], AO [%d:%d], len(aobuf) = %d',
                           imic+1, nmic, *aoshs)
                buf = fill(intor, aoshs, mol._atm, mol._bas, mol._env,
                           aosym, comp, ao2mopt, out=buf1).reshape(-1,nao_pair)
                buf = f_e1(buf, moij, ijshape, aosym, ijmosym)
                p0, p1 = p1, p1 + aoshs[2]
                iobuf[:,p0:p1] = buf.reshape(comp,aoshs[2],nij_pair)
            ti0 = log.timer_debug1('gen AO/transform MO [%d/%d]'%(istep+1,nstep), *ti0)

            async_write(istep, iobuf)
            buf2, buf_write = buf_write, buf2

    fswap = None
    return swapfile

def _load_from_h5g(h5group, row0, row1, out=None):
    nkeys = len(h5group)
    dat = h5group['0']
    ncol = sum(h5group[str(key)].shape[-1] for key in range(nkeys))
    if dat.ndim == 2:
        out = numpy.ndarray((row1-row0, ncol), dat.dtype, buffer=out)
        col1 = 0
        for key in range(nkeys):
            dat = h5group[str(key)][row0:row1]
            col0, col1 = col1, col1 + dat.shape[1]
            out[:,col0:col1] = dat
    else:  # multiple components
        out = numpy.ndarray((dat.shape[0], row1-row0, ncol), dat.dtype, buffer=out)
        col1 = 0
        for key in range(nkeys):
            dat = h5group[str(key)][:,row0:row1]
            col0, col1 = col1, col1 + dat.shape[2]
            out[:,:,col0:col1] = dat
    return out

def _transpose_to_h5g(h5group, key, dat, blksize, chunks=None):
    nrow, ncol = dat.shape
    dset = h5group.create_dataset(key, (ncol,nrow), 'f8', chunks=chunks)
    for col0, col1 in prange(0, ncol, blksize):
        dset[col0:col1] = lib.transpose(dat[:,col0:col1])

def full_iofree(mol, mo_coeff, intor='int2e', aosym='s4', comp=None,
                max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE,
                verbose=logger.WARN, compact=True):
    r'''Transfer arbitrary spherical AO integrals to MO integrals for given orbitals
    This function is a wrap for :func:`ao2mo.outcore.general`.  It's not really
    IO free.  The returned MO integrals are held in memory.  For backward compatibility,
    it is used to replace the non-existed function direct.full_iofree.

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeff : ndarray
            Transform (ij|kl) with the same set of orbitals.
        erifile : str
            To store the transformed integrals, in HDF5 format.

    Kwargs
        dataname : str
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        verbose : int
            Print level
        max_memory : float or int
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        2D/3D MO-integral array.  They may or may not have the permutation
        symmetry, depending on the given orbitals, and the kwargs compact.  If
        the four sets of orbitals are identical, the MO integrals will at most
        have 4-fold symmetry.

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> eri1 = ao2mo.outcore.full_iofree(mol, mo1)
    >>> print(eri1.shape)
    (55, 55)
    >>> eri1 = ao2mo.outcore.full_iofree(mol, mo1, compact=False)
    >>> print(eri1.shape)
    (100, 100)
    >>> eri1 = ao2mo.outcore.full_iofree(mol, mo1, intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> print(eri1.shape)
    (3, 100, 100)
    >>> eri1 = ao2mo.outcore.full_iofree(mol, mo1, intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> print(eri1.shape)
    (3, 100, 55)
    '''
    with lib.H5TmpFile() as feri:
        general(mol, (mo_coeff,)*4, feri, dataname='eri_mo',
                intor=intor, aosym=aosym, comp=comp,
                max_memory=max_memory, ioblk_size=ioblk_size,
                verbose=verbose, compact=compact)
        return numpy.asarray(feri['eri_mo'])

def general_iofree(mol, mo_coeffs, intor='int2e', aosym='s4', comp=None,
                   max_memory=MAX_MEMORY, ioblk_size=IOBLK_SIZE,
                   verbose=logger.WARN, compact=True):
    r'''For the given four sets of orbitals, transfer arbitrary spherical AO
    integrals to MO integrals on the fly.  This function is a wrap for
    :func:`ao2mo.outcore.general`.  It's not really IO free.  The returned MO
    integrals are held in memory.  For backward compatibility, it is used to
    replace the non-existed function direct.general_iofree.

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)

    Kwargs
        intor : str
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            Permutation symmetry for the AO integrals

            | 4 or '4' or 's4': 4-fold symmetry (default)
            | '2ij' or 's2ij' : symmetry between i, j in (ij|kl)
            | '2kl' or 's2kl' : symmetry between k, l in (ij|kl)
            | 1 or '1' or 's1': no symmetry
            | 'a4ij' : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a4kl' : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | 'a2ij' : anti-symmetry between i, j in (ij|kl) (TODO)
            | 'a2kl' : anti-symmetry between k, l in (ij|kl) (TODO)

        comp : int
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        2D/3D MO-integral array.  They may or may not have the permutation
        symmetry, depending on the given orbitals, and the kwargs compact.  If
        the four sets of orbitals are identical, the MO integrals will at most
        have 4-fold symmetry.

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     f5 = h5py.File(h5file, 'r')
    ...     print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    ...     f5.close()
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> mo2 = numpy.random.random((mol.nao_nr(), 8))
    >>> mo3 = numpy.random.random((mol.nao_nr(), 6))
    >>> mo4 = numpy.random.random((mol.nao_nr(), 4))
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo2,mo3,mo4))
    >>> print(eri1.shape)
    (80, 24)
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo2,mo3,mo3))
    >>> print(eri1.shape)
    (80, 21)
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo2,mo3,mo3), compact=False)
    >>> print(eri1.shape)
    (80, 36)
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo1,mo1,mo1), intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> print(eri1.shape)
    (3, 100, 100)
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo1,mo1,mo1), intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> print(eri1.shape)
    (3, 100, 55)
    '''
    with lib.H5TmpFile() as feri:
        general(mol, mo_coeffs, feri, dataname='eri_mo',
                intor=intor, aosym=aosym, comp=comp,
                max_memory=max_memory, ioblk_size=ioblk_size,
                verbose=verbose, compact=compact)
        return numpy.asarray(feri['eri_mo'])


def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) \
            or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp):
    mem_words = max(1, max_memory * 1e6 / 8)
# part of the max_memory is used to hold the AO integrals.  The iobuf is the
# buffer to temporary hold the transformed integrals before streaming to disk.
# iobuf is then divided to small blocks (ioblk_words) and streamed to disk.
    iobuf_words = max(int(mem_words//6), IOBUF_WORDS)
    ioblk_words = int(min(ioblk_size*1e6/8, iobuf_words))

    e1buflen = int(mem_words*.66/(comp*(nij_pair*2+nao_pair)))
    e1buflen = max(e1buflen, IOBUF_ROW_MIN)
    return e1buflen, mem_words, iobuf_words, ioblk_words

def guess_e2bufsize(ioblk_size, nrows, ncols):
    e2buflen = int(min(ioblk_size*1e6/8/ncols, nrows))
    e2buflen = max(e2buflen//IOBUF_ROW_MIN, 1) * IOBUF_ROW_MIN
    chunks = (IOBUF_ROW_MIN, ncols)
    return e2buflen, chunks

# based on the size of buffer, dynamic range of AO-shells for each buffer
def guess_shell_ranges(mol, aosym, max_iobuf, max_aobuf=None, ao_loc=None,
                       compress_diag=True):
    if ao_loc is None: ao_loc = mol.ao_loc_nr()
    max_iobuf = max(1, max_iobuf)

    dims = ao_loc[1:] - ao_loc[:-1]
    dijs = (dims.reshape(-1,1) * dims)
    nbas = dijs.shape[0]

    if aosym:
        if compress_diag:
            #:for i in range(mol.nbas):
            #:    di = ao_loc[i+1] - ao_loc[i]
            #:    for j in range(i):
            #:        dj = ao_loc[j+1] - ao_loc[j]
            #:        lstdij.append(di*dj)
            #:    lstdij.append(di*(di+1)//2)
            idx = numpy.arange(nbas)
            dijs[idx,idx] = dims*(dims+1)//2
            lstdij = dijs[numpy.tril_indices(nbas)]
        else:
            #:for i in range(mol.nbas):
            #:    di = ao_loc[i+1] - ao_loc[i]
            #:    for j in range(i+1):
            #:        dj = ao_loc[j+1] - ao_loc[j]
            #:        lstdij.append(di*dj)
            lstdij = dijs[numpy.tril_indices(nbas)]
    else:
        #:for i in range(mol.nbas):
        #:    di = ao_loc[i+1] - ao_loc[i]
        #:    for j in range(mol.nbas):
        #:        dj = ao_loc[j+1] - ao_loc[j]
        #:        lstdij.append(di*dj)
        lstdij = dijs.ravel()

    dij_loc = numpy.append(0, numpy.cumsum(lstdij))
    ijsh_range = balance_partition(dij_loc, max_iobuf)

    if max_aobuf is not None:
        max_aobuf = max(1, max_aobuf)
        def div_each_iobuf(ijstart, ijstop, buflen):
            # to fill each iobuf, AO integrals may need to be fill to aobuf several times
            return (ijstart, ijstop, buflen,
                    balance_partition(dij_loc, max_aobuf, ijstart, ijstop))
        ijsh_range = [div_each_iobuf(*x) for x in ijsh_range]
    return ijsh_range

def _stand_sym_code(sym):
    if isinstance(sym, int):
        return 's%d' % sym
    elif 's' == sym[0] or 'a' == sym[0]:
        return sym
    else:
        return 's' + sym

def balance_segs(segs_lst, blksize, start_id=0, stop_id=None):
    loc = numpy.append(0, numpy.cumsum(segs_lst))
    return balance_partition(loc, blksize, start_id, stop_id)
def balance_partition(ao_loc, blksize, start_id=0, stop_id=None):
    if stop_id is None:
        stop_id = len(ao_loc) - 1
    else:
        stop_id = min(stop_id, start_id+len(ao_loc)-1)
    displs = lib.misc._blocksize_partition(ao_loc[start_id:stop_id+1], blksize)
    displs = [i+start_id for i in displs]
    tasks = []
    for i0, i1 in zip(displs[:-1],displs[1:]):
        tasks.append((i0, i1, ao_loc[i1]-ao_loc[i0]))
    return tasks

del (MAX_MEMORY)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf.ao2mo import addons
    mol = gto.Mole()
    mol.verbose = 5
    #mol.output = 'out_outcore'
    mol.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvtz',
                 'O': 'cc-pvtz',}
    mol.build()
    nao = mol.nao_nr()
    npair = nao*(nao+1)//2

    rhf = scf.RHF(mol)
    rhf.scf()

    print(logger.process_clock())
    full(mol, rhf.mo_coeff, 'h2oeri.h5', max_memory=10, ioblk_size=5)
    print(logger.process_clock())
    eri0 = incore.full(rhf._eri, rhf.mo_coeff)
    feri = h5py.File('h2oeri.h5', 'r')
    print('full', abs(eri0-feri['eri_mo']).sum())
    feri.close()

    print(logger.process_clock())
    c = rhf.mo_coeff
    general(mol, (c,c,c,c), 'h2oeri.h5', max_memory=10, ioblk_size=5)
    print(logger.process_clock())
    feri = h5py.File('h2oeri.h5', 'r')
    print('general', abs(eri0-feri['eri_mo']).sum())
    feri.close()

    # set ijsame and klsame to False, then check
    c = rhf.mo_coeff
    n = c.shape[1]
    general(mol, (c,c,c,c), 'h2oeri.h5', max_memory=10, ioblk_size=5, compact=False)
    feri = h5py.File('h2oeri.h5', 'r')
    eri1 = numpy.array(feri['eri_mo']).reshape(n,n,n,n)
    eri1 = addons.restore(4, eri1, n)
    print('general', abs(eri0-eri1).sum())
