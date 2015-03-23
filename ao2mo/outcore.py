#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import os
import random
import time
import tempfile
import numpy
import h5py
import pyscf.lib
import pyscf.lib.logger as logger
from pyscf.ao2mo import _ao2mo

# default ioblk_size is 256 MB

IOBUF_WORDS_PREFER = 1e8
IOBUF_ROW_MIN = 160

def full(mol, mo_coeff, erifile, dataname='eri_mo', tmpdir=None,
         intor='cint2e_sph', aosym='s4', comp=1,
         max_memory=2000, ioblk_size=256, verbose=logger.WARN, compact=True):
    r'''Transfer arbitrary spherical AO integrals to MO integrals for given orbitals

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
        tmpdir : str
            The directory where to temporarily store the intermediate data
            (the half-transformed integrals).  By default, it's controlled by
            shell environment variable ``TMPDIR``.  The disk space requirement
            is about  comp*mo_coeffs[0].shape[1]*mo_coeffs[1].shape[1]*nao**2
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
            Components of the integrals, e.g. cint2e_ip_sph has 3 components.
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
        None

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     f5 = h5py.File(h5file)
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
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5', intor='cint2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5', intor='cint2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    general(mol, (mo_coeff,)*4, erifile, dataname, tmpdir,
            intor, aosym, comp, max_memory, ioblk_size, verbose, compact)
    return erifile

def general(mol, mo_coeffs, erifile, dataname='eri_mo', tmpdir=None,
            intor='cint2e_sph', aosym='s4', comp=1,
            max_memory=2000, ioblk_size=256, verbose=logger.WARN, compact=True):
    r'''For the given four sets of orbitals, transfer arbitrary spherical AO
    integrals to MO integrals on the fly.

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
        erifile : str
            To store the transformed integrals, in HDF5 format.

    Kwargs
        dataname : str
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        tmpdir : str
            The directory where to temporarily store the intermediate data
            (the half-transformed integrals).  By default, it's controlled by
            shell environment variable ``TMPDIR``.  The disk space requirement
            is about  comp*mo_coeffs[0].shape[1]*mo_coeffs[1].shape[1]*nao**2
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
            Components of the integrals, e.g. cint2e_ip_sph has 3 components.
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
        None

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     f5 = h5py.File(h5file)
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
    >>> ao2mo.outcore.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='cint2e_ip1_sph', aosym='s1', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)
    >>> ao2mo.outcore.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='cint2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    time_0pass = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]
    nao = mo_coeffs[0].shape[0]
    aosym = _stand_sym_code(aosym)
    if aosym in ('s4', 's2kl'):
        nao_pair = nao * (nao+1) // 2
    else:
        nao_pair = nao * nao

    if compact and ijsame and aosym in ('s4', 's2ij'):
        nij_pair = nmoi*(nmoi+1) // 2
    else:
        nij_pair = nmoi*nmoj

    if compact and klsame and aosym in ('s4', 's2kl'):
        log.debug('k-mo == l-mo')
        klmosym = 's2'
        nkl_pair = nmok*(nmok+1) // 2
        mokl = numpy.array(mo_coeffs[2], order='F', copy=False)
        klshape = (0, nmok, 0, nmok)
    else:
        klmosym = 's1'
        nkl_pair = nmok*nmol
        mokl = numpy.array(numpy.hstack((mo_coeffs[2],mo_coeffs[3])), \
                           order='F', copy=False)
        klshape = (0, nmok, nmok, nmol)

#    if nij_pair > nkl_pair:
#        log.warn('low efficiency for AO to MO trans!')

    if h5py.is_hdf5(erifile):
        feri = h5py.File(erifile)
        if dataname in feri:
            del(feri[dataname])
    else:
        feri = h5py.File(erifile, 'w')
    if comp == 1:
        chunks = (nmoj,nmol)
        h5d_eri = feri.create_dataset(dataname, (nij_pair,nkl_pair),
                                      'f8', chunks=chunks)
    else:
        chunks = (1,nmoj,nmol)
        h5d_eri = feri.create_dataset(dataname, (comp,nij_pair,nkl_pair),
                                      'f8', chunks=chunks)

    if nij_pair == 0 or nkl_pair == 0:
        feri.close()
        return erifile
    log.debug('num. MO ints = %.8g, require disk %.8g', \
              float(nij_pair)*nkl_pair*comp, nij_pair*nkl_pair*comp*8/1e6)

# transform e1
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    half_e1(mol, mo_coeffs, swapfile.name, intor, aosym, comp,
            max_memory, ioblk_size, log, compact)

    time_1pass = log.timer('AO->MO eri transformation 1 pass', *time_0pass)

    mem_words = max_memory * 1e6 / 8
    iobuf_size = min(float(nkl_pair)/(nkl_pair+nao_pair)*mem_words,
                     IOBUF_WORDS_PREFER) * 8
    iobuflen = guess_e2bufsize(ioblk_size, nij_pair, nao_pair)[0]

    log.debug('step2: kl-pair (ao %d, mo %d), mem %.8g MB, ioblock %.8g MB',
              nao_pair, nkl_pair, iobuflen*nao_pair*8/1e6,
              iobuflen*nkl_pair*8/1e6)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    ijmoblks = int(numpy.ceil(float(nij_pair)/iobuflen)) * comp
    ao_loc = numpy.array(mol.ao_loc_nr(), dtype=numpy.int32)
    ti0 = time_1pass
    buf = numpy.empty((iobuflen, nao_pair))
    istep = 0
    for row0, row1 in prange(0, nij_pair, iobuflen):
        nrow = row1 - row0

        for icomp in range(comp):
            istep += 1
            tioi = 0
            log.debug('step 2 [%d/%d], [%d,%d:%d], row = %d', \
                      istep, ijmoblks, icomp, row0, row1, nrow)

            col0 = 0
            for ic in range(klaoblks):
                dat = fswap['%d/%d'%(icomp,ic)]
                col1 = col0 + dat.shape[1]
                buf[:nrow,col0:col1] = dat[row0:row1]
                col0 = col1
            ti2 = log.timer('step 2 [%d/%d], load buf'%(istep,ijmoblks), *ti0)
            tioi += ti2[1]-ti0[1]
            pbuf = _ao2mo.nr_e2_(buf[:nrow], mokl, klshape, aosym, klmosym,
                                 ao_loc=ao_loc)

            tw1 = time.time()
            if comp == 1:
                h5d_eri[row0:row1] = pbuf
            else:
                h5d_eri[icomp,row0:row1] = pbuf
            tioi += time.time()-tw1

            ti1 = (time.clock(), time.time())
            log.debug('step 2 [%d/%d] CPU time: %9.2f, Wall time: %9.2f, I/O time: %9.2f', \
                      istep, ijmoblks, ti1[0]-ti0[0], ti1[1]-ti0[1], tioi)
            ti0 = ti1
    feri.close()
    fswap.close()

    log.timer('AO->MO eri transformation 2 pass', *time_1pass)
    log.timer('AO->MO eri transformation', *time_0pass)
    return erifile


# swapfile will be overwritten if exists.
def half_e1(mol, mo_coeffs, swapfile,
            intor='cint2e_sph', aosym='s4', comp=1,
            max_memory=2000, ioblk_size=256, verbose=logger.WARN, compact=True,
            ao2mopt=None):
    r'''Half transform arbitrary spherical AO integrals to MO integrals
    for the given two sets of orbitals

    Args:
        mol : :class:`Mole` object
            AO integrals will be generated in terms of mol._atm, mol._bas, mol._env
        mo_coeff : ndarray
            Transform (ij|kl) with the same set of orbitals.
        swapfile : str
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
            Components of the integrals, e.g. cint2e_ip_sph has 3 components.
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
    time0 = (time.clock(), time.time())
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nao = mo_coeffs[0].shape[0]
    aosym = _stand_sym_code(aosym)
    if aosym in ('s4', 's2ij'):
        nao_pair = nao * (nao+1) // 2
    else:
        nao_pair = nao * nao

    if compact and ijsame and aosym in ('s4', 's2ij'):
        log.debug('i-mo == j-mo')
        ijmosym = 's2'
        nij_pair = nmoi*(nmoi+1) // 2
        moij = numpy.array(mo_coeffs[0], order='F', copy=False)
        ijshape = (0, nmoi, 0, nmoi)
    else:
        ijmosym = 's1'
        nij_pair = nmoi*nmoj
        moij = numpy.array(numpy.hstack((mo_coeffs[0],mo_coeffs[1])), \
                           order='F', copy=False)
        ijshape = (0, nmoi, nmoi, nmoj)

    e1buflen, mem_words, iobuf_words, ioblk_words = \
            guess_e1bufsize(max_memory, ioblk_size, nij_pair, nao_pair, comp)
# The buffer to hold AO integrals in C code, see line (@)
    aobuflen = int((mem_words - iobuf_words) // (nao*nao*comp))
    shranges = guess_shell_ranges(mol, e1buflen, aobuflen, aosym)
    if ao2mopt is None:
        if intor == 'cint2e_sph':
            ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                     'CVHFsetnr_direct_scf')
        else:
            ao2mopt = _ao2mo.AO2MOpt(mol, intor)

    log.debug('step1: tmpfile %.8g MB', nij_pair*nao_pair*8/1e6)
    log.debug('step1: (ij,kl) = (%d,%d), mem cache %.8g MB, iobuf %.8g MB',
              nij_pair, nao_pair, mem_words*8/1e6, iobuf_words*8/1e6)

    fswap = h5py.File(swapfile, 'w')
    for icomp in range(comp):
        g = fswap.create_group(str(icomp)) # for h5py old version

    # transform e1
    ti0 = log.timer('Initializing ao2mo.outcore.half_e1', *time0)
    nstep = len(shranges)
    for istep,sh_range in enumerate(shranges):
        log.debug('step 1 [%d/%d], AO [%d:%d], len(buf) = %d', \
                  istep+1, nstep, *(sh_range[:3]))
        buflen = sh_range[2]
        iobuf = numpy.empty((comp,buflen,nij_pair))
        nmic = len(sh_range[3])
        p0 = 0
        for imic, aoshs in enumerate(sh_range[3]):
            log.debug1('      fill iobuf micro [%d/%d], AO [%d:%d], len(aobuf) = %d', \
                       imic+1, nmic, *aoshs)
            buf = numpy.empty((comp*aoshs[2],nao*nao)) # (@)
            _ao2mo.nr_e1fill_(intor, aoshs, mol._atm, mol._bas, mol._env,
                              aosym, comp, ao2mopt, buf)
            buf = _ao2mo.nr_e1_(buf, moij, ijshape, aosym, ijmosym)
            iobuf[:,p0:p0+aoshs[2]] = buf.reshape(comp,aoshs[2],-1)
            p0 += aoshs[2]
        ti2 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,nstep), *ti0)

        e2buflen, chunks = guess_e2bufsize(ioblk_size, nij_pair, buflen)
        for icomp in range(comp):
            dset = fswap.create_dataset('%d/%d'%(icomp,istep),
                                        (nij_pair,iobuf.shape[1]), 'f8',
                                        chunks=None)
            for col0, col1 in prange(0, nij_pair, e2buflen):
                dset[col0:col1] = pyscf.lib.transpose(iobuf[icomp,:,col0:col1])
        ti0 = log.timer('transposing to disk', *ti2)
    fswap.close()
    return swapfile

def full_iofree(mol, mo_coeff, intor='cint2e_sph', aosym='s4', comp=1,
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
        tmpdir : str
            The directory where to temporarily store the intermediate data
            (the half-transformed integrals).  By default, it's controlled by
            shell environment variable ``TMPDIR``.  The disk space requirement
            is about  comp*mo_coeffs[0].shape[1]*mo_coeffs[1].shape[1]*nao**2
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
            Components of the integrals, e.g. cint2e_ip_sph has 3 components.
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
    >>> eri1 = ao2mo.outcore.full_iofree(mol, mo1, intor='cint2e_ip1_sph', aosym='s1', comp=3)
    >>> print(eri1.shape)
    (3, 100, 100)
    >>> eri1 = ao2mo.outcore.full_iofree(mol, mo1, intor='cint2e_ip1_sph', aosym='s2kl', comp=3)
    >>> print(eri1.shape)
    (3, 100, 55)
    '''
    erifile = tempfile.NamedTemporaryFile()
    general(mol, (mo_coeff,)*4, erifile.name, dataname='eri_mo',
            intor=intor, aosym=aosym, comp=comp,
            verbose=verbose, compact=compact)
    with h5py.File(erifile.name, 'r') as feri:
        return numpy.array(feri['eri_mo'])

def general_iofree(mol, mo_coeffs, intor='cint2e_sph', aosym='s4', comp=1,
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
            Components of the integrals, e.g. cint2e_ip_sph has 3 components.
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
    ...     f5 = h5py.File(h5file)
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
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo1,mo1,mo1), intor='cint2e_ip1_sph', aosym='s1', comp=3)
    >>> print(eri1.shape)
    (3, 100, 100)
    >>> eri1 = ao2mo.outcore.general_iofree(mol, (mo1,mo1,mo1,mo1), intor='cint2e_ip1_sph', aosym='s2kl', comp=3)
    >>> print(eri1.shape)
    (3, 100, 55)
    '''
    erifile = tempfile.NamedTemporaryFile()
    general(mol, mo_coeffs, erifile.name, dataname='eri_mo',
            intor=intor, aosym=aosym, comp=comp,
            verbose=verbose, compact=compact)
    with h5py.File(erifile.name, 'r') as feri:
        return numpy.array(feri['eri_mo'])


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
    if mem_words > 2e8:
        iobuf_words = int(IOBUF_WORDS_PREFER) # 1.2GB
    else:
        iobuf_words = int(mem_words // 2)
    ioblk_words = int(min(ioblk_size*1e6/8, iobuf_words))

    e1buflen = int(min(iobuf_words//(comp*nij_pair), nao_pair))
    return e1buflen, mem_words, iobuf_words, ioblk_words

def guess_e2bufsize(ioblk_size, nrows, ncols):
    e2buflen = int(min(ioblk_size*1e6/8/ncols, nrows))
    e2buflen = max(e2buflen//IOBUF_ROW_MIN, 1) * IOBUF_ROW_MIN
    chunks = (IOBUF_ROW_MIN, ncols)
    return e2buflen, chunks

# based on the size of buffer, dynamic range of AO-shells for each buffer
def guess_shell_ranges(mol, max_iobuf, max_aobuf, aosym):
    max_iobuf = max(1, max_iobuf)
    max_aobuf = max(1, max_aobuf)
    ao_loc = mol.ao_loc_nr()

    accum = []

    if aosym in ('s4', 's2kl'):
        for i in range(mol.nbas):
            di = ao_loc[i+1] - ao_loc[i]
            for j in range(i):
                dj = ao_loc[j+1] - ao_loc[j]
                accum.append(di*dj)
            accum.append(di*(di+1)//2)
    else:
        for i in range(mol.nbas):
            di = ao_loc[i+1] - ao_loc[i]
            for j in range(mol.nbas):
                dj = ao_loc[j+1] - ao_loc[j]
                accum.append(di*dj)

    ijsh_range = []
    buflen = 0
    ij_start = 0
    for ij, dij in enumerate(accum):
        buflen += dij
        if buflen > max_iobuf and buflen > dij:
# to fill each iobuf, AO integrals may need to be fill to aobuf several times
            if max_aobuf < buflen-dij:
                ijdiv = []
                n0 = ij_start
                aobuf = 0
                for n in range(ij_start, ij):
                    aobuf += accum[n]
                    if aobuf > max_aobuf and aobuf > accum[n]:
                        ijdiv.append((n0, n, aobuf-accum[n]))
                        n0 = n
                        aobuf = accum[n]
                ijdiv.append((n0, ij, aobuf))
            else:
                ijdiv = [(ij_start, ij, buflen-dij)]

            ijsh_range.append((ij_start, ij, buflen-dij, ijdiv))

            ij_start = ij
            buflen = dij

    ij = len(accum)

    if max_aobuf < buflen:
        ijdiv = []
        n0 = ij_start
        aobuf = 0
        for n in range(ij_start, ij):
            aobuf += accum[n]
            if aobuf > max_aobuf:
                ijdiv.append((n0, n, aobuf-accum[n]))
                n0 = n
                aobuf = accum[n]
        ijdiv.append((n0, ij, aobuf))
    else:
        ijdiv = [(ij_start, ij, buflen)]
    ijsh_range.append((ij_start, ij, buflen, ijdiv))
    return ijsh_range

def _stand_sym_code(sym):
    if isinstance(sym, int):
        return 's%d' % sym
    elif 's' == sym[0] or 'a' == sym[0]:
        return sym
    else:
        return 's' + sym


if __name__ == '__main__':
    import time
    from pyscf import scf
    from pyscf import gto
    from pyscf.ao2mo import incore
    from pyscf.ao2mo import addons
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_outcore'
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

    print(time.clock())
    full(mol, rhf.mo_coeff, 'h2oeri.h5', max_memory=10, ioblk_size=5)
    print(time.clock())
    eri0 = incore.full(rhf._eri, rhf.mo_coeff)
    feri = h5py.File('h2oeri.h5', 'r')
    print('full', abs(eri0-feri['eri_mo']).sum())
    feri.close()

    print(time.clock())
    c = rhf.mo_coeff
    general(mol, (c,c,c,c), 'h2oeri.h5', max_memory=10, ioblk_size=5)
    print(time.clock())
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
