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

# default max_memory (MB) is 1000 MB, large cache may NOT improve performance
# default ioblk_size is 256 MB

def full(mol, mo_coeff, erifile, dataname='eri_mo', tmpdir=None,
         intor='cint2e_sph', aosym='s4', comp=1,
         max_memory=1000, ioblk_size=256, verbose=0, compact=True):
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
        aosym : str
            Permutation symmetry for the AO integrals

            | s4 : 4-fold symmetry (default)
            | s2ij : symmetry between i, j in (ij|kl)
            | s2kl : symmetry between k, l in (ij|kl)
            | s1 : no symmetry
            | a4ij : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | a4kl : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | s2ij : anti-symmetry between i, j in (ij|kl) (TODO)
            | s2kl : anti-symmetry between k, l in (ij|kl) (TODO)

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
    >>> mol = gto.Mole()
    >>> mol.build(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 24)
    >>> ao2mo.outcore.full(mol, (mo1,mo2,mo3,mo3), 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)
    >>> ao2mo.outcore.full(mol, (mo1,mo1,mo1,mo1), 'full.h5', intor='cint2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)
    >>> ao2mo.outcore.full(mol, (mo1,mo1,mo1,mo1), 'full.h5', intor='cint2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    general(mol, (mo_coeff,)*4, erifile, dataname, tmpdir,
            intor, aosym, comp, max_memory, ioblk_size, verbose, compact)
    return erifile

def general(mol, mo_coeffs, erifile, dataname='eri_mo', tmpdir=None,
            intor='cint2e_sph', aosym='s4', comp=1,
            max_memory=1000, ioblk_size=256, verbose=0, compact=True):
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
        aosym : str
            Permutation symmetry for the AO integrals

            | s4 : 4-fold symmetry (default)
            | s2ij : symmetry between i, j in (ij|kl)
            | s2kl : symmetry between k, l in (ij|kl)
            | s1 : no symmetry
            | a4ij : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | a4kl : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | s2ij : anti-symmetry between i, j in (ij|kl) (TODO)
            | s2kl : anti-symmetry between k, l in (ij|kl) (TODO)

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
    >>> mol = gto.Mole()
    >>> mol.build(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
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
    elif isinstance(verbose, int):
        log = logger.Logger(mol.stdout, verbose)
    else:
        log = logger.Logger(mol.stdout, mol.verbose)

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])
    klsame = compact and iden_coeffs(mo_coeffs[2], mo_coeffs[3])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]
    nmok = mo_coeffs[2].shape[1]
    nmol = mo_coeffs[3].shape[1]

    nao = mo_coeffs[0].shape[0]
    if aosym in ('s4', 's2kl'):
        nao_pair = nao*(nao+1)//2
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
        h5d_eri = feri.create_dataset(dataname, (nij_pair,nkl_pair), 'f8')
    else:
        h5d_eri = feri.create_dataset(dataname, (comp,nij_pair,nkl_pair), 'f8')

    if nij_pair == 0 or nkl_pair == 0:
        feri.close()
        return None
    log.debug('num. MO ints = %.8g, require disk %.8g', \
              float(nij_pair)*nkl_pair*comp, nij_pair*nkl_pair*comp*8/1e6)

# transform e1
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    half_e1(mol, mo_coeffs, swapfile.name, intor, aosym, comp,
            max_memory, ioblk_size, log, compact)

    time_1pass = log.timer('AO->MO eri transformation 1 pass', *time_0pass)

    e2buflen = min(int(max_memory*1e6/8)//nao_pair,
                   int(ioblk_size*1e6/8)//nkl_pair, nij_pair)
    log.debug('step2: kl-pair (ao %d, mo %d), mem cache %.8g MB, ioblock %.8g MB', \
              nao_pair, nkl_pair,
              e2buflen*nao_pair*8/1e6, e2buflen*nkl_pair*8/1e6)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    ijmoblks = int(numpy.ceil(float(nij_pair)/e2buflen)) * comp
    ti0 = time_1pass
    buf = numpy.empty((e2buflen, nao_pair))
    istep = 0
    for row0, row1 in prange(0, nij_pair, e2buflen):
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
            pbuf = _ao2mo.nr_e2_(buf[:nrow], mokl, klshape, aosym, klmosym)

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
            max_memory=1000, ioblk_size=256, verbose=0, compact=True,
            ao2mopt=None):
    r'''Half transfer arbitrary spherical AO integrals to MO integrals
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
        aosym : str
            Permutation symmetry for the AO integrals

            | s4 : 4-fold symmetry (default)
            | s2ij : symmetry between i, j in (ij|kl)
            | s2kl : symmetry between k, l in (ij|kl)
            | s1 : no symmetry
            | a4ij : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | a4kl : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | s2ij : anti-symmetry between i, j in (ij|kl) (TODO)
            | s2kl : anti-symmetry between k, l in (ij|kl) (TODO)

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
    elif isinstance(verbose, int):
        log = logger.Logger(mol.stdout, verbose)
    else:
        log = logger.Logger(mol.stdout, mol.verbose)

    ijsame = compact and iden_coeffs(mo_coeffs[0], mo_coeffs[1])

    nmoi = mo_coeffs[0].shape[1]
    nmoj = mo_coeffs[1].shape[1]

    nao = mo_coeffs[0].shape[0]
    if aosym in ('s4', 's2kl'):
        nao_pair = nao*(nao+1)//2
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

    e1buflen, e2buflen = \
            info_swap_block(max_memory, ioblk_size, nij_pair, nao_pair, comp)
    shranges = info_shell_ranges(mol, e1buflen, aosym)
    e1buflen = max([x[2] for x in shranges])
    if ao2mopt is None:
        if intor == 'cint2e_sph':
            ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                     'CVHFsetnr_direct_scf')
        else:
            ao2mopt = _ao2mo.AO2MOpt(mol, intor)

    log.debug('tmpfile %.8g MB', nij_pair*nao_pair*8/1e6)
    log.debug1('shranges = %s', shranges)
    log.debug('step1: (ij,kl) shape (%d,%d), swap-block-shape (%d,%d), mem cache %.8g MB', \
              nij_pair, nao_pair, e2buflen, e1buflen,
              comp*e1buflen*nij_pair*8/1e6)

    fswap = h5py.File(swapfile, 'w')
    for icomp in range(comp):
        fswap.create_group(str(icomp)) # for h5py old version

    # transform e1
    ti0 = log.timer('Initializing ao2mo.outcore.half_e1', *time0)
    for istep,sh_range in enumerate(shranges):
        log.debug('step 1 [%d/%d], AO [%d:%d], len(buf) = %d', \
                  istep+1, len(shranges), *sh_range)
        buf = _ao2mo.nr_e1_(intor, moij, ijshape, sh_range[:2],
                            mol._atm, mol._bas, mol._env,
                            aosym, ijmosym, comp, ao2mopt)
        ti2 = log.timer('gen AO/transform MO [%d/%d]'%(istep+1,len(shranges)),
                        *ti0)
        for icomp in range(comp):
            dset = fswap.create_dataset('%d/%d'%(icomp,istep),
                                        (nij_pair,buf.shape[1]), 'f8')
            for col0, col1 in prange(0, nij_pair, e2buflen):
                dset[col0:col1] = pyscf.lib.transpose(buf[icomp,:,col0:col1])
        ti0 = log.timer('transposing to disk', *ti2)
        # release the memory of buf before allocating temporary data
        buf = None
    return swapfile

def full_iofree(mol, mo_coeff, intor='cint2e_sph', aosym='s4', comp=1,
                verbose=0, compact=True):
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
        aosym : str
            Permutation symmetry for the AO integrals

            | s4 : 4-fold symmetry (default)
            | s2ij : symmetry between i, j in (ij|kl)
            | s2kl : symmetry between k, l in (ij|kl)
            | s1 : no symmetry
            | a4ij : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | a4kl : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | s2ij : anti-symmetry between i, j in (ij|kl) (TODO)
            | s2kl : anti-symmetry between k, l in (ij|kl) (TODO)

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
    >>> mol = gto.Mole()
    >>> mol.build(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> ao2mo.outcore.full(mol, mo1, 'full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 24)
    >>> ao2mo.outcore.full(mol, (mo1,mo2,mo3,mo3), 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)
    >>> ao2mo.outcore.full(mol, (mo1,mo1,mo1,mo1), 'full.h5', intor='cint2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)
    >>> ao2mo.outcore.full(mol, (mo1,mo1,mo1,mo1), 'full.h5', intor='cint2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    erifile = tempfile.NamedTemporaryFile()
    general(mol, (mo_coeff,)*4, erifile.name, dataname='eri_mo',
            intor=intor, aosym=aosym, comp=comp,
            verbose=verbose, compact=compact)
    feri = h5py.File(erifile.name, 'r')
    return numpy.array(feri['eri_mo'])

def general_iofree(mol, mo_coeffs, intor='cint2e_sph', aosym='s4', comp=1,
                   verbose=0, compact=True):
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
        aosym : str
            Permutation symmetry for the AO integrals

            | s4 : 4-fold symmetry (default)
            | s2ij : symmetry between i, j in (ij|kl)
            | s2kl : symmetry between k, l in (ij|kl)
            | s1 : no symmetry
            | a4ij : 4-fold symmetry with anti-symmetry between i, j in (ij|kl) (TODO)
            | a4kl : 4-fold symmetry with anti-symmetry between k, l in (ij|kl) (TODO)
            | s2ij : anti-symmetry between i, j in (ij|kl) (TODO)
            | s2kl : anti-symmetry between k, l in (ij|kl) (TODO)

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
        ndarray of transformed MO integrals.  The MO integrals may or may not
        have the permutation symmetry, depending on the given orbitals, and
        the kwargs compact.

    Examples:

    >>> from pyscf import gto
    >>> from pyscf import ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     f5 = h5py.File(h5file)
    ...     print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    ...     f5.close()
    >>> mol = gto.Mole()
    >>> mol.build(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
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
    feri = h5py.File(erifile.name, 'r')
    return numpy.array(feri['eri_mo'])


def iden_coeffs(mo1, mo2):
    return (id(mo1) == id(mo2)) \
            or (mo1.shape==mo2.shape and numpy.allclose(mo1,mo2))

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

# decide the number of blocks needed for swap file
def info_swap_block(max_memory, ioblk_size, nij_pair, nao_pair, comp):
    mem_words = int(max_memory * 1e6 / 8)
    ioblk_words = int(ioblk_size * 1e6 / 8)
    ioblk_words = min(ioblk_words, mem_words)

    # decided the buffer row and column sizes
    e1buflen = min(mem_words//(comp*nij_pair), nao_pair)
    e2buflen = min(mem_words//nao_pair, nij_pair)
    if e1buflen*e2buflen > 1.1*ioblk_words:
        e2buflen = int(ioblk_words//e1buflen + 1)
    return e1buflen, e2buflen

# based on the size of buffer, dynamic range of AO-shells for each buffer
def info_shell_ranges(mol, buflen, aosym):
    bas_dim = [(mol.bas_angular(i)*2+1)*(mol.bas_nctr(i)) \
               for i in range(mol.nbas)]
    ao_loc = [0]
    for i in bas_dim:
        ao_loc.append(ao_loc[-1]+i)
    nao = ao_loc[-1]

    ish_seg = [0] # record the starting shell id of each buffer
    bufrows = []
    ij_start = 0
    if aosym in ('s4', 's2kl'):
        for i in range(mol.nbas):
            ij_end = ao_loc[i+1]*(ao_loc[i+1]+1)//2
            if ij_end - ij_start > buflen:
                ish_seg.append(i) # put present shell to next segments
                ijend = ao_loc[i]*(ao_loc[i]+1)//2
                bufrows.append(ijend-ij_start)
                ij_start = ijend
        nao_pair = nao*(nao+1) // 2
        ish_seg.append(mol.nbas)
        bufrows.append(nao_pair-ij_start)
    else:
        for i in range(mol.nbas):
            ij_end = ao_loc[i+1] * nao
            if ij_end - ij_start > buflen:
                ish_seg.append(i) # put present shell to next segments
                ijend = ao_loc[i] * nao
                bufrows.append(ijend-ij_start)
                ij_start = ijend
        nao_pair = nao * nao
        ish_seg.append(mol.nbas)
        bufrows.append(nao_pair-ij_start)
    assert(sum(bufrows) == nao_pair)

    # for each buffer, sh_ranges record (start, end, bufrow)
    sh_ranges = list(zip(ish_seg[:-1], ish_seg[1:], bufrows))
    return sh_ranges


if __name__ == '__main__':
    import scf
    import gto
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

    import time
    print(time.clock())
    full(mol, rhf.mo_coeff, 'h2oeri.h5', max_memory=10, ioblk_size=5)
    print(time.clock())
    import incore
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
    import addons
    eri1 = addons.restore(4, eri1, n)
    print('general', abs(eri0-eri1).sum())
