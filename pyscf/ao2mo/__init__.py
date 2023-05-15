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

'''
General Integral transformation module
======================================

Simple usage::

    >>> from pyscf import gto, scf, ao2mo
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mo_ints = ao2mo.kernel(mol, mf.mo_coeff)
'''

import tempfile
import numpy
import h5py
from pyscf.ao2mo import incore
from pyscf.ao2mo import outcore
from pyscf.ao2mo import r_outcore
from pyscf.ao2mo.addons import load, restore

def full(eri_or_mol, mo_coeff, erifile=None, dataname='eri_mo', intor='int2e',
         *args, **kwargs):
    r'''MO integral transformation. The four indices (ij|kl) are transformed
    with the same set of orbitals.

    Args:
        eri_or_mol : ndarray or Mole object
            If AO integrals are given as ndarray, it can be either 8-fold or
            4-fold symmetry.  The integral transformation are computed incore
            (ie all intermediate are held in memory).
            If Mole object is given, AO integrals are generated on the fly and
            outcore algorithm is used (ie intermediate data are held on disk).
        mo_coeff : ndarray
            Orbital coefficients in 2D array
        erifile : str or h5py File or h5py Group object
            *Note* this argument is effective when eri_or_mol is Mole object.
            The file to store the transformed integrals.  If not given, the
            transformed integrals are held in memory.

    Kwargs:
        dataname : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            *Note* this argument is effective when eri_or_mol is Mole object.
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
            *Note* this argument is effective when eri_or_mol is Mole object.
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        If eri_or_mol is array or erifile is not give,  the function returns 2D
        array (or 3D array if comp > 1) of transformed MO integrals.  The MO
        integrals may or may not have the permutation symmetry (controlled by
        the kwargs compact).
        Otherwise, return the file/fileobject where the MO integrals are saved.


    Examples:

    >>> from pyscf import gto, ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     with h5py.File(h5file, 'r') as f5:
    ...         print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))

    >>> eri1 = ao2mo.full(mol, mo1)
    >>> print(eri1.shape)
    (55, 55)

    >>> eri = mol.intor('int2e_sph', aosym='s8')
    >>> eri1 = ao2mo.full(eri, mo1, compact=False)
    >>> print(eri1.shape)
    (100, 100)

    >>> ao2mo.full(mol, mo1, 'full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 55)

    >>> ao2mo.full(mol, mo1, 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)

    >>> ao2mo.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.full(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if isinstance(eri_or_mol, numpy.ndarray):
        return incore.full(eri_or_mol, mo_coeff, *args, **kwargs)
    else:
        if '_spinor' in intor:
            mod = r_outcore
        else:
            mod = outcore

        if isinstance(erifile, (str, h5py.Group)): # args[0] is erifile
            return mod.full(eri_or_mol, mo_coeff, erifile, dataname, intor,
                            *args, **kwargs)
        elif isinstance(erifile, tempfile._TemporaryFileWrapper):
            return mod.full(eri_or_mol, mo_coeff, erifile.name, dataname, intor,
                            *args, **kwargs)
        else:
            return mod.full_iofree(eri_or_mol, mo_coeff, intor, *args, **kwargs)

def general(eri_or_mol, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
            *args, **kwargs):
    r'''Given four sets of orbitals corresponding to the four MO indices,
    transfer arbitrary spherical AO integrals to MO integrals.

    Args:
        eri_or_mol : ndarray or Mole object
            If AO integrals are given as ndarray, it can be either 8-fold or
            4-fold symmetry.  The integral transformation are computed incore
            (ie all intermediate are held in memory).
            If Mole object is given, AO integrals are generated on the fly and
            outcore algorithm is used (ie intermediate data are held on disk).
        mo_coeffs : 4-item list of ndarray
            Four sets of orbital coefficients, corresponding to the four
            indices of (ij|kl)
        erifile : str or h5py File or h5py Group object
            *Note* this argument is effective when eri_or_mol is Mole object.
            The file to store the transformed integrals.  If not given, the
            transformed integrals are held in memory.

    Kwargs:
        dataname : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            *Note* this argument is effective when eri_or_mol is Mole object.
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
            *Note* this argument is effective when eri_or_mol is Mole object.
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        If eri_or_mol is array or erifile is not give,  the function returns 2D
        array (or 3D array, if comp > 1) of transformed MO integrals.  The MO
        integrals may at most have 4-fold symmetry (if the four sets of orbitals
        are identical) or may not have the permutation symmetry (controlled by
        the kwargs compact).
        Otherwise, return the file/fileobject where the MO integrals are saved.


    Examples:

    >>> from pyscf import gto, ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     with h5py.File(h5file, 'r') as f5:
    ...         print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> mo2 = numpy.random.random((mol.nao_nr(), 8))
    >>> mo3 = numpy.random.random((mol.nao_nr(), 6))
    >>> mo4 = numpy.random.random((mol.nao_nr(), 4))

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo3,mo4))
    >>> print(eri1.shape)
    (80, 24)

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo3,mo3))
    >>> print(eri1.shape)
    (80, 21)

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo3,mo3), compact=False)
    >>> print(eri1.shape)
    (80, 36)

    >>> eri1 = ao2mo.general(eri, (mo1,mo1,mo2,mo2))
    >>> print(eri1.shape)
    (55, 36)

    >>> eri1 = ao2mo.general(eri, (mo1,mo2,mo1,mo2))
    >>> print(eri1.shape)
    (80, 80)

    >>> ao2mo.general(mol, (mo1,mo2,mo3,mo4), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 24)

    >>> ao2mo.general(mol, (mo1,mo2,mo3,mo3), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 21)

    >>> ao2mo.general(mol, (mo1,mo2,mo3,mo3), 'oh2.h5', compact=False)
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 36)

    >>> ao2mo.general(mol, (mo1,mo1,mo2,mo2), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (55, 36)

    >>> ao2mo.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', dataname='new')
    >>> view('oh2.h5', 'new')
    dataset ['eri_mo', 'new'], shape (55, 55)

    >>> ao2mo.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.general(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if isinstance(eri_or_mol, numpy.ndarray):
        return incore.general(eri_or_mol, mo_coeffs, *args, **kwargs)
    else:
        if '_spinor' in intor:
            mod = r_outcore
        else:
            mod = outcore

        if isinstance(erifile, (str, h5py.Group)): # args[0] is erifile
            return mod.general(eri_or_mol, mo_coeffs, erifile, dataname, intor,
                               *args, **kwargs)
        elif isinstance(erifile, tempfile._TemporaryFileWrapper):
            return mod.general(eri_or_mol, mo_coeffs, erifile.name, dataname, intor,
                               *args, **kwargs)
        else:
            return mod.general_iofree(eri_or_mol, mo_coeffs, intor, *args, **kwargs)

def kernel(eri_or_mol, mo_coeffs, erifile=None, dataname='eri_mo', intor='int2e',
           *args, **kwargs):
    r'''Transfer arbitrary spherical AO integrals to MO integrals, for given
    orbitals or four sets of orbitals.  See also :func:`ao2mo.full` and :func:`ao2mo.general`.

    Args:
        eri_or_mol : ndarray or Mole object
            If AO integrals are given as ndarray, it can be either 8-fold or
            4-fold symmetry.  The integral transformation are computed incore
            (ie all intermediate are held in memory).
            If Mole object is given, AO integrals are generated on the fly and
            outcore algorithm is used (ie intermediate data are held on disk).
        mo_coeffs : an np array or a list of arrays
            A matrix of orbital coefficients if it is a numpy ndarray; Or four
            sets of orbital coefficients if it is a list of arrays,
            corresponding to the four indices of (ij|kl).

    Kwargs:
        erifile : str or h5py File or h5py Group object
            *Note* this argument is effective when eri_or_mol is Mole object.
            The file to store the transformed integrals.  If not given, the
            return value is an array (in memory) of the transformed integrals.
        dataname : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            The dataset name in the erifile (ref the hierarchy of HDF5 format
            http://www.hdfgroup.org/HDF5/doc1.6/UG/09_Groups.html).  By assigning
            different dataname, the existed integral file can be reused.  If
            the erifile contains the dataname, the new integrals data will
            overwrite the old one.
        intor : str
            *Note* this argument is effective when eri_or_mol is Mole object.
            Name of the 2-electron integral.  Ref to :func:`getints_by_shell`
            for the complete list of available 2-electron integral names
        aosym : int or str
            *Note* this argument is effective when eri_or_mol is Mole object.
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
            *Note* this argument is effective when eri_or_mol is Mole object.
            Components of the integrals, e.g. int2e_ip_sph has 3 components.
        max_memory : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The maximum size of cache to use (in MB), large cache may **not**
            improve performance.
        ioblk_size : float or int
            *Note* this argument is effective when eri_or_mol is Mole object.
            The block size for IO, large block size may **not** improve performance
        verbose : int
            Print level
        compact : bool
            When compact is True, depending on the four oribital sets, the
            returned MO integrals has (up to 4-fold) permutation symmetry.
            If it's False, the function will abandon any permutation symmetry,
            and return the "plain" MO integrals

    Returns:
        If eri_or_mol is array or erifile is not give,  the function returns 2D
        array (or 3D array, if comp > 1) of transformed MO integrals.  The MO
        integrals may at most have 4-fold symmetry (if the four sets of orbitals
        are identical) or may not have the permutation symmetry (controlled by
        the kwargs compact).
        Otherwise, return the file/fileobject where the MO integrals are saved.


    Examples:

    >>> from pyscf import gto, ao2mo
    >>> import h5py
    >>> def view(h5file, dataname='eri_mo'):
    ...     with h5py.File(h5file) as f5:
    ...         print('dataset %s, shape %s' % (str(f5.keys()), str(f5[dataname].shape)))
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
    >>> mo1 = numpy.random.random((mol.nao_nr(), 10))
    >>> mo2 = numpy.random.random((mol.nao_nr(), 8))
    >>> mo3 = numpy.random.random((mol.nao_nr(), 6))
    >>> mo4 = numpy.random.random((mol.nao_nr(), 4))

    >>> eri1 = ao2mo.kernel(mol, mo1)
    >>> print(eri1.shape)
    (55, 55)

    >>> eri = mol.intor('int2e_sph', aosym='s8')
    >>> eri1 = ao2mo.kernel(eri, mo1, compact=False)
    >>> print(eri1.shape)
    (100, 100)

    >>> ao2mo.kernel(mol, mo1, erifile='full.h5')
    >>> view('full.h5')
    dataset ['eri_mo'], shape (55, 55)

    >>> ao2mo.kernel(mol, mo1, 'full.h5', dataname='new', compact=False)
    >>> view('full.h5', 'new')
    dataset ['eri_mo', 'new'], shape (100, 100)

    >>> ao2mo.kernel(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.kernel(mol, mo1, 'full.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('full.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo3,mo4))
    >>> print(eri1.shape)
    (80, 24)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo3,mo3))
    >>> print(eri1.shape)
    (80, 21)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo3,mo3), compact=False)
    >>> print(eri1.shape)
    (80, 36)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo1,mo2,mo2))
    >>> print(eri1.shape)
    (55, 36)

    >>> eri1 = ao2mo.kernel(eri, (mo1,mo2,mo1,mo2))
    >>> print(eri1.shape)
    (80, 80)

    >>> ao2mo.kernel(mol, (mo1,mo2,mo3,mo4), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 24)

    >>> ao2mo.kernel(mol, (mo1,mo2,mo3,mo3), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 21)

    >>> ao2mo.kernel(mol, (mo1,mo2,mo3,mo3), 'oh2.h5', compact=False)
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (80, 36)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo2,mo2), 'oh2.h5')
    >>> view('oh2.h5')
    dataset ['eri_mo'], shape (55, 36)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', dataname='new')
    >>> view('oh2.h5', 'new')
    dataset ['eri_mo', 'new'], shape (55, 55)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s1', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 100)

    >>> ao2mo.kernel(mol, (mo1,mo1,mo1,mo1), 'oh2.h5', intor='int2e_ip1_sph', aosym='s2kl', comp=3)
    >>> view('oh2.h5')
    dataset ['eri_mo', 'new'], shape (3, 100, 55)
    '''
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        return full(eri_or_mol, mo_coeffs, erifile, dataname, intor, *args, **kwargs)
    else:
        return general(eri_or_mol, mo_coeffs, erifile, dataname, intor, *args, **kwargs)

def get_ao_eri(mol):
    '''2-electron integrals in AO basis'''
    return mol.intor('int2e', aosym='s4')

get_mo_eri = kernel
