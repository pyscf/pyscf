#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>

import numpy
from pyscf.ao2mo import incore
from pyscf.ao2mo import outcore
from pyscf.ao2mo import r_outcore

from pyscf.ao2mo.addons import load, restore

def full(eri_or_mol, mo_coeff, *args, **kwargs):
    if isinstance(eri_or_mol, numpy.ndarray):
        return incore.full(eri_or_mol, mo_coeff, *args, **kwargs)
    else:
        if 'intor' in kwargs and ('_sph' not in kwargs['intor']):
            mod = r_outcore
        else:
            mod = outcore
        if len(args) > 0 and isinstance(args[0], str): # args[0] is erifile
            fn = getattr(mod, 'full')
        else:
            fn = getattr(mod, 'full_iofree')
        return fn(eri_or_mol, mo_coeff, *args, **kwargs)

def general(eri_or_mol, mo_coeffs, *args, **kwargs):
    if isinstance(eri_or_mol, numpy.ndarray):
        return incore.general(eri_or_mol, mo_coeffs, *args, **kwargs)
    else:
        if 'intor' in kwargs and ('_sph' not in kwargs['intor']):
            mod = r_outcore
        else:
            mod = outcore
        if len(args) > 0 and isinstance(args[0], str): # args[0] is erifile
            fn = getattr(mod, 'general')
        else:
            fn = getattr(mod, 'general_iofree')
        return fn(eri_or_mol, mo_coeffs, *args, **kwargs)

def kernel(eri_or_mol, mo_coeffs, *args, **kwargs):
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        return full(eri_or_mol, mo_coeffs, *args, **kwargs)
    else:
        return general(eri_or_mol, mo_coeffs, *args, **kwargs)


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf.ao2mo import incore
    from pyscf.ao2mo import addons
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = 'ccpvdz')

    mf = scf.RHF(mol)
    mf.scf()

    eri0 = full(mf._eri, mf.mo_coeff)
    mos = (mf.mo_coeff,)*4
    print(numpy.allclose(eri0, full(mol, mf.mo_coeff)))
    print(numpy.allclose(eri0, general(mf._eri, mos)))
    print(numpy.allclose(eri0, general(mol, mos)))
    with load(full(mol, mf.mo_coeff, 'h2oeri.h5', dataname='dat1'), 'dat1') as eri1:
        print(numpy.allclose(eri0, eri1))
    with load(general(mol, mos, 'h2oeri.h5', dataname='dat1'), 'dat1') as eri1:
        print(numpy.allclose(eri0, eri1))

