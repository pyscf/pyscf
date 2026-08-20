# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

'''
Density functional theory
=========================

Simple usage::

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='def2-tzvp')
    >>> mf = dft.RKS(mol)
    >>> mf.xc = 'pbe,pbe'
    >>> mf.run()
'''

try:
    from pyscf.dft import libxc
    XC = {**libxc.XC, **libxc.XC_ALIAS}
except (ImportError, OSError):
    XC = None
try:
    from pyscf.dft import xcfun
    if XC is None:
        XC = {**xcfun.XC, **xcfun.XC_ALIAS}
except (ImportError, OSError):
    pass
from pyscf import gto
#from pyscf.dft import xc
from pyscf.dft import rks
from pyscf.dft import roks
from pyscf.dft import uks
from pyscf.dft import gks
from pyscf.dft import rks_symm
from pyscf.dft import uks_symm
from pyscf.dft import gks_symm
from pyscf.dft import dks
from pyscf.dft import gen_grid
grid = gen_grid
from pyscf.dft import radi
from pyscf.dft import numint
from pyscf.df import density_fit
from pyscf.dft.rks import KohnShamDFT
from pyscf.dft.gen_grid import sg1_prune, nwchem_prune, treutler_prune, \
        stratmann, original_becke, Grids
from pyscf.dft.radi import BRAGG_RADII, COVALENT_RADII, \
        delley, mura_knowles, gauss_chebyshev, treutler, treutler_ahlrichs, \
        treutler_atomic_radii_adjust, becke_atomic_radii_adjust


def KS(mol, xc='LDA,VWN'):
    if mol.spin == 0:
        return RKS(mol, xc)
    else:
        return UKS(mol, xc)
KS.__doc__ = '''
A wrap function to create DFT object (RKS or UKS).\n
''' + rks.RKS.__doc__
DFT = KS

def RKS(mol, xc='LDA,VWN'):
    if mol.spin == 0:
        if not mol.symmetry or mol.groupname == 'C1':
            return rks.RKS(mol, xc)
        else:
            return rks_symm.RKS(mol, xc)
    else:
        return ROKS(mol, xc)
RKS.__doc__ = rks.RKS.__doc__

def ROKS(mol, xc='LDA,VWN'):
    if not mol.symmetry or mol.groupname == 'C1':
        return roks.ROKS(mol, xc)
    else:
        return rks_symm.ROKS(mol, xc)
ROKS.__doc__ = roks.ROKS.__doc__

def UKS(mol, xc='LDA,VWN'):
    if not mol.symmetry or mol.groupname == 'C1':
        return uks.UKS(mol, xc)
    else:
        return uks_symm.UKS(mol, xc)
UKS.__doc__ = uks.UKS.__doc__

def GKS(mol, xc='LDA,VWN'):
    if not mol.symmetry or mol.groupname == 'C1':
        return gks.GKS(mol, xc)
    else:
        return gks_symm.GKS(mol, xc)
GKS.__doc__ = gks.GKS.__doc__

def DKS(mol, xc='LDA,VWN'):
    from pyscf.scf import dhf
    if dhf.zquatev and mol.spin == 0:
        return dks.RDKS(mol, xc=xc)
    else:
        return dks.UDKS(mol, xc=xc)
UDKS = dks.UDKS
RDKS = dks.RDKS

def X2C(mol, *args):
    '''X2C Kohn-Sham'''
    from pyscf.scf import dhf
    from pyscf.x2c import dft
    if dhf.zquatev and mol.spin == 0:
        return dft.RKS(mol, *args)
    else:
        return dft.UKS(mol, *args)
X2C_KS = X2C

def RKSpU(mol, xc='LDA,VWN', **kwargs):
    from pyscf.dft import rkspu
    return rkspu.RKSpU(mol, xc=xc, **kwargs)

def UKSpU(mol, xc='LDA,VWN', **kwargs):
    from pyscf.dft import ukspu
    return ukspu.UKSpU(mol, xc=xc, **kwargs)


def RKS3C(mol, method='b97-3c'):
    '''Create a RKS object with the composite 3c method applied.

    Examples:

    >>> mf = mol.RKS3C().density_fit().run()
    >>> mf = mol.RKS3C(method='r2scan-3c').density_fit().run()

    See :func:`pyscf.dft.dft3c.dft3c` for the supported methods.
    '''
    return RKS(mol).dft3c(method)
RKS3C.__doc__ = rks.RKS.__doc__ + RKS3C.__doc__

def UKS3C(mol, method='b97-3c'):
    '''Create a UKS object with the composite 3c method applied.

    See :func:`pyscf.dft.dft3c.dft3c` for the supported methods.
    '''
    return UKS(mol).dft3c(method)
UKS3C.__doc__ = uks.UKS.__doc__ + UKS3C.__doc__

def ROKS3C(mol, method='b97-3c'):
    '''Create a ROKS object with the composite 3c method applied.

    See :func:`pyscf.dft.dft3c.dft3c` for the supported methods.
    '''
    return ROKS(mol).dft3c(method)
ROKS3C.__doc__ = roks.ROKS.__doc__ + ROKS3C.__doc__

def GKS3C(mol, method='b97-3c'):
    '''Create a GKS object with the composite 3c method applied.

    See :func:`pyscf.dft.dft3c.dft3c` for the supported methods.
    '''
    return GKS(mol).dft3c(method)
GKS3C.__doc__ = gks.GKS.__doc__ + GKS3C.__doc__
