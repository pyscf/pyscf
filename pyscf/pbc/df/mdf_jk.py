#!/usr/bin/env python
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
J. Chem. Phys. 147, 164119 (2017)
'''

import numpy
from pyscf.lib import logger
from pyscf.pbc.df import df_jk
from pyscf.pbc.df import aft_jk

#
# Divide the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

def density_fit(mf, auxbasis=None, mesh=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        mesh : tuple
            number of grids in each direction
        with_df : MDF object
    '''
    from pyscf.pbc.df import mdf
    from pyscf.pbc.scf.khf import KSCF
    if with_df is None:
        if isinstance(mf, KSCF):
            kpts = mf.kpts
        else:
            kpts = numpy.reshape(mf.kpt, (1,3))

        with_df = mdf.MDF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if mesh is not None:
            with_df.mesh = mesh

    mf = mf.copy()
    mf.with_df = with_df
    mf._eri = None
    return mf


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None):
    vj_kpts = df_jk.get_j_kpts(mydf, dm_kpts, hermi, kpts, kpts_band)
    vj_kpts += aft_jk.get_j_kpts(mydf, dm_kpts, hermi, kpts, kpts_band)
    return vj_kpts


def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpts_band=None,
               exxdiv=None):
    if exxdiv is not None and exxdiv != 'ewald':
        logger.warn(mydf, 'MDF does not support exxdiv %s. '
                    'exxdiv needs to be "ewald" or None', exxdiv)
        raise RuntimeError('GDF does not support exxdiv %s' % exxdiv)
    vk_kpts = df_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpts_band, None)
    vk_kpts += aft_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)
    return vk_kpts


##################################################
#
# Single k-point
#
##################################################

def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpts_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    vj1, vk1 = df_jk.get_jk(mydf, dm, hermi, kpt, kpts_band, with_j, with_k, None)
    vj, vk = aft_jk.get_jk(mydf, dm, hermi, kpt, kpts_band, with_j, with_k, exxdiv)
    if with_j: vj += vj1
    if with_k: vk += vk1
    return vj, vk


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf

    L = 5.
    n = 11
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.mesh = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5
    #print cell.nimgs
    #cell.nimgs = [4,4,4]

    mf = pscf.RHF(cell)
    auxbasis = 'weigend'
    mf = density_fit(mf, auxbasis)
    mf.with_df.mesh = (n,) * 3
    dm = mf.get_init_guess()
    vj = get_jk(mf.with_df, dm, exxdiv=mf.exxdiv, with_k=False)[0]
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698951141791')
    vj, vk = get_jk(mf.with_df, dm, exxdiv=mf.exxdiv)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698951141791')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.348980782463')

    kpts = cell.make_kpts([2]*3)[:4]
    from pyscf.pbc.df import MDF
    with_df = MDF(cell, kpts)
    with_df.auxbasis = 'weigend'
    with_df.mesh = [n] * 3
    dms = numpy.array([dm]*len(kpts))
    vj, vk = with_df.get_jk(dms, exxdiv=mf.exxdiv, kpts=kpts)
    print(numpy.einsum('ij,ji->', vj[0], dms[0]), - 46.69784775484954)
    print(numpy.einsum('ij,ji->', vj[1], dms[1]), - 46.69815612398015)
    print(numpy.einsum('ij,ji->', vj[2], dms[2]), - 46.69526857884275)
    print(numpy.einsum('ij,ji->', vj[3], dms[3]), - 46.69571387135913)
    print(numpy.einsum('ij,ji->', vk[0], dms[0]), - 37.27054185436858)
    print(numpy.einsum('ij,ji->', vk[1], dms[1]), - 37.27081050772277)
    print(numpy.einsum('ij,ji->', vk[2], dms[2]), - 37.27081024429790)
    print(numpy.einsum('ij,ji->', vk[3], dms[3]), - 37.27090527533867)
