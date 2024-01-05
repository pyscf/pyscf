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
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact

import numpy as np

def isdf_eri_robust_fit(mydf, W, aoRg, aoR, V_r, verbose=None):
    '''

    Ref:
    (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720

    '''

    cell = mydf.cell
    nao  = cell.nao
    ngrid = np.prod(cell.mesh)
    vol = cell.vol

    eri = numpy.zeros((nao,nao,nao,nao))

    pair_Rg = np.einsum('ix,jx->ijx', aoRg, aoRg)
    pair_R  = np.einsum('ix,jx->ijx', aoR, aoR)

    ### step 1, term1

    path = np.einsum_path('ijx,xy,kly->ijkl', pair_Rg, V_r, pair_R, optimize='optimal')

    if verbose is not None and verbose > 0:
        # print("aoRg.shape     = ", aoRg.shape)
        # print("aoR.shape      = ", aoR.shape)
        # print("V_r.shape      = ", V_r.shape)
        print("path for term1 is ", path[0])
        print("opt            is ", path[1])

    # exit(1)

    path    = path[0]
    eri_tmp = np.einsum('ijx,xy,kly->ijkl', pair_Rg, V_r, pair_R, optimize=path)

    ### step 2, term2

    eri = eri_tmp + eri_tmp.transpose(2,3,0,1)

    ### step 3, term3

    path = np.einsum_path('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize='optimal')

    if verbose is not None and verbose > 0:
        print("path for term3 is ", path[0])
        print("opt            is ", path[1])

    path    = path[0]
    eri    -= np.einsum('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize=path)
    # eri     = np.einsum('ijx,xy,kly->ijkl', pair_Rg, W, pair_Rg, optimize=path)

    print("ngrids = ", np.prod(cell.mesh))

    return eri * ngrid / vol


def get_eri(mydf, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_get_eri_compact', True)):

    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'isdf_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros((nao,nao,nao,nao))

    kpti, kptj, kptk, kptl = kptijkl
    q = kptj - kpti
    coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    coords = cell.gen_uniform_grids(mydf.mesh)
    max_memory = mydf.max_memory - lib.current_memory()[0]

####################
# gamma point, the integral is real and with s4 symmetry
    if gamma_point(kptijkl):

        #:ao_pairs_G = get_ao_pairs_G(mydf, kptijkl[:2], q, compact=compact)
        #:ao_pairs_G *= numpy.sqrt(coulG).reshape(-1,1)
        #:eri = lib.dot(ao_pairs_G.T, ao_pairs_G, cell.vol/ngrids**2)
        # ao = mydf._numint.eval_ao(cell, coords, kpti)[0]
        # ao = numpy.asarray(ao.T, order='C')
        # eri = _contract_compact(mydf, (ao,ao), coulG, max_memory=max_memory)
        
        eri = isdf_eri_robust_fit(mydf, mydf.W, mydf.aoRg, mydf.aoR, mydf.V_R, verbose=mydf.cell.verbose)
        
        if compact:
            return ao2mo.restore(4, eri, nao)
        else:
            return eri.reshape(nao**2,nao**2)
    else:
        raise NotImplementedError


def general(mydf, mo_coeffs, kpts=None,
            compact=getattr(__config__, 'pbc_df_ao2mo_general_compact', True)):
    '''General MO integral transformation'''

    from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
    warn_pbc2d_eri(mydf)
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    mo_coeffs = [numpy.asarray(mo, order='F') for mo in mo_coeffs]
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(cell, 'fft_ao2mo: momentum conservation not found in '
                        'the given k-points %s', kptijkl)
        return numpy.zeros([mo.shape[1] for mo in mo_coeffs])

    allreal = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)
    q = kptj - kpti
    # coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    # coords = cell.gen_uniform_grids(mydf.mesh)
    max_memory = mydf.max_memory - lib.current_memory()[0]

    if gamma_point(kptijkl) and allreal:

        if ((iden_coeffs(mo_coeffs[0], mo_coeffs[1]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and
             iden_coeffs(mo_coeffs[0], mo_coeffs[3]))):
            
            moRg = numpy.asarray(lib.dot(mo_coeffs[0].T, mydf.aoRg), order='C')
            moR  = numpy.asarray(lib.dot(mo_coeffs[0].T, mydf.aoR), order='C')

            eri = isdf_eri_robust_fit(mydf, mydf.W, moRg, moR, mydf.V_R, verbose=mydf.cell.verbose)
        
            if compact:
                return ao2mo.restore(4, eri, nao)
            else:
                return eri.reshape(nao**2, nao**2)
        else:
            
            raise NotImplementedError

    else:
        raise NotImplementedError
    
    return

def ao2mo_7d(mydf, mo_coeff_kpts, kpts=None, factor=1, out=None):
    raise NotImplementedError