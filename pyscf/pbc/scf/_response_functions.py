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

from pyscf import lib

def _get_jk_kshift(mf, dm_kpts, hermi, kpts, kshift, with_j=True, with_k=True,
                   omega=None):
    from pyscf.pbc.df.df_jk import get_j_kpts_kshift, get_k_kpts_kshift
    vj = vk = None
    if with_j:
        vj = get_j_kpts_kshift(mf.with_df, dm_kpts, kshift, hermi=hermi, kpts=kpts)
    if with_k:
        vk = get_k_kpts_kshift(mf.with_df, dm_kpts, kshift, hermi=hermi, kpts=kpts,
                               exxdiv=mf.exxdiv)
    return vj, vk
def _get_jk(mf, cell, dm1, hermi, kpts, kshift, with_j=True, with_k=True, omega=None):
    from pyscf.pbc import df
    if kshift == 0:
        return mf.get_jk(cell, dm1, hermi=hermi, kpts=kpts,
                         with_j=with_j, with_k=with_k, omega=omega)
    elif omega is not None and omega != 0:
        raise NotImplementedError
    elif mf.rsjk is not None or not isinstance(mf.with_df, df.df.DF):
        lib.logger.error(mf, 'Non-zero kshift is only supported by GDF/RSDF.')
        raise NotImplementedError
    else:
        return _get_jk_kshift(mf, dm1, hermi, kpts, kshift,
                              with_j=with_j, with_k=with_k, omega=omega)
def _get_j(mf, cell, dm1, hermi, kpts, kshift, omega=None):
    return _get_jk(mf, cell, dm1, hermi, kpts, kshift, True, False, omega)[0]
def _get_k(mf, cell, dm1, hermi, kpts, kshift, omega=None):
    return _get_jk(mf, cell, dm1, hermi, kpts, kshift, False, True, omega)[1]
