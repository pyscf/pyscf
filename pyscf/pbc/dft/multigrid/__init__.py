#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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

from .multigrid import MultiGridFFTDF
from .multigrid import (
    multigrid_fftdf as multigrid_fftdf,
    _gen_rhf_response as _gen_rhf_response,
    _gen_uhf_response as _gen_uhf_response,
    nr_rks as nr_rks_v1,
    nr_rks_fxc as nr_rks_fxc,
    nr_rks_fxc_st as nr_rks_fxc_st,
    nr_uks as nr_uks_v1,
    nr_uks_fxc as nr_uks_fxc
)

from .multigrid_pair import MultiGridFFTDF2
from .multigrid_pair import nr_rks as nr_rks_v2
from .multigrid_pair import nr_uks as nr_uks_v2

def nr_rks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    if isinstance(mydf, MultiGridFFTDF2):
        return nr_rks_v2(mydf, xc_code, dm_kpts, hermi=hermi, kpts=kpts,
                         kpts_band=kpts_band, with_j=with_j,
                         return_j=return_j, verbose=verbose) 
    elif isinstance(mydf, MultiGridFFTDF):
        return nr_rks_v1(mydf, xc_code, dm_kpts, hermi=hermi, kpts=kpts,
                         kpts_band=kpts_band, with_j=with_j,
                         return_j=return_j, verbose=verbose)
    else:
        raise TypeError("Wrong density fitting type for multigrid DFT.")

def nr_uks(mydf, xc_code, dm_kpts, hermi=1, kpts=None,
           kpts_band=None, with_j=False, return_j=False, verbose=None):
    if isinstance(mydf, MultiGridFFTDF2):
        return nr_uks_v2(mydf, xc_code, dm_kpts, hermi=hermi, kpts=kpts,
                         kpts_band=kpts_band, with_j=with_j,
                         return_j=return_j, verbose=verbose)
    elif isinstance(mydf, MultiGridFFTDF):
        return nr_uks_v1(mydf, xc_code, dm_kpts, hermi=hermi, kpts=kpts,
                         kpts_band=kpts_band, with_j=with_j,
                         return_j=return_j, verbose=verbose)
    else:
        raise TypeError("Wrong density fitting type for multigrid DFT.")
