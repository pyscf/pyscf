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

import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc.rintermediates import _get_vvvv

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# uccsd intermediates has been moved to gccsd intermediates

def _get_vvVV(eris):
    if eris.vvVV is None and hasattr(eris, 'VVL'):  # DF eris
        vvL = np.asarray(eris.vvL)
        VVL = np.asarray(eris.VVL)
        vvVV = lib.dot(vvL, VVL.T)
    elif len(eris.vvVV.shape) == 2:
        vvVV = np.asarray(eris.vvVV)
    else:
        return eris.vvVV

    nvira = int(np.sqrt(vvVV.shape[0]*2))
    nvirb = int(np.sqrt(vvVV.shape[1]*2))
    vvVV1 = np.zeros((nvira**2,nvirb**2))
    vtrila = np.tril_indices(nvira)
    vtrilb = np.tril_indices(nvirb)
    lib.takebak_2d(vvVV1, vvVV, vtrila[0]*nvira+vtrila[1], vtrilb[0]*nvirb+vtrilb[1])
    lib.takebak_2d(vvVV1, vvVV, vtrila[1]*nvira+vtrila[0], vtrilb[1]*nvirb+vtrilb[0])
    lib.takebak_2d(vvVV1, vvVV, vtrila[0]*nvira+vtrila[1], vtrilb[1]*nvirb+vtrilb[0])
    lib.takebak_2d(vvVV1, vvVV, vtrila[1]*nvira+vtrila[0], vtrilb[0]*nvirb+vtrilb[1])
    return vvVV1.reshape(nvira,nvira,nvirb,nvirb)

def _get_VVVV(eris):
    if eris.VVVV is None and hasattr(eris, 'VVL'):  # DF eris
        VVL = np.asarray(eris.VVL)
        nvir = int(np.sqrt(eris.VVL.shape[0]*2))
        return ao2mo.restore(1, lib.dot(VVL, VVL.T), nvir)
    elif len(eris.VVVV.shape) == 2:
        nvir = int(np.sqrt(eris.VVVV.shape[0]*2))
        return ao2mo.restore(1, np.asarray(eris.VVVV), nvir)
    else:
        return eris.VVVV
