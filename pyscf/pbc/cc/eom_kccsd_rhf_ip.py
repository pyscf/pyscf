#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Authors: Artem Pulkin, pyscf authors

from pyscf.pbc.lib.kpts_helper import VectorSplitter, VectorComposer
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.pbc.cc import kccsd_rhf

import numpy as np


def iter_12(cc_or_eom, k):
    """Iterates over IP index slices."""
    if isinstance(cc_or_eom, kccsd_rhf.RCCSD):
        cc = cc_or_eom
    else:
        cc = cc_or_eom._cc
    o, v = padding_k_idx(cc, kind="split")
    kconserv = cc.khelper.kconserv

    yield (o[k],)

    for ki in range(cc.nkpts):
        for kj in range(cc.nkpts):
            kb = kconserv[ki, k, kj]
            yield (ki,), (kj,), o[ki], o[kj], v[kb]


def amplitudes_to_vector(cc_or_eom, t1, t2, kshift=0, kconserv=None):
    """IP amplitudes to vector."""
    itr = iter_12(cc_or_eom, kshift)
    t1, t2 = np.asarray(t1), np.asarray(t2)

    vc = VectorComposer(t1.dtype)
    vc.put(t1[np.ix_(*next(itr))])
    for slc in itr:
        vc.put(t2[np.ix_(*slc)])
    return vc.flush()


def vector_to_amplitudes(cc_or_eom, vec, kshift=0):
    """IP vector to apmplitudes."""
    expected_vs = vector_size(cc_or_eom, kshift)
    if expected_vs != len(vec):
        raise ValueError("The size of the vector passed {:d} should be exactly {:d}".format(len(vec), expected_vs))

    itr = iter_12(cc_or_eom, kshift)

    nocc = cc_or_eom.nocc
    nmo = cc_or_eom.nmo
    nkpts = cc_or_eom.nkpts

    vs = VectorSplitter(vec)
    r1 = vs.get(nocc, slc=next(itr))
    r2 = np.zeros((nkpts, nkpts, nocc, nocc, nmo - nocc), vec.dtype)
    for slc in itr:
        vs.get(r2, slc=slc)
    return r1, r2


def vector_size(cc_or_eom, kshift=0):
    """The total number of elements in IP vector."""
    size = 0
    for slc in iter_12(cc_or_eom, kshift):
        size += np.prod(tuple(len(i) for i in slc))
    return size
