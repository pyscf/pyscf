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

def convert_to_gcisd(myci):
    from pyscf import scf
    from pyscf.ci import gcisd
    if isinstance(myci, gcisd.GCISD):
        return myci

    mf = scf.addons.convert_to_ghf(myci._scf)
    gci = gcisd.GCISD(mf)
    assert(myci._nocc is None)
    assert(myci._nmo is None)
    gci.__dict__.update(myci.__dict__)
    gci._scf = mf
    gci.mo_coeff = mf.mo_coeff
    gci.mo_occ = mf.mo_occ
    if isinstance(myci.frozen, (int, np.integer)):
        gci.frozen = myci.frozen * 2
    else:
        raise NotImplementedError
    gci.ci = gcisd.from_rcisdvec(myci.ci, myci.nocc, mf.mo_coeff.orbspin)
    return gci
