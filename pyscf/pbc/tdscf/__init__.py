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

from pyscf.pbc import scf
from pyscf.pbc.tdscf import rhf
from pyscf.pbc.tdscf import uhf
from pyscf.pbc.tdscf import krhf
from pyscf.pbc.tdscf import kuhf

try:
    from pyscf.pbc.tdscf import rks
    from pyscf.pbc.tdscf import uks
    from pyscf.pbc.tdscf import krks
    from pyscf.pbc.tdscf import kuks
except (ImportError, IOError):
    pass

def TDHF(mf):
    import numpy
    if isinstance(mf, scf.khf.KSCF):
        return KTDHF(mf)
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError('Only supports gamma-point TDHF')
    if isinstance(mf, scf.hf.KohnShamDFT):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    #TODO: mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        # Is it correct to call TDUHF for ROHF?
        mf = mf.to_uhf()
    return mf.TDHF()

def TDA(mf):
    import numpy
    if isinstance(mf, scf.khf.KSCF):
        return KTDA(mf)
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError('Only supports gamma-point TDA')
    #TODO: mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, scf.hf.KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDA()

def TDDFT(mf):
    import numpy
    if isinstance(mf, scf.khf.KSCF):
        return KTDDFT(mf)
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError('Only supports gamma-point TDDFT')
    if isinstance(mf, scf.hf.KohnShamDFT):
        #TODO: mf = mf.remove_soscf()
        if isinstance(mf, scf.rohf.ROHF):
            mf = mf.to_uks()
        return mf.TDDFT()
    else:
        return TDHF(mf)

def KTDHF(mf):
    if isinstance(mf, scf.hf.KohnShamDFT):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    #TODO: mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        mf = mf.to_uhf()
    return mf.TDHF()

def KTDA(mf):
    #TODO: mf = mf.remove_soscf()
    if isinstance(mf, scf.rohf.ROHF):
        if isinstance(mf, scf.hf.KohnShamDFT):
            mf = mf.to_uks()
        else:
            mf = mf.to_uhf()
    return mf.TDA()

def KTDDFT(mf):
    if isinstance(mf, scf.hf.KohnShamDFT):
        #TODO: mf = mf.remove_soscf()
        if isinstance(mf, scf.rohf.ROHF):
            mf = mf.to_uks()
        return mf.TDDFT()
    else:
        return KTDHF(mf)

KTD = KTDDFT

