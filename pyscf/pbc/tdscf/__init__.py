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

from pyscf.pbc.tdscf import rhf
from pyscf.pbc.tdscf import uhf
from pyscf.pbc.tdscf import rks
from pyscf.pbc.tdscf import uks
from pyscf.pbc.tdscf import krhf
from pyscf.pbc.tdscf import kuhf
from pyscf.pbc.tdscf import krks
from pyscf.pbc.tdscf import kuks

def TDHF(mf):
    import numpy
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError
    from pyscf.pbc import scf
    if getattr(mf, 'xc', None):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        #mf = scf.addons.convert_to_uhf(mf) # To remove newton decoration
        return uhf.TDHF(mf)
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        return rhf.TDHF(mf)

def TDA(mf):
    import numpy
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        #mf = scf.addons.convert_to_uhf(mf)
        if getattr(mf, 'xc', None):
            return uks.TDA(mf)
        else:
            return uhf.TDA(mf)
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        if getattr(mf, 'xc', None):
            return rks.TDA(mf)
        else:
            return rhf.TDA(mf)

def TDDFT(mf):
    import numpy
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        #mf = scf.addons.convert_to_uhf(mf)
        if getattr(mf, 'xc', None):
            return uks.tddft(mf)
        else:
            return uhf.TDHF(mf)
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        if getattr(mf, 'xc', None):
            return rks.tddft(mf)
        else:
            return rhf.TDHF(mf)

def KTDHF(mf):
    from pyscf.pbc import scf
    if getattr(mf, 'xc', None):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        return kuhf.TDHF(mf)
    else:
        return krhf.TDHF(mf)

def KTDA(mf):
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        if getattr(mf, 'xc', None):
            return kuks.TDA(mf)
        else:
            return kuhf.TDA(mf)
    else:
        if getattr(mf, 'xc', None):
            return krks.TDA(mf)
        else:
            return krhf.TDA(mf)

def KTDDFT(mf):
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        if getattr(mf, 'xc', None):
            return kuks.tddft(mf)
        else:
            return kuhf.TDHF(mf)
    else:
        if getattr(mf, 'xc', None):
            return krks.tddft(mf)
        else:
            return krhf.TDHF(mf)

KTD = KTDDFT

