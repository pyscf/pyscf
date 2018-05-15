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

def TDHF(mf):
    import numpy
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError
    from pyscf import tdscf
    from pyscf.pbc import scf
    if hasattr(mf, 'xc'):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        #mf = scf.addons.convert_to_uhf(mf) # To remove newton decoration
        return tdscf.uhf.TDHF(mf)
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        return tdscf.rhf.TDHF(mf)

def TDA(mf):
    import numpy
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError
    from pyscf import tdscf
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        #mf = scf.addons.convert_to_uhf(mf)
        if hasattr(mf, 'xc'):
            return tdscf.uks.TDA(mf)
        else:
            return tdscf.uhf.TDA(mf)
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        if hasattr(mf, 'xc'):
            return tdscf.rks.TDA(mf)
        else:
            return tdscf.rhf.TDA(mf)

def TDDFT(mf):
    import numpy
    if numpy.abs(getattr(mf, 'kpt', 0)).max() > 1e-9:
        raise NotImplementedError
    from pyscf import tdscf
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        #mf = scf.addons.convert_to_uhf(mf)
        if hasattr(mf, 'xc'):
            if mf._numint.libxc.is_hybrid_xc(mf.xc):
                return tdscf.uks.TDDFT(mf)
            else:
                return tdscf.uks.TDDFTNoHybrid(mf)
        else:
            return tdscf.uhf.TDHF(mf)
    else:
        #mf = scf.addons.convert_to_rhf(mf)
        if hasattr(mf, 'xc'):
            if mf._numint.libxc.is_hybrid_xc(mf.xc):
                return tdscf.rks.TDDFT(mf)
            else:
                return tdscf.rks.TDDFTNoHybrid(mf)
        else:
            return tdscf.rhf.TDHF(mf)

def KTDHF(mf):
    from pyscf.pbc import scf
    if hasattr(mf, 'xc'):
        raise RuntimeError('TDHF does not support DFT object %s' % mf)
    if isinstance(mf, scf.uhf.UHF):
        return uhf.TDHF(mf)
    else:
        return rhf.TDHF(mf)

def KTDA(mf):
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        if hasattr(mf, 'xc'):
            return uks.TDA(mf)
        else:
            return uhf.TDA(mf)
    else:
        if hasattr(mf, 'xc'):
            return rks.TDA(mf)
        else:
            return rhf.TDA(mf)

def KTDDFT(mf):
    from pyscf.pbc import scf
    if isinstance(mf, scf.uhf.UHF):
        if hasattr(mf, 'xc'):
            if mf._numint.libxc.is_hybrid_xc(mf.xc):
                return uks.TDDFT(mf)
            else:
                return uks.TDDFTNoHybrid(mf)
        else:
            return uhf.TDHF(mf)
    else:
        if hasattr(mf, 'xc'):
            if mf._numint.libxc.is_hybrid_xc(mf.xc):
                return rks.TDDFT(mf)
            else:
                return rks.TDDFTNoHybrid(mf)
        else:
            return rhf.TDHF(mf)

KTD = KTDDFT

