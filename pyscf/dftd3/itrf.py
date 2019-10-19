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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
DFT-D3 interface.

This interface is based on the open source project
https://github.com/cuanto/libdftd3
'''

import sys
import ctypes
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import __config__

try:
    from pyscf.dftd3 import settings
except ImportError:
    settings = lambda: None
    settings.DFTD3PATH = getattr(__config__, 'dftd3_DFTD3PATH', None)

try:
    libdftd3 = numpy.ctypeslib.load_library('libdftd3.so', settings.DFTD3PATH)
except:
    libdftd3 = None


# For code compatibility in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str


FUNC_CODE = {
#   mf.xc            name in dftd3 library      dftd3 versions
    'BLYP'         : ('b-lyp',                  (2,3,4,5,6)),
    'B88,LYP'      : ('b-lyp',                  (2,3,4,5,6)),
    'BP86'         : ('b-p',                    (2,3,4,5,6)),
    'B88,P86'      : ('b-p',                    (2,3,4,5,6)),
    'B88B95'       : ('b1b95',                  (3,4)),
    'B1B95'        : ('b1b95',                  (3,4)),
    'B3LYP'        : ('b3-lyp',                 (2,3,4,5,6)),
    'B3LYP/631GD'  : ('b3-lyp/6-31gd',          (4,)),
    'B3LYPG'       : ('b3-lyp',                 (2,3,4,5,6)),
    'B3PW91'       : ('b3pw91',                 (3,4)),
    'B97-D'        : ('b97-d',                  (2,3,4,5,6)),
    'BHANDHLYP'    : ('bh-lyp',                 (3,4)),
    'BMK,BMK'      : ('bmk',                    (3,4)),
    'BOP'          : ('bop',                    (3,4)),
    'B88,OP_B88'   : ('bop',                    (3,4)),
    'BPBE'         : ('bpbe',                   (3,4)),
    'B88,PBE'      : ('bpbe',                   (3,4)),
    'CAMB3LYP'     : ('cam-b3lyp',              (3,4)),
    'CAM_B3LYP'    : ('cam-b3lyp',              (3,4)),
    #''             : ('dsd-blyp',               (2,4)),
    #''             : ('dsd-blyp-fc',            (4,)),
    'HCTH-120'     : ('hcth120',                (3,4)),
    'HF'           : ('hf',                     (3,4)),
    #''             : ('hf/minis',               (4,)),
    #''             : ('hf/mixed',               (4,)),
    'HF/SV'        : ('hf/sv',                  (4,)),
    #''             : ('hf3c',                   (4,)),
    #''             : ('hf3cv',                  (4,)),
    'HSE06'        : ('hse06',                  (3,4)),
    'HSE_SOL'      : ('hsesol',                 (4,)),
    'LRC-WPBE'     : ('lc-wpbe',                (3,4,5,6)),
    'LRC-WPBEH'    : ('lc-wpbe',                (3,4,5,6)),
    'M05'          : ('m05',                    (3,)),
    'M05,M05'      : ('m05',                    (3,)),
    'M05-2X'       : ('m052x',                  (3,)),
    'M06'          : ('m06',                    (3,)),
    'M06,M06'      : ('m06',                    (3,)),
    'M06-2X'       : ('m062x',                  (3,)),
    'M06_HF'       : ('m06hf',                  (3,)),
    'M06-L'        : ('m06l',                   (3,)),
    #''             : ('mpw1b95',                (3,4)),
    #''             : ('mpwb1k',                 (3,4)),
    #''             : ('mpwlyp',                 (3,4)),
    'OLYP'         : ('o-lyp',                  (3,4)),
    'OPBE'         : ('opbe',                   (3,4)),
    'OTPSS_D'      : ('otpss',                  (3,4)),
    'OTPSS-D'      : ('otpss',                  (3,4)),
    'PBE'          : ('pbe',                    (2,3,4,5,6)),
    'PBE,PBE'      : ('pbe',                    (2,3,4,5,6)),
    'PBE0'         : ('pbe0',                   (2,3,4,5,6)),
    'PBEH'         : ('pbe0',                   (2,3,4,5,6)),
    #''             : ('pbeh3c',                 (4,)),
    #''             : ('pbeh-3c',                (4,)),
    'PBESOL'       : ('pbesol',                 (3,4)),
    #''             : ('ptpss',                  (3,4)),
    #''             : ('pw1pw',                  (4,)),
    #''             : ('pw6b95',                 (2,3,4)),
    #''             : ('pwb6k',                  (4,)),
    #''             : ('pwgga',                  (4,)),
    #''             : ('pwpb95',                 (3,4)),
    'REVPBE'       : ('revpbe',                 (2,3,4)),
    'REVPBE0'      : ('revpbe0',                (3,4)),
    #''             : ('revpbe38',               (3,4)),
    #''             : ('revssb',                 (3,4)),
    'RPBE'         : ('rpbe',                   (3,4)),
    'RPBE,RPBE'    : ('rpbe',                   (3,4)),
    'RPW86,PBE'    : ('rpw86-pbe',              (3,4)),
    'SLATER'       : ('slater-dirac-exchange',  (3,)),
    'XALPHA'       : ('slater-dirac-exchange',  (3,)),
    'SSB,PBE'      : ('ssb',                    (3,4)),
    'TPSS'         : ('tpss',                   (2,3,4)),
    'TPSS0'        : ('tpss0',                  (3,4)),
    'TPSSH'        : ('tpssh',                  (3,4)),
    #''             : ('dftb3',                  (4,)),
}


def dftd3(scf_method):
    '''Apply DFT-D3 corrections to SCF or MCSCF methods

    Args:
        scf_method : a HF or DFT object

    Returns:
        Same method object as the input scf_method with DFT-D3 energy
        corrections

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = dftd3(dft.RKS(mol))
    >>> mf.kernel()
    -101.940495711284
    '''
    from pyscf.scf import hf
    from pyscf.mcscf import casci
    assert(isinstance(scf_method, hf.SCF) or
           isinstance(scf_method, casci.CASCI))

    # DFT-D3 has been initialized
    if getattr(scf_method, 'with_dftd3', None):
        return scf_method

    method_class = scf_method.__class__

    # A DFTD3 extension class is defined because other extensions are applied
    # based on the dynamic class. If DFT-D3 correction was applied by patching
    # the functions of object scf_method, these patches may not be realized by
    # other extensions.
    class DFTD3(method_class, _DFTD3):
        def dump_flags(self, verbose=None):
            method_class.dump_flags(self, verbose)
            if self.with_dftd3:
                self.with_dftd3.dump_flags(verbose)
            return self

        def energy_nuc(self):
            # Adding DFT D3 correction to nuclear part because it is computed
            # based on nuclear coordinates only.  It does not depend on
            # quantum effects.
            enuc = method_class.energy_nuc(self)
            if self.with_dftd3:
                enuc += self.with_dftd3.kernel()[0]
            return enuc

        def nuc_grad_method(self):
            scf_grad = method_class.nuc_grad_method(self)
            return grad(scf_grad)
        Gradients = lib.alias(nuc_grad_method, alias_name='Gradients')

    mf = DFTD3.__new__(DFTD3)
    mf.__dict__.update(scf_method.__dict__)

    with_dftd3 = _DFTD3(mf.mol)
    if isinstance(scf_method, casci.CASCI):
        with_dftd3.xc = 'hf'
    else:
        with_dftd3.xc = getattr(scf_method, 'xc', 'HF').upper().replace(' ', '')
    mf.with_dftd3 = with_dftd3
    mf._keys.update(['with_dftd3'])
    return mf

def grad(scf_grad):
    '''Apply DFT-D3 corrections to SCF or MCSCF nuclear gradients methods

    Args:
        scf_grad : a HF or DFT gradient object (grad.HF or grad.RKS etc)
            Once this function is applied on the SCF object, it affects all
            post-HF calculations eg MP2, CCSD, MCSCF etc

    Returns:
        Same gradeints method object as the input scf_grad method

    Examples:

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = mm_charge(scf.RHF(mol), [(0.5,0.6,0.8)], [-0.3])
    >>> mf.kernel()
    -101.940495711284
    >>> hfg = mm_charge_grad(grad.hf.RHF(mf), coords, charges)
    >>> hfg.kernel()
    [[-0.25912357 -0.29235976 -0.38245077]
     [-1.70497052 -1.89423883  1.2794798 ]]
    '''
    from pyscf.grad import rhf as rhf_grad
    assert(isinstance(scf_grad, rhf_grad.Gradients))

    if not getattr(scf_grad.base, 'with_dftd3', None):
        scf_grad.base = dftd3(scf_grad.base)

    grad_class = scf_grad.__class__
    class DFTD3Grad(grad_class, _DFTD3Grad):
        def grad_nuc(self, mol=None, atmlst=None):
            nuc_g = grad_class.grad_nuc(self, mol, atmlst)
            with_dftd3 = getattr(self.base, 'with_dftd3', None)
            if with_dftd3:
                d3_g = with_dftd3.kernel()[1]
                if atmlst is not None:
                    d3_g = d3_g[atmlst]
                nuc_g += d3_g
            return nuc_g
    mfgrad = DFTD3Grad.__new__(DFTD3Grad)
    mfgrad.__dict__.update(scf_grad.__dict__)
    return mfgrad


class _DFTD3(object):
    def __init__(self, mol):
        self.mol = mol
        self.verbose = mol.verbose
        self.xc = 'hf'
        self.version = 4  # 1..6
        self.libdftd3 = libdftd3
        self.edisp = None
        self.grads = None

    def dump_flags(self, verbose=None):
        logger.info(self, '** DFTD3 parameter **')
        logger.info(self, 'func %s', self.xc)
        logger.info(self, 'version %s', self.version)
        return self

    def kernel(self):
        mol = self.mol
        basis_type = _get_basis_type(mol)
        if self.xc in FUNC_CODE:
            func, supported_versions = FUNC_CODE[self.xc]
            if func == 'b3lyp' and basis_type == '6-31gd':
                func, supported_versions = FUNC_CODE['B3LYP/631GD']
            elif func == 'hf' and basis_type == 'sv':
                func, supported_versions = FUNC_CODE['HF/SV']
        else:
            raise RuntimeError('Functional %s not found' % self.xc)
        assert(self.version in supported_versions)

        # dft-d3 has special treatment for def2-TZ basis
        tz = (basis_type == 'def2-TZ')

        coords = mol.atom_coords()
        nuc_types = [gto.charge(mol.atom_symbol(ia))
                     for ia in range(mol.natm)]
        nuc_types = numpy.asarray(nuc_types, dtype=numpy.int32)

        edisp = ctypes.c_double(0)
        grads = numpy.zeros((mol.natm,3))

        drv = self.libdftd3.wrapper
        drv(ctypes.c_int(mol.natm),
            coords.ctypes.data_as(ctypes.c_void_p),
            nuc_types.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_char_p(func),
            ctypes.c_int(self.version),
            ctypes.c_int(tz),
            ctypes.byref(edisp),
            grads.ctypes.data_as(ctypes.c_void_p))
        self.edisp = edisp.value
        self.grads = grads
        return edisp.value, grads

    def reset(self, mol):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        self.mol = mol
        return self

class _DFTD3Grad:
    pass

def _get_basis_type(mol):
    def classify(mol_basis):
        basis_type = 'other'
        if isinstance(mol_basis, (str, unicode)):
            mol_basis = gto.basis._format_basis_name(mol_basis)
            if mol_basis[:6] == 'def2tz':
                basis_type = 'def2-TZ'
            elif mol_basis[:6] == 'def2sv':
                basis_type = 'sv'
            elif mol_basis[:5] == '631g*':
                basis_type = '6-31gd'
            elif mol_basis[:4] == '631g' and 'd' in mol_basis:
                basis_type = '6-31gd'
        return basis_type

    if isinstance(mol.basis, dict):
        basis_types = [classify(b) for b in mol.basis.values()]
        basis_type = 'other'
        for bt in basis_types:
            if bt != 'other':
                basis_type = bt
                break
        if (len(basis_types) > 1 and
            all(b == basis_type for b in basis_types)):
            logger.warn('Mutliple types of basis found in mol.basis. '
                        'Type %s is applied\n' % basis_type)
    else:
        basis_type = classify(mol.basis)
    return basis_type


if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = 'cc-pvdz'
    mol.build()

    mf = dftd3(scf.RHF(mol))
    print(mf.kernel()) # -75.99396273778923

    mfs = mf.as_scanner()
    e1 = mfs(''' O                  0.00100000    0.00000000   -0.11081188
             H                 -0.00000000   -0.84695236    0.59109389
             H                 -0.00000000    0.89830571    0.52404783 ''')
    e2 = mfs(''' O                 -0.00100000    0.00000000   -0.11081188
             H                 -0.00000000   -0.84695236    0.59109389
             H                 -0.00000000    0.89830571    0.52404783 ''')
    print((e1 - e2)/0.002 * lib.param.BOHR)
    mf.nuc_grad_method().kernel()

