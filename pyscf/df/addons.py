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

import sys
import copy
import numpy
from pyscf.lib import logger
from pyscf import gto
from pyscf import ao2mo
from pyscf.data import elements
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf import __config__

DFBASIS = getattr(__config__, 'df_addons_aug_etb_beta', 'weigend')
ETB_BETA = getattr(__config__, 'df_addons_aug_dfbasis', 2.0)
FIRST_ETB_ELEMENT = getattr(__config__, 'df_addons_aug_start_at', 36)  # 'Rb'

# For code compatiblity in python-2 and python-3
if sys.version_info >= (3,):
    unicode = str

# Obtained from http://www.psicode.org/psi4manual/master/basissets_byfamily.html
DEFAULT_AUXBASIS = {
    # AO basis       JK-fit                     MP2-fit
    'ccpvdz'      : ('cc-pvdz-jkfit'          , 'cc-pvdz-ri'         ),
    'ccpvdpdz'    : ('cc-pvdz-jkfit'          , 'cc-pvdz-ri'         ),
    'augccpvdz'   : ('aug-cc-pvdz-jkfit'      , 'aug-cc-pvdz-ri'     ),
    'augccpvdpdz' : ('aug-cc-pvdz-jkfit'      , 'aug-cc-pvdz-ri'     ),
    'ccpvtz'      : ('cc-pvtz-jkfit'          , 'cc-pvtz-ri'         ),
    'augccpvtz'   : ('aug-cc-pvtz-jkfit'      , 'aug-cc-pvtz-ri'     ),
    'ccpvqz'      : ('cc-pvqz-jkfit'          , 'cc-pvqz-ri'         ),
    'augccpvqz'   : ('aug-cc-pvqz-jkfit'      , 'aug-cc-pvqz-ri'     ),
    'ccpv5z'      : ('cc-pv5z-jkfit'          , 'cc-pv5z-ri'         ),
    'augccpv5z'   : ('aug-cc-pv5z-jkfit'      , 'aug-cc-pv5z-ri'     ),
    'def2svp'     : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    'def2svpd'    : ('def2-svp-jkfit'         , 'def2-svpd-ri'       ),
    'def2tzvp'    : ('def2-tzvp-jkfit'        , 'def2-tzvp-ri'       ),
    'def2tzvpd'   : ('def2-tzvp-jkfit'        , 'def2-tzvpd-ri'      ),
    'def2tzvpp'   : ('def2-tzvpp-jkfit'       , 'def2-tzvpp-ri'      ),
    'def2tzvppd'  : ('def2-tzvpp-jkfit'       , 'def2-tzvppd-ri'     ),
    'def2qzvp'    : ('def2-qzvp-jkfit'        , 'def2-qzvp-ri'       ),
    'def2qzvpd'   : ('def2-qzvp-jkfit'        , None                 ),
    'def2qzvpp'   : ('def2-qzvpp-jkfit'       , 'def2-qzvpp-ri'      ),
    'def2qzvppd'  : ('def2-qzvpp-jkfit'       , 'def2-qzvppd-ri'     ),
    'sto3g'       : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    '321g'        : ('def2-svp-jkfit'         , 'def2-svp-ri'        ),
    '631g'        : ('cc-pvdz-jkfit'          , 'cc-pvdz-ri'         ),
    '631+g'       : ('heavy-aug-cc-pvdz-jkfit', 'heavyaug-cc-pvdz-ri'),
    '631++g'      : ('aug-cc-pvdz-jkfit'      , 'aug-cc-pvdz-ri'     ),
    '6311g'       : ('cc-pvtz-jkfit'          , 'cc-pvtz-ri'         ),
    '6311+g'      : ('heavy-aug-cc-pvtz-jkfit', 'heavyaug-cc-pvtz-ri'),
    '6311++g'     : ('aug-cc-pvtz-jkfit'      , 'aug-cc-pvtz-ri'     ),
}

class load(ao2mo.load):
    '''load 3c2e integrals from hdf5 file. It can be used in the context
    manager:

    with load(cderifile) as eri:
        print(eri.shape)
    '''
    def __init__(self, eri, dataname='j3c'):
        ao2mo.load.__init__(self, eri, dataname)


def aug_etb_for_dfbasis(mol, dfbasis=DFBASIS, beta=ETB_BETA,
                        start_at=FIRST_ETB_ELEMENT):
    '''augment weigend basis with even-tempered gaussian basis
    exps = alpha*beta^i for i = 1..N
    '''
    nuc_start = gto.charge(start_at)
    uniq_atoms = set([a[0] for a in mol._atom])

    newbasis = {}
    for symb in uniq_atoms:
        nuc_charge = gto.charge(symb)
        if nuc_charge < nuc_start:
            newbasis[symb] = dfbasis
        #?elif symb in mol._ecp:
        else:
            conf = elements.CONFIGURATION[nuc_charge]
            max_shells = 4 - conf.count(0)
            emin_by_l = [1e99] * 8
            emax_by_l = [0] * 8
            l_max = 0
            for b in mol._basis[symb]:
                l = b[0]
                l_max = max(l_max, l)
                if l >= max_shells+1:
                    continue

                if isinstance(b[1], int):
                    e_c = numpy.array(b[2:])
                else:
                    e_c = numpy.array(b[1:])
                es = e_c[:,0]
                cs = e_c[:,1:]
                es = es[abs(cs).max(axis=1) > 1e-3]
                emax_by_l[l] = max(es.max(), emax_by_l[l])
                emin_by_l[l] = min(es.min(), emin_by_l[l])

            l_max1 = l_max + 1
            emin_by_l = numpy.array(emin_by_l[:l_max1])
            emax_by_l = numpy.array(emax_by_l[:l_max1])

# Estimate the exponents ranges by geometric average
            emax = numpy.sqrt(numpy.einsum('i,j->ij', emax_by_l, emax_by_l))
            emin = numpy.sqrt(numpy.einsum('i,j->ij', emin_by_l, emin_by_l))
            liljsum = numpy.arange(l_max1)[:,None] + numpy.arange(l_max1)
            emax_by_l = [emax[liljsum==ll].max() for ll in range(l_max1*2-1)]
            emin_by_l = [emin[liljsum==ll].min() for ll in range(l_max1*2-1)]
            # Tune emin and emax
            emin_by_l = numpy.array(emin_by_l) * 2  # *2 for alpha+alpha on same center
            emax_by_l = numpy.array(emax_by_l) * 2  #/ (numpy.arange(l_max1*2-1)*.5+1)

            ns = numpy.log((emax_by_l+emin_by_l)/emin_by_l) / numpy.log(beta)
            etb = []
            for l, n in enumerate(numpy.ceil(ns).astype(int)):
                if n > 0:
                    etb.append((l, n, emin_by_l[l], beta))
            if etb:
                newbasis[symb] = gto.expand_etbs(etb)
            else:
                raise RuntimeError(f'Failed to generate even-tempered auxbasis for {symb}')

    return newbasis

def aug_etb(mol, beta=ETB_BETA):
    '''To generate the even-tempered auxiliary Gaussian basis'''
    return aug_etb_for_dfbasis(mol, beta=beta, start_at=0)

def make_auxbasis(mol, mp2fit=False):
    '''Depending on the orbital basis, generating even-tempered Gaussians or
    the optimized auxiliary basis defined in DEFAULT_AUXBASIS
    '''
    uniq_atoms = set([a[0] for a in mol._atom])
    if isinstance(mol.basis, str):
        _basis = dict(((a, mol.basis) for a in uniq_atoms))
    elif 'default' in mol.basis:
        default_basis = mol.basis['default']
        _basis = dict(((a, default_basis) for a in uniq_atoms))
        _basis.update(mol.basis)
        del (_basis['default'])
    else:
        _basis = mol._basis

    auxbasis = {}
    for k in _basis:
        if isinstance(_basis[k], str):
            balias = gto.basis._format_basis_name(_basis[k])
            if gto.basis._is_pople_basis(balias):
                balias = balias.split('g')[0] + 'g'
            if balias in DEFAULT_AUXBASIS:
                if mp2fit:
                    auxb = DEFAULT_AUXBASIS[balias][1]
                else:
                    auxb = DEFAULT_AUXBASIS[balias][0]
                if auxb is not None:
                    try:
                        # Test if basis auxb for element k is available
                        gto.basis.load(auxb, k)
                    except BasisNotFoundError:
                        pass
                    else:
                        auxbasis[k] = auxb
                        logger.info(mol, 'Default auxbasis %s is used for %s %s',
                                    auxb, k, _basis[k])

    if len(auxbasis) != len(_basis):
        # Some AO basis not found in DEFAULT_AUXBASIS
        auxbasis, auxdefault = aug_etb(mol), auxbasis
        auxbasis.update(auxdefault)
        aux_etb = set(auxbasis) - set(auxdefault)
        if aux_etb:
            logger.info(mol, 'Even tempered Gaussians are generated as '
                        'DF auxbasis for  %s', ' '.join(aux_etb))
            for k in aux_etb:
                logger.debug(mol, '  ETB auxbasis for %s  %s', k, auxbasis[k])
    return auxbasis

def make_auxmol(mol, auxbasis=None):
    '''Generate a fake Mole object which uses the density fitting auxbasis as
    the basis sets.  If auxbasis is not specified, the optimized auxiliary fitting
    basis set will be generated according to the rules recorded in
    pyscf.df.addons.DEFAULT_AUXBASIS.  If the optimized auxiliary basis is not
    available (either not specified in DEFAULT_AUXBASIS or the basis set of the
    required elements not defined in the optimized auxiliary basis),
    even-tempered Gaussian basis set will be generated.

    See also the paper JCTC, 13, 554 about generating auxiliary fitting basis.
    '''
    pmol = copy.copy(mol)  # just need shallow copy

    if auxbasis is None:
        auxbasis = make_auxbasis(mol)
    elif '+etb' in auxbasis:
        dfbasis = auxbasis[:-4]
        auxbasis = aug_etb_for_dfbasis(mol, dfbasis)
    pmol.basis = auxbasis

    if isinstance(auxbasis, (str, unicode, list, tuple)):
        uniq_atoms = set([a[0] for a in mol._atom])
        _basis = dict([(a, auxbasis) for a in uniq_atoms])
    elif 'default' in auxbasis:
        uniq_atoms = set([a[0] for a in mol._atom])
        _basis = dict(((a, auxbasis['default']) for a in uniq_atoms))
        _basis.update(auxbasis)
        del (_basis['default'])
    else:
        _basis = auxbasis
    pmol._basis = pmol.format_basis(_basis)

    # Note: To pass parameters like gauge origin, rsh-omega to auxmol,
    # mol._env[:PTR_ENV_START] must be copied to auxmol._env
    pmol._atm, pmol._bas, pmol._env = \
            pmol.make_env(mol._atom, pmol._basis, mol._env[:gto.PTR_ENV_START])
    pmol._built = True
    logger.debug(mol, 'num shells = %d, num cGTOs = %d',
                 pmol.nbas, pmol.nao_nr())
    return pmol

del (DFBASIS, ETB_BETA, FIRST_ETB_ELEMENT)
