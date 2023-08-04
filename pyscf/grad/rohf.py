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
Non-relativistic ROHF analytical nuclear gradients
'''

import numpy
from functools import reduce
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import uhf as uhf_grad

def make_rdm1e(mf_grad, mo_energy, mo_coeff, mo_occ):
    '''Energy weighted density matrix'''
    mf = mf_grad.base
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    fock = mf.get_fock(dm=dm)
    fa, fb = fock.focka, fock.fockb
    mocc_a = mo_coeff[:,mo_occ>0 ]
    mocc_b = mo_coeff[:,mo_occ==2]
    rdm1e_a = reduce(numpy.dot, (mocc_a, mocc_a.conj().T, fa, mocc_a, mocc_a.conj().T))
    rdm1e_b = reduce(numpy.dot, (mocc_b, mocc_b.conj().T, fb, mocc_b, mocc_b.conj().T))
    return numpy.array((rdm1e_a, rdm1e_b))

def _tag_rdm1 (self, dm, mo_coeff, mo_occ):
    mo_coeff = numpy.stack ((mo_coeff,mo_coeff), axis=0)
    mo_occa = numpy.array (mo_occ>0, dtype=numpy.double)
    mo_occb = numpy.array (mo_occ==2, dtype=numpy.double)
    assert (mo_occa.sum () + mo_occb.sum () == mo_occ.sum())
    mo_occ = numpy.vstack ((mo_occa,mo_occb))
    return lib.tag_array (dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

class Gradients(rhf_grad.Gradients):
    '''Non-relativistic restricted open-shell Hartree-Fock gradients
    '''

    get_veff = uhf_grad.get_veff

    make_rdm1e = make_rdm1e

    grad_elec = uhf_grad.grad_elec

    _tag_rdm1 = _tag_rdm1

Grad = Gradients

from pyscf import scf
scf.rohf.ROHF.Gradients = lib.class_as_method(Gradients)
