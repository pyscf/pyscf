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
See also pyscf/hessian/rhf.py
'''

import numpy
from pyscf import lib
from pyscf.hessian import rhf as rhf_hess
from pyscf.data import elements
from pyscf.data import nist


def kernel(hobj):
    mol = hobj.mol
    atom_charges = mol.atom_charges()
    atmlst = numpy.where(atom_charges != 0)[0]  # Exclude ghost atoms
    natm = len(atmlst)
    mass = numpy.array([elements.MASSES[atom_charges[i]] for i in atmlst])
    #reduced_mass = 1./(1./mass).sum()

    if hobj.nroots is None:
        h = hobj.hess(atmlst=atmlst)
        h = numpy.einsum('ijxy,i,j->ijxy', h, mass**-.5, mass**-.5)
        h = h.transpose(0,2,1,3).reshape(natm*3,natm*3)
        e, c = numpy.linalg.eigh(h)
        c = c.T
    else: # Solve some roots
        h_op, hdiag = hobj.gen_hop()
        hdiag = hdiag.reshape(-1,3)[atmlst].ravel()
        def vib_mode_h_op(x):
            x1 = numpy.zeros((mol.natm,3))
            x1[atmlst] = numpy.einsum('i,ix->ix', mass**-.5, x.reshape(natm,3))
            hx = h_op(x1).reshape(-1,3)[atmlst]
            hx = numpy.einsum('i,ix->ix', mass**-.5, hx)
            return hx.ravel()

        def precond(x, e, *args):
            hdiagd = hdiag-e
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            return x/hdiagd
        e, c = lib.davidson(vib_mode_h_op, hdiag, precond, tol=hobj.conv_tol,
                            nroots=hobj.nroots, verbose=5)
        c = numpy.asarray(c)

    hartree_kj = nist.HARTREE2J*1e3
    unit2cm = ((hartree_kj * nist.AVOGADRO)**.5 / (nist.BOHR*1e-10)
               / (2*numpy.pi*nist.LIGHT_SPEED_SI) * 1e-2)
    hobj.freq = numpy.sign(e) * abs(e)**.5 * unit2cm
    lib.logger.note(hobj, 'Freq %s', hobj.freq)

# TODO: Remove translation and rotation modes
    hobj.mode = c.reshape(-1,natm,3)
    # Transform back to cartesian coordinates
    #hobj.mode = numpy.einsum('i,kix->kix', mass**-.5, c.reshape(-1,natm,3))
    return hobj.freq, hobj.mode

class Frequency(rhf_hess.Hessian):
    def __init__(self, mf):
        self.nroots = None
        self.freq = None
        self.mode = None
        self.conv_tol = 1e-3
        rhf_hess.Hessian.__init__(self, mf)
        self.atmlst = None

    kernel = kernel

Freq = Frequency


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)] ]
    mol.basis = '631g'
    mol.unit = 'B'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    mf.scf()
    n3 = mol.natm * 3
    hobj = rhf_hess.Hessian(mf)
    e2 = hobj.kernel()
    numpy.random.seed(1)
    x = numpy.random.random((mol.natm,3))
    e2x = numpy.einsum('abxy,ax->by', e2, x)
    print(lib.finger(e2x) - -0.19160804881270971)
    hop = Freq(mf).gen_hop()[0]
    print(lib.finger(hop(x)) - -0.19160804881270971)
    print(abs(e2x-hop(x).reshape(mol.natm,3)).sum())
    print(Freq(mf).set(nroots=1).kernel()[0])
    print(numpy.linalg.eigh(e2.transpose(0,2,1,3).reshape(n3,n3))[0])
