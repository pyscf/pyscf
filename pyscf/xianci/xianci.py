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
# Contact "Bingbing Suo" <bsuo@nwu.edu.cn> to download and install Xian-CI
# program.
#

'''
Generate Xian-CI input file and integral file
'''

import numpy
import h5py
from pyscf import ao2mo
from pyscf import symm


def write_integrals(xci, orb):
    mol = xci.mol
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orb)
    h1e = reduce(numpy.dot, (orb.T, xci.get_hcore(), orb))
    norb = orb.shape[1]
    if xci._eri is not None:
        h2e = ao2mo.restore(1, ao2mo.full(xci._eri, orb), norb)
    else:
        h2e = ao2mo.restore(1, ao2mo.full(mol, orb), norb)

    with h5py.File(xci.integralfile, 'w') as f:
        f['h1e']    = h1e
        f['h2e']    = h2e
        f['norb'  ] = numpy.array(norb, dtype=numpy.int32)
        f['group' ] = mol.groupname
        f['orbsym'] = numpy.asarray(orbsym, dtype=numpy.int32)
        f['ecore' ] = mol.energy_nuc()

def write_input(xci, orb):
    mol = xci.mol
    with open(xci.inputfile, 'w') as f:
        f.write('%d 1 %d %g %g    ! nroots; default; IC modes(0: UC, 1: WK, 2: CW, 3: VD, 4: FC, 8: DS, 9: SS, 10: SD); PLP cut; Ref cut \n' %
                (xci.nroots, xci.ic_mode, xci.plpcut, xci.refcut))
        f.write('%d  %2.1f  %d   %d  %f ! CI electrons; spin value; total irrep; irrep index; CI coeff print criterion \n' %
                (xci.nelec, mol.spin*.5, len(mol.irrep_id), xci.wfnsym,
                 xci.print_thr))
        f.write('%d %d %d %d              !  norb_frz,norb_dz,norb_act,next_frz\n' %
                (xci.frozen, xci.ndocc, xci.ncas, xci.next_frozen))
        f.write('%d %d                  ! RUNPT2 modes(1: MS-SR-MRPT2; 2: MS-MR-MRPT2; 3: Dyall-MRPT2) ; RUNCI modes(0: ICMRCI; 1: ICMRPT2)\n' %
                (xci.pt2_mode, xci.ic_mode))
        f.write('0 0 0 0 0 0\n')
        f.write('4   0               ! default\n')

class XianCiInpHandler(object):
    def __init__(self, method, inputfile, integralfile):
        self.mol = method.mol
        assert(mol.symmetry)
        if hasattr(method, '_eri'):
            self._eri = method._eri
        elif hasattr(method, '_scf') and hasattr(method._scf, '_eri'):
            self._eri = method._scf._eri
        else:
            self._eri = None
        self.get_hcore = method.get_hcore
        self.mo_coeff = method.mo_coeff

        #self.exe = settings.XIANCIEXE
        self.inputfile = inputfile
        self.integralfile = integralfile

        self.nroots = 1
        self.nelec = mol.nelectron
        self.ic_mode = 0
        self.plpcut = 1e-3
        self.refcut = 1e-3
        self.print_thr = 0.05
        self.wfnsym = 1
        self.frozen = 0
        self.ndocc = 0
        self.ncas = self.mo_coeff.shape[1]
        self.next_frozen = 0
        self.pt2_mode = 0
        self.ic_mode = 0

    def gen_input(self):
        write_input(self, self.mo_coeff)
        write_integrals(self, self.mo_coeff)

    def kernel(self):
        self.gen_input()

if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(
        atom = [['O',(0, 0, 0)],
                ['H',(0.790689766, 0, 0.612217330)],
                ['H',(-0.790689766, 0, 0.612217330)]],
        basis = 'ccpvdz',
        symmetry = 1)

    mf = scf.RHF(mol).run()
    XianCiInpHandler(mf, 'drt.inp', 'eri.h5').kernel()
