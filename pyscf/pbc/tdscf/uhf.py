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

from pyscf import lib
from pyscf.tdscf import uhf


class TDA(uhf.TDA):
    def gen_vind(self, mf):
        vind, hdiag = uhf.TDA.gen_vind(self, mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

    def nuc_grad_method(self):
        raise NotImplementedError

CIS = TDA


class TDHF(uhf.TDHF):
    def gen_vind(self, mf):
        vind, hdiag = uhf.TDHF.gen_vind(self, mf)
        def vindp(x):
            with lib.temporary_env(mf, exxdiv=None):
                return vind(x)
        return vindp, hdiag

    def nuc_grad_method(self):
        raise NotImplementedError


RPA = TDUHF = TDHF



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None

    mol.atom = [
        ['H' , (0. , 0. , .917)],
        ['F' , (0. , 0. , 0.)], ]
    mol.basis = '631g'
    mol.build()

    mf = scf.UHF(mol).run()
    td = TDA(mf)
    td.nstates = 5
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 11.01748568  11.01748568  11.90277134  11.90277134  13.16955369]

    td = TDHF(mf)
    td.nstates = 5
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 10.89192986  10.89192986  11.83487865  11.83487865  12.6344099 ]

    mol.spin = 2
    mf = scf.UHF(mol).run()
    td = TDA(mf)
    td.nstates = 6
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# FIXME:  first state
# [ 0.02231607274  3.32113736  18.55977052  21.01474222  21.61501962  25.0938973 ]

    td = TDHF(mf)
    td.nstates = 4
    td.verbose = 3
    print(td.kernel()[0] * 27.2114)
# [ 3.31267103  18.4954748   20.84935404  21.54808392]

