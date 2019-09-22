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
# Contributor(s):
# - Wirawan Purwanto
# - Qiming Sun <osirpt.sun@gmail.com>
#
#

import numpy
import h5py
from pyscf.lib.chkfile import load
from pyscf.lib.chkfile import dump, save
from pyscf.lib.chkfile import load_mol, save_mol


def load_mcscf(chkfile):
    return load_mol(chkfile), load(chkfile, 'mcscf')

def dump_mcscf(mc, chkfile=None, key='mcscf',
               e_tot=None, mo_coeff=None, ncore=None, ncas=None,
               mo_occ=None, mo_energy=None, e_cas=None, ci_vector=None,
               casdm1=None, overwrite_mol=True):
    '''Save CASCI/CASSCF calculation results or intermediates in chkfile.
    '''
    if chkfile is None: chkfile = mc.chkfile
    if ncore is None: ncore = mc.ncore
    if ncas is None: ncas = mc.ncas
    if e_tot is None: e_tot = mc.e_tot
    if e_cas is None: e_cas = mc.e_cas
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    #if ci_vector is None: ci_vector = mc.ci

    if h5py.is_hdf5(chkfile):
        fh5 = h5py.File(chkfile, 'a')
        if key in fh5:
            del(fh5[key])
    else:
        fh5 = h5py.File(chkfile, 'w')

    if 'mol' not in fh5:
        fh5['mol'] = mc.mol.dumps()
    elif overwrite_mol:
        del(fh5['mol'])
        fh5['mol'] = mc.mol.dumps()

    fh5[key+'/mo_coeff'] = mo_coeff

    def store(subkey, val):
        if val is not None:
            fh5[key+'/'+subkey] = val
    store('e_tot', e_tot)
    store('e_cas', e_cas)
    store('ci', ci_vector)
    store('ncore', ncore)
    store('ncas', ncas)
    store('mo_occ', mo_occ)
    store('mo_energy', mo_energy)
    store('casdm1', casdm1)
    fh5.close()
