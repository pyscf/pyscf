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
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
Functions to support chkfiles with MPI
'''

import numpy as np
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import gto
from pyscf.lib import chkfile as chkutil
from pyscf.agf2 import mpi_helper
from pyscf.agf2.aux_space import GreensFunction, SelfEnergy


def load(chkfile, key):
    ''' Load array(s) from chkfile.

    See pyscf.lib.chkfile
    '''

    if mpi_helper.rank == 0:
        vals = chkutil.load(chkfile, key)
    else:
        vals = None

    mpi_helper.barrier()
    vals = mpi_helper.bcast_dict(vals)

    return vals


def load_mol(chkfile):
    ''' Load the mol from the chkfile.

    See pyscf.lib.chkfile
    '''

    if mpi_helper.rank == 0:
        mol = chkutil.load_mol(chkfile)
        dumps = mol.dumps()
    else:
        dumps = None

    mpi_helper.barrier()
    dumps = mpi_helper.bcast_dict(dumps)
    mol = gto.loads(dumps)

    return mol


def load_agf2(chkfile):
    ''' Load the AGF2 data from the chkfile.
    '''

    if mpi_helper.rank == 0:
        dic = chkutil.load(chkfile, 'agf2')
    else:
        dic = None

    mpi_helper.barrier()
    dic = mpi_helper.bcast_dict(dic)

    if 'gf' in dic:
        gf = dic['gf']
        dic['gf'] = GreensFunction(gf['energy'], gf['coupling'], chempot=gf['chempot'])
    elif 'gfa' in dic:
        gfa, gfb = dic['gfa'], dic['gfb']
        dic['gf'] = (GreensFunction(gfa['energy'], gfa['coupling'], chempot=gfa['chempot']),
                     GreensFunction(gfb['energy'], gfb['coupling'], chempot=gfb['chempot']))
        del (dic['gfa'], dic['gfb'])

    if 'se' in dic:
        se = dic['se']
        dic['se'] = SelfEnergy(se['energy'], se['coupling'], chempot=se['chempot'])
    elif 'sea' in dic:
        sea, seb = dic['sea'], dic['seb']
        dic['se'] = (SelfEnergy(sea['energy'], sea['coupling'], chempot=sea['chempot']),
                     SelfEnergy(seb['energy'], seb['coupling'], chempot=seb['chempot']))
        del (dic['sea'], dic['seb'])

    if 'ngf' in dic:
        dic['nmom'] = (dic.get('ngf', None), dic.get('nse', None))
        del (dic['ngf'], dic['nse'])

    if 'frozena' in dic:
        dic['frozen'] = (dic['frozena'], dic['frozenb'])
        del (dic['frozena'], dic['frozenb'])

    return load_mol(chkfile), dic


def dump_agf2(agf2, chkfile=None, key='agf2',
              gf=None, se=None, frozen=None, nmom=None,
              mo_energy=None, mo_coeff=None, mo_occ=None):
    ''' Save the AGF2 calculation to a chkfile.
    '''

    if mpi_helper.rank != 0:
        # only dump on root process
        return agf2

    if chkfile is None: chkfile = agf2.chkfile

    if gf is None: gf = agf2.gf
    if se is None: se = agf2.se
    if frozen is None: frozen = agf2.frozen
    if mo_energy is None: mo_energy = agf2.mo_energy
    if mo_coeff is None: mo_coeff = agf2.mo_coeff
    if mo_occ is None: mo_occ = agf2.mo_occ

    if isinstance(gf, (tuple, list)):
        if frozen is not None:
            if isinstance(frozen, int) or isinstance(frozen[0], int):
                frozen = [frozen, frozen]

    if h5py.is_hdf5(chkfile):
        fh5 = h5py.File(chkfile, 'a')
        if key in fh5:
            del (fh5[key])
    else:
        fh5 = h5py.File(chkfile, 'w')

    if 'mol' not in fh5:
        fh5['mol'] = agf2.mol.dumps()

    def store(subkey, val):
        if val is not None:
            fh5[key+'/'+subkey] = val

    store('e_1b', agf2.e_1b)
    store('e_2b', agf2.e_2b)
    store('e_init', agf2.e_init)
    store('converged', agf2.converged)
    store('mo_energy', mo_energy)
    store('mo_coeff', mo_coeff)
    store('mo_occ', mo_occ)
    store('_nmo', agf2._nmo)
    store('_nocc', agf2._nocc)

    if gf is not None:
        if isinstance(gf, (tuple, list)):
            store('gfa/energy', gf[0].energy)
            store('gfa/coupling', gf[0].coupling)
            store('gfa/chempot', gf[0].chempot)
            store('gfb/energy', gf[1].energy)
            store('gfb/coupling', gf[1].coupling)
            store('gfb/chempot', gf[1].chempot)
            if frozen is not None:
                store('frozena', frozen[0])
                store('frozenb', frozen[1])
        else:
            store('gf/energy', gf.energy)
            store('gf/coupling', gf.coupling)
            store('gf/chempot', gf.chempot)
            store('frozen', frozen)

    if se is not None:
        if isinstance(se, (tuple, list)):
            store('sea/energy', se[0].energy)
            store('sea/coupling', se[0].coupling)
            store('sea/chempot', se[0].chempot)
            store('seb/energy', se[1].energy)
            store('seb/coupling', se[1].coupling)
            store('seb/chempot', se[1].chempot)
        else:
            store('se/energy', se.energy)
            store('se/coupling', se.coupling)
            store('se/chempot', se.chempot)

    if getattr(agf2, 'nmom', None) is not None:
        if nmom is None:
            nmom = agf2.nmom
        ngf, nse = nmom
        store('ngf', ngf)
        store('nse', nse)

    fh5.close()

    return agf2
