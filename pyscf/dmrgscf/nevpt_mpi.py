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
# Authors: Sheng Guo
#          Qiming Sun <osirpt.sun@gmail.com>
#

import os
import sys
import time
import math
import copy
import subprocess
from functools import reduce
import numpy
import h5py
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf.dmrgscf import dmrg_sym
from pyscf.dmrgscf import dmrgci
from pyscf import ao2mo
from pyscf import gto
from pyscf.mcscf import casci
from pyscf.tools import fcidump
from pyscf.dmrgscf import settings

def writeh2e(h2e,f,tol,shift0 =1,shift1 =1,shift2 =1,shift3 =1):
    for i in range(0,h2e.shape[0]):
        for j in range(0,h2e.shape[1]):
            for k in range(0,h2e.shape[2]):
                for l in range(0,h2e.shape[3]):
                    if (abs(h2e[i,j,k,l]) > tol):
                        #if ( j==k or j == l) :
                        #if (j==k and k==l) :
                        f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h2e[i,j,k,l], i+shift0, j+shift1, k+shift2, l+shift3))


def writeh1e(h1e,f,tol,shift0 =1,shift1 =1):
    for i in range(0,h1e.shape[0]):
        for j in range(0,h1e.shape[1]):
            if (abs(h1e[i,j]) > tol):
                f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h1e[i,j], i+shift0, j+shift1, 0, 0))

def writeh2e_sym(h2e,f,tol,shift0 =1,shift1 =1,shift2 =1,shift3 =1):
    for i in range(0,h2e.shape[0]):
        for j in range(0,i+1):
            for k in range(0,h2e.shape[2]):
                for l in range(0,k+1):
                    if (abs(h2e[i,j,k,l]) > tol and i*h2e.shape[0]+j >= k*h2e.shape[2]+l ):
                        f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h2e[i,j,k,l], i+shift0, j+shift1, k+shift2, l+shift3))

def writeh1e_sym(h1e,f,tol,shift0 =1,shift1 =1):
    for i in range(0,h1e.shape[0]):
        for j in range(0,i+1):
            if (abs(h1e[i,j]) > tol):
                f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h1e[i,j], i+shift0, j+shift1, 0, 0))


def write_chk(mc,root,chkfile):

    t0 = (time.clock(), time.time())
    fh5 = h5py.File(chkfile,'w')

    if mc.fcisolver.nroots > 1:
        mc.mo_coeff,_, mc.mo_energy = mc.canonicalize(mc.mo_coeff,ci=root)


    fh5['mol']        =       mc.mol.dumps()
    fh5['mc/mo']      =       mc.mo_coeff
    fh5['mc/ncore']   =       mc.ncore
    fh5['mc/ncas']    =       mc.ncas
    nvirt = mc.mo_coeff.shape[1] - mc.ncas-mc.ncore
    fh5['mc/nvirt']   =       nvirt
    fh5['mc/nelecas'] =       mc.nelecas
    fh5['mc/root']    =       root
    fh5['mc/orbe']    =       mc.mo_energy
    fh5['mc/nroots']   =       mc.fcisolver.nroots
    fh5['mc/wfnsym']   =       mc.fcisolver.wfnsym
    if hasattr(mc.mo_coeff, 'orbsym'):
        fh5.create_dataset('mc/orbsym',data=mc.mo_coeff.orbsym)

    if hasattr(mc.mo_coeff, 'orbsym') and mc.mol.symmetry:
        orbsym = numpy.asarray(mc.mo_coeff.orbsym)
        pair_irrep = orbsym.reshape(-1,1) ^ orbsym
    else:
        pair_irrep = None

    ncore = mc.ncore
    nocc = mc.ncore + mc.ncas
    mo_core = mc.mo_coeff[:,:mc.ncore]
    mo_cas  = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    mo_virt = mc.mo_coeff[:,mc.ncore+mc.ncas:]
    core_dm = numpy.dot(mo_core,mo_core.T) *2
    core_vhf = mc.get_veff(mc.mol,core_dm)
    h1e_Sr =  reduce(numpy.dot, (mo_virt.T,mc.get_hcore()+core_vhf , mo_cas))
    h1e_Si =  reduce(numpy.dot, (mo_cas.T, mc.get_hcore()+core_vhf , mo_core))
    h1e, e_core = mc.h1e_for_cas()
    if pair_irrep is not None:
        h1e_Sr[pair_irrep[nocc:,ncore:nocc] != 0] = 0
        h1e_Si[pair_irrep[ncore:nocc,:ncore] != 0] = 0
        h1e[pair_irrep[ncore:nocc,ncore:nocc] != 0] = 0
    fh5['h1e_Si']     =       h1e_Si
    fh5['h1e_Sr']     =       h1e_Sr
    fh5['h1e']        =       h1e
    fh5['e_core']     =       e_core

    if mc._scf._eri is None:
        h2e_t = ao2mo.general(mc.mol, (mc.mo_coeff,mo_cas,mo_cas,mo_cas), compact=False)
        h2e_t = h2e_t.reshape(-1,mc.ncas,mc.ncas,mc.ncas)
        if pair_irrep is not None:
            sym_forbid = (pair_irrep[:,ncore:nocc].reshape(-1,1) !=
                          pair_irrep[ncore:nocc,ncore:nocc].ravel()).reshape(h2e_t.shape)
            h2e_t[sym_forbid] = 0
        h2e =h2e_t[mc.ncore:mc.ncore+mc.ncas,:,:,:]
        fh5['h2e'] = h2e

        h2e_Sr =h2e_t[mc.ncore+mc.ncas:,:,:,:]
        fh5['h2e_Sr'] = h2e_Sr

        h2e_Si =numpy.transpose(h2e_t[:mc.ncore,:,:,:], (1,0,2,3))
        fh5['h2e_Si'] = h2e_Si

    else:
        eri = mc._scf._eri
        h2e_t = ao2mo.general(eri, [mc.mo_coeff,mo_cas,mo_cas,mo_cas], compact=False)
        h2e_t = h2e_t.reshape(-1,mc.ncas,mc.ncas,mc.ncas)
        if pair_irrep is not None:
            sym_forbid = (pair_irrep[:,ncore:nocc].reshape(-1,1) !=
                          pair_irrep[ncore:nocc,ncore:nocc].ravel()).reshape(h2e_t.shape)
            h2e_t[sym_forbid] = 0
        h2e =h2e_t[mc.ncore:mc.ncore+mc.ncas,:,:,:]
        fh5['h2e'] = h2e

        h2e_Sr =h2e_t[mc.ncore+mc.ncas:,:,:,:]
        fh5['h2e_Sr'] = h2e_Sr

        h2e_Si =numpy.transpose(h2e_t[:mc.ncore,:,:,:], (1,0,2,3))
        fh5['h2e_Si'] = h2e_Si

    fh5.close()

    logger.timer(mc,'Write MPS NEVPT integral', *t0)

def default_nevpt_schedule(mol, maxM=500, tol=1e-7):
    nevptsolver = dmrgci.DMRGCI(mol, maxM, tol)
    nevptsolver.scheduleSweeps = [0, 4]
    nevptsolver.scheduleMaxMs  = [maxM, maxM]
    nevptsolver.scheduleTols   = [0.0001, tol]
    nevptsolver.scheduleNoises = [0.0001, 0.0]
    nevptsolver.twodot_to_onedot = 4
    nevptsolver.maxIter = 6
    return nevptsolver

def DMRG_COMPRESS_NEVPT(mc, maxM=500, root=0, nevptsolver=None, tol=1e-7,
                        nevpt_integral=None):

    if isinstance(nevpt_integral, str) and h5py.is_hdf5(nevpt_integral):
        nevpt_integral_file = os.path.abspath(nevpt_integral)
        mol = chkfile.load_mol(nevpt_integral_file)

        fh5 = h5py.File(nevpt_integral_file, 'r')
        ncas = fh5['mc/ncas'].value
        ncore = fh5['mc/ncore'].value
        nvirt = fh5['mc/nvirt'].value
        nelecas = fh5['mc/nelecas'].value
        nroots = fh5['mc/nroots'].value
        wfnsym = fh5['mc/wfnsym'].value
        fh5.close()
    else :
        mol = mc.mol
        ncas = mc.ncas
        ncore = mc.ncore
        nvirt = mc.mo_coeff.shape[1] - mc.ncas-mc.ncore
        nelecas = mc.nelecas
        nroots = mc.fcisolver.nroots
        wfnsym = mc.fcisolver.wfnsym
        nevpt_integral_file = None

    if nevptsolver is None:
        nevptsolver = default_nevpt_schedule(mol, maxM, tol)
        #nevptsolver.__dict__.update(mc.fcisolver.__dict__)
        nevptsolver.wfnsym = wfnsym
        nevptsolver.block_extra_keyword = mc.fcisolver.block_extra_keyword
    nevptsolver.nroots = nroots
    nevptsolver.executable = settings.BLOCKEXE_COMPRESS_NEVPT
    if nevptsolver.executable == getattr(mc.fcisolver, 'executable', None):
        logger.warn(mc, 'DMRG executable file for nevptsolver %s is the same '
                    'to the executable file for DMRG solver %s. If they are '
                    'both compiled by MPI compilers, they may cause error or '
                    'random results in DMRG-NEVPT calculation.')

    nevpt_scratch = os.path.abspath(nevptsolver.scratchDirectory)
    dmrg_scratch = os.path.abspath(mc.fcisolver.scratchDirectory)

    # Integrals are not given by the kwarg nevpt_integral
    if nevpt_integral_file is None:
        nevpt_integral_file = os.path.join(nevpt_scratch, 'nevpt_perturb_integral')
        write_chk(mc, root, nevpt_integral_file)

    conf = dmrgci.writeDMRGConfFile(nevptsolver, nelecas, False, with_2pdm=False,
                                    extraline=['fullrestart','nevpt_state_num %d'%root])
    with open(conf, 'r') as f:
        block_conf = f.readlines()
        block_conf = [l for l in block_conf if 'prefix' not in l]
        block_conf = ''.join(block_conf)

    with h5py.File(nevpt_integral_file) as fh5:
        if 'dmrg.conf' in fh5:
            del(fh5['dmrg.conf'])
        fh5['dmrg.conf'] = block_conf

    if nevptsolver.verbose >= logger.DEBUG1:
        logger.debug1(nevptsolver, 'Block Input conf')
        logger.debug1(nevptsolver, block_conf)

    t0 = (time.clock(), time.time())

    # function nevpt_integral_mpi is called in this cmd
    cmd = ' '.join((nevptsolver.mpiprefix,
                    os.path.realpath(os.path.join(__file__, '..', 'nevpt_mpi.py')),
                    nevpt_integral_file,
                    nevptsolver.executable,
                    dmrg_scratch, nevpt_scratch))
    logger.debug(nevptsolver, 'DMRG_COMPRESS_NEVPT cmd %s', cmd)

    try:
        output = subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        logger.error(nevptsolver, cmd)
        raise err

    if nevptsolver.verbose >= logger.DEBUG1:
        logger.debug1(nevptsolver, open(os.path.join(nevpt_scratch, '0', 'dmrg.out')).read())

    perturb_file = os.path.join(nevpt_scratch, '0', 'Perturbation_%d'%root)
    fh5 = h5py.File(perturb_file, 'r')
    Vi_e  =  fh5['Vi/energy'].value
    Vr_e  =  fh5['Vr/energy'].value
    fh5.close()
    logger.note(nevptsolver,'Nevpt Energy:')
    logger.note(nevptsolver,'Sr Subspace:  E = %.14f'%( Vr_e))
    logger.note(nevptsolver,'Si Subspace:  E = %.14f'%( Vi_e))

    logger.timer(nevptsolver,'MPS NEVPT calculation time', *t0)
    return perturb_file


def nevpt_integral_mpi(mc_chkfile, blockfile, dmrg_scratch, nevpt_scratch):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    mc_chkfile = os.path.abspath(mc_chkfile)
    dmrg_scratch = os.path.abspath(dmrg_scratch)
    nevpt_scratch = os.path.abspath(os.path.join(nevpt_scratch, str(rank)))

    nevpt_inp = os.path.join(nevpt_scratch, 'dmrg.conf')
    nevpt_out = os.path.join(nevpt_scratch, 'dmrg.out')
    if not os.path.exists(nevpt_scratch):
        os.makedirs(os.path.join(nevpt_scratch, 'node0'))

    ncas, partial_core, partial_virt = _write_integral_file(mc_chkfile, nevpt_scratch, comm)

    nevpt_conf = _load(mc_chkfile, 'dmrg.conf', comm)
    with open(nevpt_inp, 'w') as f:
        f.write(nevpt_conf)
        f.write('restart_mps_nevpt %d %d %d \n'%(ncas, partial_core, partial_virt))

    _distribute_dmrg_files(dmrg_scratch, nevpt_scratch, comm)

    root = _load(mc_chkfile, 'mc/root', comm)

    env = copy.copy(os.environ)
    for k in env.keys():
        if 'MPI' in k or 'SLURM' in k:
# remove PBS and SLURM environments to prevent Block running in MPI mode
            del(env[k])

    p = subprocess.Popen(['%s %s > %s'%(blockfile,nevpt_inp,nevpt_out)],
                         env=env, shell=True, cwd=nevpt_scratch)
    p.wait()

    f = open(os.path.join(nevpt_scratch, 'node0', 'Va_%d'%root), 'r')
    Vr_energy = float(f.readline())
    Vr_norm = float(f.readline())
    f.close()

    f = open(os.path.join(nevpt_scratch, 'node0', 'Vi_%d'%root), 'r')
    Vi_energy = float(f.readline())
    Vi_norm = float(f.readline())
    f.close()

    #Vr_total = 0.0
    #Vi_total = 0.0
    Vi_total_e = comm.reduce(Vi_energy,root=0)
    Vi_total_norm = comm.reduce(Vi_norm,root=0)
    Vr_total_e = comm.reduce(Vr_energy,root=0)
    Vr_total_norm = comm.reduce(Vr_norm,root=0)
    #comm.Reduce(Vi_energy,Vi_total,op=MPI.SUM, root=0)
    if rank == 0:

        fh5 = h5py.File(os.path.join(nevpt_scratch, 'Perturbation_%d'%root), 'w')
        fh5['Vi/energy']      =    Vi_total_e
        fh5['Vi/norm']        =    Vi_total_norm
        fh5['Vr/energy']      =    Vr_total_e
        fh5['Vr/norm']        =    Vr_total_norm
        fh5.close()
        #return (sum(Vi_total), sum(Vr_total))
        #print 'Vi total', sum(Vi_total)

    #comm.Reduce(Vr_energy, Vr_total, op=MPI.SUM, root=0)
#    if rank == 0:
#        print 'Vr total', Vr_total
#        print 'Vr total', sum(Vr_total)


def _load(chkfile, key, comm):
    rank = comm.Get_rank()
    if rank == 0:
        with h5py.File(chkfile, 'r') as fh5:
            return comm.bcast(fh5[key].value)
    else:
        return comm.bcast(None)

def _write_integral_file(mc_chkfile, nevpt_scratch, comm):
    mpi_size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        fh5 = h5py.File(mc_chkfile, 'r')
        def load(key):
            if key in fh5:
                return comm.bcast(fh5[key].value)
            else:
                return comm.bcast([])
    else:
        def load(key):
            return comm.bcast(None)

    mol       = gto.loads(load('mol'))
    ncore     = load('mc/ncore')
    ncas      = load('mc/ncas')
    nvirt     = load('mc/nvirt')
    orbe      = load('mc/orbe')
    orbsym    = list(load('mc/orbsym'))
    nelecas   = load('mc/nelecas')
    h1e_Si    = load('h1e_Si')
    h1e_Sr    = load('h1e_Sr')
    h1e       = load('h1e')
    e_core    = load('e_core')
    h2e       = load('h2e')
    h2e_Si    = load('h2e_Si')
    h2e_Sr    = load('h2e_Sr')

    if rank == 0:
        fh5.close()

    if mol.symmetry and len(orbsym) > 0:
        orbsym = orbsym[ncore:ncore+ncas] + orbsym[:ncore] + orbsym[ncore+ncas:]
        orbsym = dmrg_sym.convert_orbsym(mol.groupname, orbsym)
    else:
        orbsym = [1] * (ncore+ncas+nvirt)

    partial_size = int(math.floor((ncore+nvirt)/float(mpi_size)))
    num_of_orb_begin = min(rank*partial_size, ncore+nvirt)
    num_of_orb_end = min((rank+1)*partial_size, ncore+nvirt)
    #Adjust the distrubution the non-active orbitals to make sure one processor has at most one more orbital than average.
    if rank < (ncore+nvirt - partial_size*mpi_size):
        num_of_orb_begin += rank
        num_of_orb_end += rank + 1
    else :
        num_of_orb_begin += ncore+nvirt - partial_size*mpi_size
        num_of_orb_end += ncore+nvirt - partial_size*mpi_size

    if num_of_orb_begin < ncore:
        if num_of_orb_end < ncore:
            h1e_Si = h1e_Si[:,num_of_orb_begin:num_of_orb_end]
            h2e_Si = h2e_Si[:,num_of_orb_begin:num_of_orb_end,:,:]
            h1e_Sr = []
            h2e_Sr = []
       # elif num_of_orb_end > ncore + nvirt :
       #     h1e_Si = h1e_Si[:,num_of_orb_begin:]
       #     h2e_Si = h2e_Si[:,num_of_orb_begin:,:,:]
       #     #h2e_Sr = []
       #     orbsym = orbsym[:ncas] + orbsym[num_of_orb_begin:]
       #     norb = ncas + ncore + nvirt - num_of_orb_begin
        else :
            h1e_Si = h1e_Si[:,num_of_orb_begin:]
            h2e_Si = h2e_Si[:,num_of_orb_begin:,:,:]
            h1e_Sr = h1e_Sr[:num_of_orb_end - ncore,:]
            h2e_Sr = h2e_Sr[:num_of_orb_end - ncore,:,:,:]
    elif num_of_orb_begin < ncore + nvirt :
        if num_of_orb_end <= ncore + nvirt:
            h1e_Si = []
            h2e_Si = []
            h1e_Sr = h1e_Sr[num_of_orb_begin - ncore:num_of_orb_end - ncore,:]
            h2e_Sr = h2e_Sr[num_of_orb_begin - ncore:num_of_orb_end - ncore,:,:,:]
    #    else :
    #        h1e_Si = []
    #        h2e_Si = []
    #        h1e_Sr = h1e_Sr[num_of_orb_begin - ncore:,:]
    #        h2e_Sr = h2e_Sr[num_of_orb_begin - ncore:,:,:,:]
    #        orbsym = orbsym[:ncas] + orbsym[ncas+num_of_orb_begin: ]
    #        norb = ncas + ncore + nvirt - num_of_orb_begin
    else :
        raise RuntimeError('No job for this processor.  It may block MPI.COMM_WORLD.barrier')


    norb = ncas + num_of_orb_end - num_of_orb_begin
    orbsym = orbsym[:ncas] + orbsym[ncas + num_of_orb_begin:ncas + num_of_orb_end]

    if num_of_orb_begin >= ncore:
        partial_core = 0
        partial_virt = num_of_orb_end - num_of_orb_begin
    else:
        if num_of_orb_end >= ncore:
            partial_core = ncore -num_of_orb_begin
            partial_virt = num_of_orb_end - ncore
        else:
            partial_core = num_of_orb_end -num_of_orb_begin
            partial_virt = 0

    tol = float(1e-15)

    f = open(os.path.join(nevpt_scratch, 'FCIDUMP'), 'w')
    nelec = nelecas[0] + nelecas[1]
    fcidump.write_head(f,norb, nelec, ms=abs(nelecas[0]-nelecas[1]), orbsym=orbsym)
    #h2e in active space
    writeh2e_sym(h2e,f,tol)
    #h1e in active space
    writeh1e_sym(h1e,f,tol)


    orbe =list(orbe[:ncore]) + list(orbe[ncore+ncas:])
    orbe = orbe[num_of_orb_begin:num_of_orb_end]
    for i in range(len(orbe)):
        f.write('% .16f  %4d  %4d  %4d  %4d\n'%(orbe[i],i+1+ncas,i+1+ncas,0,0))
    f.write('%.16f  %4d  %4d  %4d  %4d\n'%(e_core,0,0,0,0))
    if (len(h2e_Sr)):
        writeh2e(h2e_Sr,f,tol, shift0 = ncas + partial_core+1)
    f.write('% 4d  %4d  %4d  %4d  %4d\n'%(0,0,0,0,0))
    if (len(h2e_Si)):
        writeh2e(h2e_Si,f,tol, shift1 = ncas+1)
    f.write('% 4d  %4d  %4d  %4d  %4d\n'%(0,0,0,0,0))
    if (len(h1e_Sr)):
        writeh1e(h1e_Sr,f,tol, shift0 = ncas + partial_core+1)
    f.write('% 4d  %4d  %4d  %4d  %4d\n'%(0,0,0,0,0))
    if (len(h1e_Si)):
        writeh1e(h1e_Si,f,tol, shift1 = ncas+1)
    f.write('% 4d  %4d  %4d  %4d  %4d\n'%(0,0,0,0,0))
    f.write('% 4d  %4d  %4d  %4d  %4d\n'%(0,0,0,0,0))
    f.close()

    return ncas, partial_core, partial_virt


def _distribute_dmrg_files(dmrg_scratch_dir, nevpt_scratch_dir, comm):
    rank = comm.Get_rank()
    if rank==0:
        names = set(['dmrg.e', 'statefile.0.tmp', 'RestartReorder.dat', 'wave', 'Rotation'])
        filenames = []
        for fn in os.listdir(os.path.join(dmrg_scratch_dir, 'node0')):
            fnip = fn.split('-')[0]
            if fnip in names:
                filenames.append(fn)
    else:
        filenames = None

    filenames = comm.bcast(filenames, root=0)

    for i in range(len(filenames)):
        if rank == 0:
            with open(os.path.join(dmrg_scratch_dir, 'node0', filenames[i]),'rb') as f:
                data = f.read()
        else:
            data = None
        data = comm.bcast(data, root=0)
        if data==None:
            print('empty file')
        with open(os.path.join(nevpt_scratch_dir, 'node0', filenames[i]),'wb') as f:
            f.write(data)


if __name__ == '__main__':

    nevpt_integral_mpi(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])


