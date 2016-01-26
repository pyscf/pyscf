#!/usr/bin/env python
import time
import numpy
from pyscf.mrpt.nevpt2 import sc_nevpt
from pyscf.dmrgscf.dmrg_sym import *
import pyscf.lib.logger as logger
import pyscf.tools
from pyscf import ao2mo
from pyscf import mcscf
import h5py

def writeh2e(h2e,f,tol,shift0 =1,shift1 =1,shift2 =1,shift3 =1):
    for i in xrange(0,h2e.shape[0]):
        for j in xrange(0,h2e.shape[1]):
            for k in xrange(0,h2e.shape[2]):
                for l in xrange(0,h2e.shape[3]):
                    if (abs(h2e[i,j,k,l]) > tol):
                        #if ( j==k or j == l) :
                        #if (j==k and k==l) :
                        f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h2e[i,j,k,l], i+shift0, j+shift1, k+shift2, l+shift3))


def writeh1e(h1e,f,tol,shift0 =1,shift1 =1):
    for i in xrange(0,h1e.shape[0]):
        for j in xrange(0,h1e.shape[1]):
            if (abs(h1e[i,j]) > tol):
                f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h1e[i,j], i+shift0, j+shift1, 0, 0))

def writeh2e_sym(h2e,f,tol,shift0 =1,shift1 =1,shift2 =1,shift3 =1):
    for i in xrange(0,h2e.shape[0]):
        for j in xrange(0,i+1):
            for k in xrange(0,h2e.shape[2]):
                for l in xrange(0,k+1):
                    if (abs(h2e[i,j,k,l]) > tol and i*h2e.shape[0]+j >= k*h2e.shape[2]+l ):
                        f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h2e[i,j,k,l], i+shift0, j+shift1, k+shift2, l+shift3))

def writeh1e_sym(h1e,f,tol,shift0 =1,shift1 =1):
    for i in xrange(0,h1e.shape[0]):
        for j in xrange(0,i+1):
            if (abs(h1e[i,j]) > tol):
                f.write('% .16f  %4d  %4d  %4d  %4d\n'%(h1e[i,j], i+shift0, j+shift1, 0, 0))


def write_chk(mc,root,chkfile):

    t0 = (time.clock(), time.time())
    fh5 = h5py.File(chkfile,'w')

    if mc.fcisolver.nroots > 1:
        mc.mo_coeff,_, mc.mo_energy = mc.canonicalize(mc.mo_coeff,ci=root)


    fh5['mol']        =       format(mc.mol.pack())
    fh5['mc/mo']      =       mc.mo_coeff 
    fh5['mc/ncore']   =       mc.ncore    
    fh5['mc/ncas']    =       mc.ncas     
    nvirt = mc.mo_coeff.shape[1] - mc.ncas-mc.ncore
    fh5['mc/nvirt']   =       nvirt    
    fh5['mc/nelecas'] =       mc.nelecas 
    fh5['mc/root']    =       root
    fh5['mc/orbe']    =       mc.mo_energy
    fh5['mc/nroots']   =       mc.fcisolver.nroots
    if hasattr(mc, 'orbsym'):
        fh5.create_dataset('mc/orbsym',data=mc.orbsym)
    else :
        fh5.create_dataset('mc/orbsym',data=[])

    mo_core = mc.mo_coeff[:,:mc.ncore]
    mo_cas  = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    mo_virt = mc.mo_coeff[:,mc.ncore+mc.ncas:]
    core_dm = numpy.dot(mo_core,mo_core.T) *2
    core_vhf = mc.get_veff(mc.mol,core_dm)
    h1e_Sr =  reduce(numpy.dot, (mo_virt.T,mc.get_hcore()+core_vhf , mo_cas))
    h1e_Si =  reduce(numpy.dot, (mo_cas.T, mc.get_hcore()+core_vhf , mo_core))
    fh5['h1e_Si']     =       h1e_Si   
    fh5['h1e_Sr']     =       h1e_Sr   
    h1e = mc.h1e_for_cas()
    fh5['h1e']        =       h1e[0]

    if mc._scf._eri is None:
        h2e_t = ao2mo.outcore.general_iofree(mc.mol, (mc.mo_coeff,mo_cas,mo_cas,mo_cas),compact=False).reshape(-1,mc.ncas,mc.ncas,mc.ncas) 
        h2e =h2e_t[mc.ncore:mc.ncore+mc.ncas,:,:,:]
        fh5['h2e'] = h2e

        h2e_Sr =h2e_t[mc.ncore+mc.ncas:,:,:,:]
        fh5['h2e_Sr'] = h2e_Sr

        h2e_Si =numpy.transpose(h2e_t[:mc.ncore,:,:,:], (1,0,2,3))
        fh5['h2e_Si'] = h2e_Si

    else:
        eri = mc._scf._eri
        h2e_t = ao2mo.incore.general(eri,[mc.mo_coeff,mo_cas,mo_cas,mo_cas],compact=False).reshape(-1,mc.ncas,mc.ncas,mc.ncas)
        h2e =h2e_t[mc.ncore:mc.ncore+mc.ncas,:,:,:]
        fh5['h2e'] = h2e

        h2e_Sr =h2e_t[mc.ncore+mc.ncas:,:,:,:]
        fh5['h2e_Sr'] = h2e_Sr

        h2e_Si =numpy.transpose(h2e_t[:mc.ncore,:,:,:], (1,0,2,3))
        fh5['h2e_Si'] = h2e_Si


    fh5.close()

    logger.timer(mc,'Write MPS NEVPT integral', *t0)

def nevpt_integral_mpi(mc_chkfile,blockfile,dmrginp,dmrgout,scratch):

    from pyscf import fci
    from mpi4py import MPI
    import math
    import os
    from pyscf.scf import _vhf
    from subprocess import call
    from subprocess import check_call


    comm = MPI.COMM_WORLD
    mpi_size = MPI.COMM_WORLD.Get_size()
    rank = comm.Get_rank()


    if rank == 0:
        fh5 = h5py.File(mc_chkfile,'r')

        moldic    =     eval(fh5['mol'].value)
        mol = pyscf.gto.Mole()
        mol.build(False,False,**moldic)
        mo_coeff  =     fh5['mc/mo'].value
        ncore     =     fh5['mc/ncore'].value
        ncas      =     fh5['mc/ncas'].value
        nvirt     =     fh5['mc/nvirt'].value
        orbe      =     fh5['mc/orbe'].value
        root      =     fh5['mc/root'].value
        orbsym    =     list(fh5['mc/orbsym'].value)
        nelecas   =     fh5['mc/nelecas'].value
        h1e_Si    =     fh5['h1e_Si'].value
        h1e_Sr    =     fh5['h1e_Sr'].value
        h1e       =     fh5['h1e'].value
        h2e       =     fh5['h2e'].value
        h2e_Si    =     fh5['h2e_Si'].value
        h2e_Sr    =     fh5['h2e_Sr'].value
        fh5.close()
        headnode = MPI.Get_processor_name()
    else:
        mol = None
        mo_coeff  =  None  
        ncore     =  None  
        ncas      =  None  
        nvirt     =  None  
        orbe      =  None  
        root      =  None  
        orbsym    =  None  
        nelecas   =  None 
        h1e_Si    =  None  
        h1e_Sr    =  None  
        h1e       =  None  
        h2e       =  None  
        h2e_Si    =  None  
        h2e_Sr    =  None  
        headnode  =  None
    comm.barrier()
    mol = comm.bcast(mol,root=0)
    mo_coeff = comm.bcast(mo_coeff,root=0)
    ncas = comm.bcast(ncas,root=0)
    ncore = comm.bcast(ncore,root=0)
    nvirt = comm.bcast(nvirt,root=0)
    root = comm.bcast(root,root=0)
    orbsym = comm.bcast(orbsym,root=0)
    nelecas = comm.bcast(nelecas,root=0)
    orbe = comm.bcast(orbe,root=0)
    h1e_Si = comm.bcast(h1e_Si,root=0)
    h1e_Sr = comm.bcast(h1e_Sr,root=0)
    h1e = comm.bcast(h1e,root=0)
    h2e = comm.bcast(h2e,root=0)
    h2e_Si = comm.bcast(h2e_Si,root=0)
    h2e_Sr = comm.bcast(h2e_Sr,root=0)
    headnode = comm.bcast(headnode,root=0)



    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:ncore+ncas]
    mo_virt = mo_coeff[:,ncore+ncas:]

    nelec = nelecas[0] + nelecas[1]

    if mol.symmetry and len(orbsym):
        orbsym = orbsym[ncore:ncore+ncas] + orbsym[:ncore] + orbsym[ncore+ncas:]
        if mol.groupname.lower() == 'dooh':
            orbsym = [IRREP_MAP['D2h'][i % 10] for i in orbsym]
        elif mol.groupname.lower() == 'cooh':
            orbsym = [IRREP_MAP['C2h'][i % 10] for i in orbsym]
        else:
            orbsym = [IRREP_MAP[mol.groupname][i] for i in orbsym]
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
        print 'No job for this processor'
        return


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

    newscratch = os.path.join('%s/'%scratch,'%d'%(rank))
    if not os.path.exists('%s'%newscratch):
        os.makedirs('%s'%newscratch)
        os.makedirs('%s/node0'%newscratch)
    check_call('cp %s %s/%s'%(dmrginp,newscratch,dmrginp),shell=True)

    f = open('%s/%s'%(newscratch,dmrginp), 'a')
    f.write('restart_mps_nevpt %d %d %d \n'%(ncas,partial_core, partial_virt))
    f.close()

    tol = float(1e-15)

    #from subprocess import Popen
    #from subprocess import PIPE
    #print 'scratch', scratch
    ##p1 = Popen(['cp %s/* %d/'%(scratch, rank)],shell=True,stderr=PIPE)
    #p1 = Popen(['cp','%s/*'%scratch, '%d/'%rank],shell=True,stderr=PIPE)
    #print p1.communicate()
    #p2 = Popen(['cp %s/node0/* %d'%(scratch, rank)],shell=True,stderr=PIPE)
    ##p2 = Popen(['cp','%s/node0/*'%scratch, '%d/'%rank],shell=True,stderr=PIPE)
    #print p2.communicate()
    #import os
    #call('cp %s/* %d/'%(scratch,rank),shell = True,stderr=os.devnull)
    #call('cp %s/node0/* %d/'%(scratch,rank),shell = True,stderr=os.devnull)
    f1 =open(os.devnull,'w')
    if MPI.Get_processor_name() == headnode:
        call('cp %s/* %s/'%(scratch,newscratch),stderr=f1,shell = True)
        call('cp %s/node0/* %s/node0'%(scratch,newscratch),shell = True)
    else:
        call('scp %s:%s/* %s/'%(headnode,scratch,newscratch),stderr=f1,shell = True)
        call('scp %s:%s/node0/* %s/node0'%(headnode,scratch,newscratch),shell = True)
    f1.close()
    f = open('%s/FCIDUMP'%newscratch,'w')

    pyscf.tools.fcidump.write_head(f,norb, nelec, ms=abs(nelecas[0]-nelecas[1]), orbsym=orbsym)
    #h2e in active space
    writeh2e_sym(h2e,f,tol)
    #h1e in active space
    writeh1e_sym(h1e,f,tol)


    orbe =list(orbe[:ncore]) + list(orbe[ncore+ncas:])
    orbe = orbe[num_of_orb_begin:num_of_orb_end]
    for i in xrange(len(orbe)):
        f.write('% .16f  %4d  %4d  %4d  %4d\n'%(orbe[i],i+1+ncas,i+1+ncas,0,0))
    f.write('% 4d  %4d  %4d  %4d  %4d\n'%(0,0,0,0,0))
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


    current_path = os.getcwd()
    os.chdir('%s'%newscratch)

    check_call('%s %s > %s'%(blockfile,dmrginp,dmrgout), shell=True)
    f = open('node0/Va_%d'%root,'r')
    Vr_energy = float(f.readline())
    Vr_norm = float(f.readline())
    f.close()
    f = open('node0/Vi_%d'%root,'r')
    Vi_energy = float(f.readline())
    Vi_norm = float(f.readline())
    f.close()
    comm.barrier()
    #Vr_total = 0.0
    #Vi_total = 0.0
    Vi_total_e = comm.gather(Vi_energy,root=0)
    Vi_total_norm = comm.gather(Vi_norm,root=0)
    Vr_total_e = comm.gather(Vr_energy,root=0)
    Vr_total_norm = comm.gather(Vr_norm,root=0)
    #comm.Reduce(Vi_energy,Vi_total,op=MPI.SUM, root=0)
    os.chdir('%s'%current_path)
    if rank == 0:

        fh5 = h5py.File('Perturbation_%d'%root,'w')
        fh5['Vi/energy']      =    sum(Vi_total_e)
        fh5['Vi/norm']        =    sum(Vi_total_norm)
        fh5['Vr/energy']      =    sum(Vr_total_e)
        fh5['Vr/norm']        =    sum(Vr_total_norm)
        fh5.close()
        #return (sum(Vi_total), sum(Vr_total))
        #print 'Vi total', sum(Vi_total)

    #comm.Reduce(Vr_energy, Vr_total, op=MPI.SUM, root=0)
#    if rank == 0:
#        print 'Vr total', Vr_total
#        print 'Vr total', sum(Vr_total)



if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf import fci
    from pyscf import mcscf

    import sys
    nevpt_integral_mpi(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])


