#!/usr/bin/env python
#

import numpy
import pyscf.lib
from pyscf.fci import cistring
from pyscf.fci import direct_spin0
#from pyscf import fci
from pyscf.fci import direct_spin1 as fcisolver
from pyscf.ftsolver import lanczos_grand
from pyscf.ftsolver import ed_canonical
from pyscf.ftsolver.utils import logger as log

def rdm12s_ftfci(h1e,g2e,norb,nelec,T,mu,m=50,Tmin=1e-3,\
                   dcompl=False,symm='RHF',**kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver

    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    if T < Tmin: # only for half-filling
        e, v = fcisolver.kernel(h1e, g2e, norb, nelec)
        RDM1, RDM2 = fcisolver.make_rdm12s(v,norb,nelec)
        return numpy.asarray(RDM1), numpy.asarray(RDM2), e

    # loop over na and nb
    Z = 0.
    E = 0.
    RDM1=numpy.zeros((2, norb, norb), dtype=numpy.complex128)
    RDM2=numpy.zeros((3, norb, norb, norb, norb), dtype=numpy.complex128)
    #for ns in range(1):
    #    rdm1,rdm2,e,z = lanczos_grand.ft_solver(h1e,g2e,fcisolver,norb,nelec,T,mu=0,m=m)   
    #    RDM1 += rdm1
    #    RDM2 += rdm2
    #    E    += e
    #    Z    += z
    
    # calculate the number of states
    nstate = 0.
    
    dN = 0
    for na in range(norb/2-dN, norb/2+dN+1):
        for nb in range(norb/2-dN, norb/2+dN+1):
            nstate += cistring.num_strings(norb,na)*cistring.num_strings(norb,nb)

    for na in range(norb/2-dN, norb/2+dN+1):
        for nb in range(na, norb/2+dN+1):
            ne = (na,nb)
            ntot = na + nb
            nci = cistring.num_strings(norb,na)*cistring.num_strings(norb,nb)
            factor = numpy.exp(0*(ntot-norb)/T)
            if nci < 1e2:
                rdm1, rdm2, e, z = ed_canonical.ftsolver(h1e,g2e,fcisolver,norb,ne,T,mu,symm='UHF') 
                Z += z/nstate*factor
                E += e/nstate*factor
                RDM1 += rdm1/nstate*factor
                RDM2 += rdm2/nstate*factor

            else:
                rdm1, rdm2, e, z = lanczos_grand.ft_solver(h1e,g2e,fcisolver,norb,ne,T,mu)
                Z += z*(nci/nstate)*factor
                E += e*(nci/nstate)*factor
                RDM1 += rdm1*(nci/nstate)*factor
                RDM2 += rdm2*(nci/nstate)*factor
                #if nb > na:
                #    Z += 2*(nci/nstate)*z*factor
                #    E += 2*(nci/nstate)*e*factor
                #    RDM1 += (nci/nstate)*rdm1*factor
                #    RDM1 += (nci/nstate)*permute_rdm1(rdm1)*factor
                #    RDM2 += 2*(nci/nstate)*rdm2*factor
                #else:
                #    Z += (nci/nstate)*z*factor
                #    E += (nci/nstate)*e*factor
                #    RDM1 += (nci/nstate)*rdm1*factor
                #    #RDM1 += (nci/nstate)*permute_rdm1(rdm1)*factor/2.
                #    RDM2 += (nci/nstate)*rdm2*factor
                #    

    E    /= Z
    RDM1 /= Z
    RDM2 /= Z

    if not dcompl:
        E = E.real
        RDM1 = RDM1.real
        RDM2 = RDM2.real
    return RDM1, RDM2, E

def permute_rdm1(dm):
    norb = dm.shape[-1]
    dm_n = numpy.copy(dm)
    #dm_n = numpy.flipud(dm)
    npair = norb/2
    for a in range(2):
        for i in range(npair):
            dm_n[a][[2*i, 2*i+1]] = dm_n[a][[2*i+1,2*i]]
            dm_n[a][:,[2*i, 2*i+1]] = dm_n[a][:,[2*i+1,2*i]]

    return dm
    return dm_n


if __name__ == '__main__':

    from pyscf.fci import direct_uhf as fcisolver
    norb = 12
    nelec = (norb/2,norb/2)
    u = 4.0
    T = 0.05
    mu = 2
    h1e = numpy.zeros((norb, norb))
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.0
        h1e[i,(i-1)%norb] = -1.0
    #h1e[0,-1] = 0.
    #h1e[-1,0] = 0.
    g2e_ = numpy.zeros((norb,)*4)
    for i in range(norb):
        g2e_[i,i,i,i] = u
    h1e = (h1e, h1e)
    g2e = (numpy.zeros((norb,)*4), g2e_, numpy.zeros((norb,)*4))
    rdm1, rdm2, e = rdm12s_ftfci(h1e,g2e,norb,nelec,T,mu,m=200,Tmin=1e-3,\
                   dcompl=False,symm='UHF')

    e0, v = fcisolver.kernel(h1e,g2e,norb,nelec,nroots=1)
    rdm10,rdm20 = fcisolver.make_rdm12s(v,norb,nelec)

    print e/norb - e0/norb
    print numpy.linalg.norm(rdm1-rdm10)
    print numpy.linalg.norm(rdm2-rdm20)
    print rdm1[0][:4,:4]
    print rdm10[0][:4,:4]
