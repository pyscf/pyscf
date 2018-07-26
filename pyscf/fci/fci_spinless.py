#!/usr/bin/env python
# Spinless Fermions
# Author: Ushnish Ray
#

import numpy
import pyscf.lib
from pyscf.fci import cistring

def contract_1e(f1e, fcivec, norb, nelec):
    
    link_indexa = cistring.gen_linkstr_index_o0(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    
    t1 = numpy.zeros((norb,norb,na),dtype=numpy.complex128)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * fcivec[str0]
    
    fcinew = numpy.dot(f1e.reshape(-1), t1.reshape(-1,na))
    return fcinew.reshape(fcivec.shape)


def contract_2e(eri, fcivec, norb, nelec, opt=None):
    
    link_indexa = cistring.gen_linkstr_index_o0(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)

    ci0 = fcivec
    t1 = numpy.zeros((norb,norb,na),dtype=numpy.complex128)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * ci0[str0]

    t1 = numpy.dot(eri.reshape(norb*norb,-1), t1.reshape(norb*norb,-1))
    t1 = t1.reshape(norb,norb,na)
    fcinew = numpy.zeros_like(ci0,dtype=numpy.complex128)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a,i,str0]

    return fcinew.reshape(fcivec.shape)


def absorb_h1e(h1e, g2e, norb, nelec, fac=1.):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    h2e = g2e.copy().astype(numpy.complex128) 
#   Expect g2e in full form
#   h2e = pyscf.ao2mo.restore(1, eri, norb)
    f1e = h1e - numpy.einsum('jiik->jk', g2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    f1e = f1e.astype(numpy.complex128)
      
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    return h2e * fac

def make_hdiag(h1e, g2e, norb, nelec, opt=None):
    
    link_indexa = cistring.gen_linkstr_index_o0(range(norb), nelec)
    occslista = [tab[:nelec,0] for tab in link_indexa]
   #g2e = pyscf.ao2mo.restore(1, g2e, norb)
    diagj = numpy.einsum('iijj->ij',g2e)
    diagk = numpy.einsum('ijji->ij',g2e)

    hdiag = []
    for aocc in occslista:
        e1 = h1e[aocc,aocc].sum() 
        e2 = diagj[aocc][:,aocc].sum() - diagk[aocc][:,aocc].sum() 
        hdiag.append(e1 + e2*.5)

    return numpy.array(hdiag)

def kernel(h1e, g2e, norb, nelec):

    na = cistring.num_strings(norb, nelec)
   
    h2e = absorb_h1e(h1e, g2e, norb, nelec, .5)    
          
    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, g2e, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    
    ci0 = numpy.random.random(na)
    ci0 /= numpy.linalg.norm(ci0)

    #e, c = pyscf.lib.davidson(hop, ci0, precond, max_space=100)
    e, c = pyscf.lib.davidson(hop, ci0, precond)
    return e, c


# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)
    #fcivec = fcivec.reshape(na,na)
    rdm1 = numpy.zeros((norb,norb),dtype=numpy.complex128)
    
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[str1].conj(),fcivec[str0])
    for str0, tab in enumerate(link_index):
        for a, i, str1, sign in link_index[str0]:
            rdm1[a,i] += sign * numpy.dot(fcivec[:,str1].conj(),fcivec[:,str0])
    return rdm1

# dm_pq,rs = <|p^+ q r^+ s|>
def make_rdm12(fcivec, norb, nelec, opt=None):
    link_index = cistring.gen_linkstr_index(range(norb), nelec)
    na = cistring.num_strings(norb, nelec)

    rdm1 = numpy.zeros((norb,norb),dtype=numpy.complex128)
    rdm2 = numpy.zeros((norb,norb,norb,norb), dtype=numpy.complex128)
    t1 = numpy.zeros((na,norb,norb),dtype=numpy.complex128)
    for str0, tab in enumerate(link_index):     
        for a, i, str1, sign in link_index[str0]:
            t1[str1,i,a] += sign * fcivec[str0]

    rdm1 += numpy.einsum('m,mij->ij', fcivec.conj(), t1)
    #i^+ j|0> => <0|j^+ i, so swap i and j
    rdm2 += numpy.einsum('mij,mkl->jikl', t1.conj(), t1)
    
    return reorder_rdm(rdm1, rdm2)
    
def reorder_rdm(rdm1, rdm2):
    '''reorder from rdm2(pq,rs) = <E^p_q E^r_s> to rdm2(pq,rs) = <e^{pr}_{qs}>.
    Although the "reoredered rdm2" is still in Mulliken order (rdm2[e1,e1,e2,e2]),
    it is the right 2e DM (dotting it with int2e gives the energy of 2e parts)
    '''
    nmo = rdm1.shape[0]
    #if inplace:
    rdm2 = rdm2.reshape(nmo,nmo,nmo,nmo)
    #else:
    #    rdm2 = rdm2.copy().reshape(nmo,nmo,nmo,nmo)
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1
    return rdm1, rdm2


if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo

    norb = 10
    nelec = 5
    h1e = numpy.zeros([norb,norb]) 
    for i in range(0,norb):
       h1e[i,i] = 0.0
       h1e[i,(i+1)%norb] = -1.0
       h1e[i,(i-1)%norb] = -1.0

    eri = numpy.zeros([norb,norb,norb,norb])
#   for i in range(0, norb):
#       eri[i,i,i,i]=4.0
    for i in range(0,norb):
        j = (i+1)%norb
        eri[i,j,i,j] = 1.0

    e1,c = kernel(h1e, eri, norb, nelec)
   
    print "xxxxxxxxxxxxxxxxxxx" 
    print e1*2./norb
#    print c


