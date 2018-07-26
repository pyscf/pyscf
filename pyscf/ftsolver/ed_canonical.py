
'''
test the results of FT using exact diagonalization. (for small systems lol)
Adeline C. Sun Apr. 22 2016
'''


import numpy as np
from numpy import linalg as nl
from pyscf import fci
from pyscf.fci import cistring
import sys
import os


def ftsolver(h1e,g2e,norb,nelec,T,mu=0,symm='UHF', Tmin=1.e-3,\
                dcompl=False,**kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
        h1e = h1e[0]
        g2e = g2e[1]
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver

    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    nelec = (neleca,nelecb)
    ne = neleca + nelecb
    ndim = na*nb

    if T < Tmin:
        e, v = fcisolver.kernel(h1e,g2e,norb,nelec)
        RDM1, RDM2 = fcisolver.make_rdm12s(v,norb,nelec)
        z = np.exp(-(e-mu*ne)/T)
        return np.asarray(RDM1)*ndim, np.asarray(RDM2)*ndim, e*ndim

    ew, ev = diagH(h1e,g2e,norb,nelec,fcisolver)
    rdm1, rdm2 = [], []
    RDM1 = np.zeros((2, norb, norb))
    RDM2 = np.zeros((3, norb, norb, norb, norb))
    
    Z = np.sum(np.exp(-(ew-mu*ne)/T))
    E = np.sum(np.exp(-(ew-mu*ne)/T)*ew) # not normalized
    #print Z, E

    for i in range(ndim):
        dm1, dm2 = fcisolver.make_rdm12s(ev[:,i].copy(),norb,nelec)
        RDM1 += np.asarray(dm1)*np.exp(-(ew[i]-mu*ne)/T)
        RDM2 += np.asarray(dm2)*np.exp(-(ew[i]-mu*ne)/T)

    if symm is not 'UHF' and len(RDM1.shape)==3:
        RDM1 = np.sum(RDM1, axis=0)
        RDM2 = np.sum(RDM2, axis=0)
    
    RDM1 /= Z
    RDM2 /= Z
    E    /= Z

    return RDM1, RDM2, E

def diagH(h1e,g2e,norb,nelec,fcisolver):
    '''
        exactly diagonalize the hamiltonian.
    '''
    h2e = fcisolver.absorb_h1e(h1e, g2e, norb, nelec, .5)
    if isinstance(nelec, (int, np.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ndim = na*nb
    eyebas = np.eye(ndim)
    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyebas[i])
        Hmat.append(hc)

    Hmat = np.asarray(Hmat).T
    ew, ev = nl.eigh(Hmat)
    return ew, ev

if __name__ == "__main__":

    from pyscf.fci import direct_uhf as fcisolver
    import sys
    norb = 6
    nelec = 6
    h1e = np.zeros((norb,norb))
    g2e = np.zeros((norb,norb,norb,norb))
    #T = 0.02
    u = float(sys.argv[1])
    mu= 2.0
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.
        h1e[i,(i-1)%norb] = -1.
    #for i in range(norb):
    #    h1e[i,i] = -1. * mu
    h1e[0,-1] = 0.
    h1e[-1,0] = 0.
    for i in range(norb):
        g2e[i,i,i,i] = u

    for i in range(norb):
        h1e[i,i] = -u/2.
    dm1,_,e1 = ftsolver((h1e,h1e),(g2e, g2e, g2e), norb,nelec,1,symm='UHF')
    print e1/norb
    #for beta in np.linspace(0.02, 3.0, 20): #[0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ##for beta in [0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    #    T = 1./beta
    #    #dm_fd, e = FD((h1e,h1e+noise),norb,nelec,T,mu=0.0)
    #    dm1,_,e1 = ftsolver((h1e,h1e),(g2e, g2e, g2e), norb,nelec,T,symm='UHF')
    #    print beta, '        ', e1/norb
    
