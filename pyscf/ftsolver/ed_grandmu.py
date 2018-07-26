'''
exact diagonalization solver with grand canonical statistics.
Chong Sun 08/07/17
taking temperature (T) and chemical potential (mu) as input
Chong Sun 01/16/18
'''

import numpy as np
from numpy import linalg as nl
from functools import reduce
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import fci
from pyscf.fci import cistring
from pyscf.fci import direct_uhf
from scipy.optimize import minimize
import datetime
import scipy
import sys
import os


def rdm12s_fted(h1e,g2e,norb,nelec,T,mu=0.0,symm='RHF',Tmin=1.e-3, \
                dcompl=False,**kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    

    Z = 0.
    E = 0.
    RDM1=np.zeros((2, norb, norb), dtype=np.complex128)
    RDM2=np.zeros((3, norb, norb, norb, norb), dtype=np.complex128)

    # non-interacting case
    if np.linalg.norm(g2e[1]) == 0:
        RDM1, E = FD(h1e,norb,nelec,T,mu,symm)
        return RDM1, RDM2, E

    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    if T < Tmin:
        e, v = fcisolver.kernel(h1e, g2e, norb, nelec)
        RDM1, RDM2 = fcisolver.make_rdm12s(v,norb,nelec)
        return np.asarray(RDM1), np.asarray(RDM2), e

    # Calculating E, RDM1, Z
    ews, evs = solve_spectrum(h1e,g2e,norb,fcisolver)
    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            ne = na + nb
            ndim = len(ews[na, nb]) 
            Z += np.sum(np.exp((-ews[na, nb]+mu*ne)/T))
            E += np.sum(np.exp((-ews[na, nb]+mu*ne)/T)*ews[na,nb])
         
            for i in range(ndim):
                dm1, dm2 = fcisolver.make_rdm12s(evs[na,nb][:,i].copy(),norb,(na,nb))
                dm1 = np.asarray(dm1,dtype=np.complex128)
                dm2 = np.asarray(dm2,dtype=np.complex128)
                RDM1 += dm1*np.exp((ne*mu-ews[na, nb][i])/T)
                RDM2 += dm2*np.exp((ne*mu-ews[na, nb][i])/T)

    E    /= Z
    RDM1 /= Z
    RDM2 /= Z

    if not dcompl:
        E = E.real
        RDM1 = RDM1.real
        RDM2 = RDM2.real
    return RDM1, RDM2, E
######################################################################

def rdm12s_fted(h1e,g2e,norb,nelec,T,mu=0.0,symm='RHF',Tmin=1.e-3, \
                dcompl=False,**kwargs):

    if symm is 'RHF':
        from pyscf.fci import direct_spin1 as fcisolver
    elif symm is 'SOC':
        from pyscf.fci import fci_slow_spinless as fcisolver
        dcompl=True
    elif symm is 'UHF':
        from pyscf.fci import direct_uhf as fcisolver
    else:
        from pyscf.fci import direct_spin1 as fcisolver
    

    Z = 0.
    E = 0.
    RDM1=np.zeros((2, norb, norb), dtype=np.complex128)
    RDM2=np.zeros((3, norb, norb, norb, norb), dtype=np.complex128)

    # non-interacting case
    if np.linalg.norm(g2e[1]) == 0:
        RDM1, E = FD(h1e,norb,nelec,T,mu,symm)
        return RDM1, RDM2, E

    if symm != 'UHF' and isinstance(h1e, tuple):
        h1e = h1e[0]
        g2e = g2e[1]

    if T < Tmin:
        e, v = fcisolver.kernel(h1e, g2e, norb, nelec)
        RDM1, RDM2 = fcisolver.make_rdm12s(v,norb,nelec)
        return np.asarray(RDM1), np.asarray(RDM2), e

    # Calculating E, RDM1, Z
    ews, evs = solve_spectrum(h1e,g2e,norb,fcisolver)
    N = 0
    for na in range(0,norb+1):
        for nb in range(0,norb+1):
            ne = na + nb
            ndim = len(ews[na, nb]) 
            Z += np.sum(np.exp((-ews[na, nb]+mu*ne)/T))
            E += np.sum(np.exp((-ews[na, nb]+mu*ne)/T)*ews[na,nb])
            N += ne * np.sum(np.exp((-ews[na, nb]+mu*ne)/T))
         
            for i in range(ndim):
                dm1, dm2 = fcisolver.make_rdm12s(evs[na,nb][:,i].copy(),norb,(na,nb))
                dm1 = np.asarray(dm1,dtype=np.complex128)
                dm2 = np.asarray(dm2,dtype=np.complex128)
                RDM1 += dm1*np.exp((ne*mu-ews[na, nb][i])/T)
                RDM2 += dm2*np.exp((ne*mu-ews[na, nb][i])/T)

    E    /= Z
    N    /= Z
    RDM1 /= Z
    RDM2 /= Z

    #print "%.6f        %.6f"%(1./T, N.real)
    #print "The number of electrons in embedding space: ", N.real

    if not dcompl:
        E = E.real
        RDM1 = RDM1.real
        RDM2 = RDM2.real
    return RDM1, RDM2, E


def solve_spectrum(h1e,h2e,norb,fcisolver):
    EW = np.empty((norb+1,norb+1), dtype=object)
    EV = np.empty((norb+1,norb+1), dtype=object)
    for na in range(0, norb+1):
        for nb in range(0, norb+1):
            ew, ev = diagH(h1e,h2e,norb,(na,nb),fcisolver)
            EW[na, nb] = ew
            EV[na, nb] = ev
    return EW, EV

def solve_mu(h1e,h2e,norb,nelec,fcisolver,T,mu0=0., ews=None):
    if ews is None:
        ews, _ = solve_spectrum(h1e,h2e,norb,fcisolver)

    def Ne_average(mu):
        N = 0
        Z = 0.
        for na in range(0, norb+1):
            for nb in range(0, norb+1):
                ne = na + nb
                N += ne * np.sum(np.exp((-ews[na,nb]+ne*mu)/T))
                Z += np.sum(np.exp((-ews[na,nb]+ne*mu)/T))
        return N/Z
    
    mu_n = minimize(lambda mu:(Ne_average(mu)-nelec)**2, mu0,tol=1e-6).x
    return mu_n

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

    eyemat = np.eye(ndim)
    def hop(c):
        hc = fcisolver.contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    Hmat = []
    for i in range(ndim):
        hc = hop(eyemat[i])
        Hmat.append(hc)


    Hmat = np.asarray(Hmat).T
    ew, ev = nl.eigh(Hmat)
    return ew, ev

def FD(h1e,norb,nelec,T,mu,symm='RHF'):
    #ew, ev = np.linalg.eigh(h1e)
    htot = np.zeros((2*norb, 2*norb))
    htot[:norb,:norb] = h1e[0]
    htot[norb:,norb:] = h1e[1]
            
    ew, ev = np.linalg.eigh(htot)
    def fermi(mu):
        return 1./(1.+np.exp((ew-mu)*beta))
    if T < 1.e-3:
        beta = np.inf
        eocc = np.ones(2*norb)
        eocc[nelec:]*=0.
    else:
        beta = 1./T
        eocc = fermi(mu)

    dm1 = np.asarray(np.dot(ev, np.dot(np.diag(eocc), ev.T.conj())), dtype=np.complex128)
    e = np.sum(ew*eocc)
    DM1 = (dm1[:norb, :norb], dm1[norb:, norb:])

    return DM1, e

if __name__ == '__main__':

    import sys
    norb = 2
    nimp = 2
    nelec = 2
    h1e = np.zeros((norb,norb))
    g2e = np.zeros((norb,norb,norb,norb))
    #T = 0.02
    u = float(sys.argv[1])
    mu= u/2
    #mu= 0.0
    for i in range(norb):
        h1e[i,(i+1)%norb] = -1.
        h1e[i,(i-1)%norb] = -1.
    #h1e[0,-1] = 0.
    #h1e[-1,0] = 0.

    #for i in range(norb):
    #    h1e[i,i] += -u/2.

    for i in range(norb):
        g2e[i,i,i,i] = u

    T = 10
    dm1,_,e1 = rdm12s_fted((h1e,h1e),(g2e*0, g2e, g2e*0),norb,nelec,T,mu, symm='UHF')
    print e1/norb#+u/2.
