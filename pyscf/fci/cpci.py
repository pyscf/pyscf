from pyscf.fci import direct_spin1
from pyscf.fci import rdm as fci_rdm

import scipy.sparse.linalg as sla
import numpy as np

def solve(h1e, eri, ECI, ci, dHc, norb, nelec,
        dci0=None, tol=1e-9, max_cycle=20):
    '''
    solve for dc^CI / dR from
    (H^CI - E^CI + 2 c^CI outer c^CI) dc^CI / dR = dHc
    i.e. M dc = dHc

    the solution is c^CI response if dHc = d(E^CI-H^CI) / dR dot c^CI

    h1e: 1e integral
    eri: 2e integral
    ECI: CI energy
    ci:  CI vector
    '''
    lidxa, lidxb = direct_spin1._unpack(norb, nelec, None)
    Na = lidxa.shape[0]
    Nb = lidxb.shape[0]

    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)

    def M(dc):
        '''
        NOTE dc will be a flattened vector with shape Na*Nb
        '''
        cdc = np.dot(ci.reshape(-1), dc)
        Mdc = 2 * (cdc * ci)
        dc = dc.reshape((Na, Nb))
        Mdc += direct_spin1.contract_2e(h2e, dc, norb, nelec, (lidxa,lidxb))
        Mdc -= (ECI * dc)
        return Mdc

    M = sla.LinearOperator((Na*Nb,Na*Nb), M)

    if dci0 is None:
        dci0 = np.zeros(Na*Nb)
    sol, stat = sla.gmres(M, dHc.flatten(), x0=dci0,
            tol=tol, atol=tol*np.linalg.norm(dHc), maxiter=max_cycle)
    # stat == 0 means converged

    return sol.reshape((Na,Nb)), stat

if __name__ == "__main__":
    np.random.seed(1)

    from pyscf import gto, scf, fci, ao2mo
    mol = gto.Mole()
    mol.atom = '''
    O 0 0      0
    H 0 1.0000 0.000
    H 0 -0.5 0.866025404'''
    mol.basis = 'sto-3g'
    mol.build()
    nao = mol.nao
    nelec = mol.nelectron

    mf = scf.RHF(mol)
    mf.kernel()

    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.full(mf._eri, mf.mo_coeff, nao)

    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    E, c = solver.kernel(h1, h2, nao, nelec)

    dh1 = np.random.random(h1.shape)
    dh1 = 1e-4 * dh1 / np.linalg.norm(h1)
    dh1 = (dh1 + dh1.T) / 2
    dh2 = np.random.random(h2.shape)
    dh2 = 1e-4 * dh2 / np.linalg.norm(h2)
    dh2 = (dh2 + dh2.T) / 2

    E1, c1 = solver.kernel(h1+dh1/2, h2+dh2/2, nao, nelec)
    E2, c2 = solver.kernel(h1-dh1/2, h2-dh2/2, nao, nelec)
    num_dc = c1-c2

    def Hc(h1, h2, c):
        h2e = fci.direct_spin1.absorb_h1e(h1, h2, nao, nelec, 0.5)
        return fci.direct_spin1.contract_2e(h2e, c, nao, nelec)

    dHc = (E1-E2) * c - Hc(dh1, dh2, c)
    dc, stat = solve(h1, h2, E, c, dHc, nao, nelec, tol=1e-12)
    assert stat == 0

    # check gmres via (H-E) dc == dHc
    Hdc = Hc(h1, h2, dc) - E*dc
    assert np.linalg.norm(Hdc-dHc) / np.linalg.norm(dHc) < 1e-7

    # check CPCI eq via (H-E) num_dc == dHc
    Hndc = Hc(h1, h2, num_dc) - E*num_dc
    assert np.linalg.norm(Hndc-dHc) < 1e-7

    # check identity (H2-E2) num_dc == (num_dE-num_dH) c1
    H2ndc = Hc(h1-dh1/2, h2-dh2/2, num_dc) - E2*num_dc
    ndHc1 = -Hc(dh1, dh2, c1) + (E1-E2)*c1
    assert np.linalg.norm(H2ndc-ndHc1) < 1e-7

    # check FCI energy only needs H response
    assert abs(E1 - E2 - c.flatten() @ Hc(dh1, dh2, c).flatten()) < 1e-7
