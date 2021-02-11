import logging
from timeit import default_timer as timer

import numpy as np

import pyscf.pbc
import pyscf.lib
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import k2gamma as pyscf_k2gamma

try:
    from .util import *
except (SystemError, ImportError):
    import functools
    einsum = functools.partial(np.einsum, optimize=True)

__all__ = ["k2gamma"]

log = logging.getLogger(__name__)

TEST_MODULE = False

def k2gamma(mf, kpts):
    """kpts is the mesh!"""
    mf_sc = pyscf_k2gamma.k2gamma(mf, kpts, tol_orth=1e-4)
    ncells = np.product(kpts)
    assert ncells == (mf_sc.cell.natm // mf.cell.natm)
    # Scale total energy to supercell size
    mf_sc.e_tot = ncells * mf.e_tot
    mf_sc.converged = mf.converged
    # k2gamma GDF
    if isinstance(mf.with_df, pyscf.pbc.df.GDF):
        mf_sc = mf_sc.density_fit()
        j3c = gdf_k2gamma(mf.cell, mf.with_df, mf.kpts)

        if TEST_MODULE:
            eri_3c = einsum("Lij,Lkl->ijkl", j3c.conj(), j3c)
            n = j3c.shape[-1]
            eri = mf_sc.with_df.get_eri(compact=False).reshape(n,n,n,n)
            log.info("ERI error: norm=%.3e max=%.3e", np.linalg.norm(eri_3c-eri), abs(eri_3c-eri).max())
            log.info("ERI from unfolding:\n%r", eri_3c[0,0,0,:20])
            log.info("ERI exact:\n%r", eri[0,0,0,:20])

        mf_sc.with_df._cderi = j3c

    return mf_sc

def kptidx_bz(cell, kpts, kpt):
    """Get k-point index of k-point kpts, translatted into the BZ."""
    kpt_internal = cell.get_scaled_kpts([kpt])[0]
    # Map into BZ
    kpt_internal[kpt_internal>=1] -= 1
    kpt_internal[kpt_internal<0] += 1
    kpt_external = cell.get_abs_kpts([kpt_internal])[0]
    res = kpts_helper.member(kpt_external, kpts)
    if len(res) != 1:
        raise RuntimeError("Error finding k-point %r in %r" % (kpt_external, kpts))
    return res[0]

def gdf_k2gamma(cell, gdf, kpts, compact=True):
    """Unfold k-point sampled three center-integrals (l|ki,qj) into supercell integrals (L|IJ).

    Sizes:
        K:      Number of k-points
        L:      Size of auxiliary basis in primitive cell
        N:      Number of AOs in primitive cell.
        K*L:    Size of auxiliary basis in supercell
        K*N:    Number of AOs in supercell.

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Cell object of the primitive cell.
    gdf : pyscf.pbc.df.GDF
        Gaussian-density fitting object of the primitive cell.
    kpts : ndarray(K, 3)
        K-point array.
    compact : bool, optional
        Store only lower-triangular part of unfolded supercell AOs.

    Returns
    -------
    j3c : ndarray(K*L, K*N, K*N) or ndarray(K*L, (K*N)*(K*N+1)/2)
        Three-center integrals of the supercell.
    """

    nao = cell.nao_nr()
    naux = gdf.auxcell.nao_nr()
    nk = len(kpts)
    dtype = np.complex

    # Evaluate k-point sampled primitive three-center integrals
    t0 = timer()
    j3c_k = np.zeros((naux, nk, nao, nk, nao), dtype=dtype)
    for i, ki in enumerate(kpts):
        for j, kj in enumerate(kpts[:i+1]):
            for lr, li, sign in gdf.sr_loop([ki, kj], compact=False):
                assert (sign == 1)
                nauxk = lr.shape[0]
                assert nauxk <= naux
                tmp = (lr+1j*li).reshape(nauxk, nao, nao)
                j3c_k[:nauxk,i,:,j,:] += tmp
                if i != j:
                    j3c_k[:nauxk,j,:,i,:] += tmp.transpose([0,2,1]).conj()
    log.debug("Time to evaluate k-point sampled three-center integrals: %.3f", timer()-t0)

    scell, phase = pyscf_k2gamma.get_phase(cell, kpts)

    t0 = timer()
    kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0]
    # Check that first k-point is Gamma point
    assert np.all(kpts[0] == 0)
    if compact:
        ncomp = nk*nao*(nk*nao+1)//2
        j3c = np.zeros((nk, naux, ncomp), dtype=dtype)
    else:
        j3c = np.zeros((nk, naux, nk*nao, nk*nao), dtype=dtype)

    # TEST PHASE MATRIX
#    phase2 = einsum("kR,qS->kRqS", phase, phase.conj())
#    for i in range(nk):
#        for j in range(nk):
#            for i2 in range(nk):
#                if i == i2:
#                    continue
#                for j2 in range(nk):
#                    if j == j2:
#                        continue
#                    diff_max = abs(phase2[:,i,:,j] - phase2[:,i2,:,j2].T.conj()).max()
#                    diff_norm = np.linalg.norm(phase2[:,i,:,j] - phase2[:,i2,:,j2].T.conj())
#                    print("Difference %d %d to %d %d = %e %e" % (i, j, i2, j2, diff_max, diff_norm))
#
    #1/0

    # Bottlenecking unfolding step O(N^4):
    for i in range(nk):
        for j in range(i+1):
            l = kconserv[i,j]
            tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao, nk*nao)
            #tmp2 = einsum("Lab,RS->LRaSb", j3c_k[:,i,:,j], phase2[:,i,:,j]).reshape(naux, nk*nao, nk*nao)
            #assert np.allclose(tmp, tmp2)
            if compact:
                j3c[l]+= pyscf.lib.pack_tril(tmp)
            else:
                j3c[l] += tmp

            if i != j:
                l = kconserv[j,i]
                if compact:
                    j3c[l] += pyscf.lib.pack_tril(tmp.transpose([0,2,1]).conj())
                else:
                    j3c[l] += tmp.transpose([0,2,1]).conj()
    if compact:
        j3c = j3c.reshape(nk*naux, ncomp) / np.sqrt(nk)
    else:
        j3c = j3c.reshape(nk*naux, nk*nao, nk*nao) / np.sqrt(nk)

    log.debug("Memory for 3c-integrals: %.3f GB", j3c.nbytes/1e9)
    log.debug("Time to unfold three-center integrals: %.3f", timer()-t0)
    log.debug("Max imaginary element of j3c: %.3e", abs(j3c.imag).max())

    return j3c

if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    from pyscf.pbc import df, tools

    cell = gto.Cell()
    cell.atom = '''
    He  0.  0.  0.
    He  1.  0.  1.
    '''
    cell.basis = '6-31g'
    cell.a = 3.0*np.eye(3)
    cell.verbose = 4
    cell.precision = 1e-6
    cell.build()

    nd = 1
    compact = True
    #compact = False

    for i, k in enumerate(range(2, 20)):
        if nd == 1:
            kmesh = [1, 1, k]
        elif nd == 2:
            kmesh = [1, k, k]
        elif nd == 3:
            kmesh = [k, k, k]
        kpts = cell.make_kpts(kmesh)
        print("K-points: %r" % kpts)
        nk = len(kpts)
        norb = cell.nao_nr()

        # Primitive cell
        nao = cell.nao_nr()
        mf = scf.KRHF(cell, kpts)
        mf = mf.density_fit()
        t0 = timer()
        mf.with_df.build()
        t_pc_build = timer() - t0
        print("Time for primitive cell build: %.6f" % (t_pc_build))

        # Supercell unfolding
        t0 = timer()
        scell, phase = pyscf_k2gamma.get_phase(cell, kpts)
        j3c = gdf_k2gamma(cell, mf.with_df, kpts, compact=compact)
        t_pc_eri = timer() - t0
        print("Time for primitive cell ERI: %.6f" % (t_pc_eri))

        # Supercell direct
        nao_sc = scell.nao_nr()
        mf_sc = scf.RHF(scell)
        mf_sc = mf_sc.density_fit()
        t0 = timer()
        mf_sc.with_df.build()
        t_sc_build = timer() - t0
        print("Time for supercell build: %.6f" % (t_sc_build))

        # Test correctness
        if True:
        #if False:
            if compact:
                eri_3c = np.dot(j3c.T.conj(), j3c)
            else:
                eri_3c = np.einsum('Lpq,Lrs->pqrs', j3c.conj(), j3c)
            eri_sc = mf_sc.with_df.get_eri(compact=compact)
            if not compact:
                eri_sc = eri_sc.reshape([nao_sc]*4)
            assert (eri_3c.shape == eri_sc.shape)
            err = abs(eri_3c - eri_sc).max()
            print("MAX. ERROR: %.8e" % err)
            assert err < 1e-8

        # Write timings to file
        if i == 0:
            with open("k2gamma-timings.txt", "w") as f:
                f.write("Nkpts   GDF(prim)   j3c-unfolding   build+unfolding   GDF(SC)\n")

        with open("k2gamma-timings.txt", "a") as f:
            f.write("%3d   %.5f   %.5f   %.5f   %.5f\n" % (len(kpts), t_pc_build, t_pc_eri, t_pc_build+t_pc_eri, t_sc_build))
