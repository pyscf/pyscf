import logging
from timeit import default_timer as timer

import numpy as np

import pyscf.pbc
import pyscf.lib
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import k2gamma as pyscf_k2gamma

from .util import *

#from .gdf_k2gamma import gdf_k2gamma

__all__ = ["k2gamma"]

log = logging.getLogger(__name__)

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

        # TEST
        if False:
            eri_3c = einsum("Lij,Lkl->ijkl", j3c.conj(), j3c)
            n = j3c.shape[-1]
            eri = mf_sc.with_df.get_eri(compact=False).reshape(n,n,n,n)
            log.info("ERI error: norm=%.3e max=%.3e", np.linalg.norm(eri_3c-eri), abs(eri_3c-eri).max())
            log.info("ERI from unfolding:\n%r", eri_3c[0,0,0,:20])
            log.info("ERI exact:\n%r", eri[0,0,0,:20])
            raise SystemExit()

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

    Parameters
    ----------
    cell : pyscf.pbc.gto.Cell
        Cell object of the primitive cell.
    gdf : pyscf.pbc.df.GDF
        Gaussian-density fitting object of the primitive cell.
    kpts : ndarray(N, 3)
        K-point array.
    compact : bool, optional

    Returns
    -------
    j3c : ndarray(M, K, K)
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

    # Bottlenecking unfolding step O(N^4):
    t0 = timer()
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    log.debug("Time to generate momentum conserving k-points: %.3f", timer()-t0)
    t0 = timer()
    # Check that first k-point is Gamma point
    assert np.all(kpts[0] == 0)
    if compact:
        ncomp = nk*nao*(nk*nao+1)//2
        j3c = np.zeros((nk, naux, ncomp), dtype=dtype)
    else:
        j3c = np.zeros((nk, naux, nk*nao, nk*nao), dtype=dtype)

    for i in range(nk):
        for j in range(i+1):
            l = kconserv[i,j,0]
            tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao, nk*nao)
            if compact:
                j3c[l]+= pyscf.lib.pack_tril(tmp)
            else:
                j3c[l] += tmp

            if i != j:
                l = kconserv[j,i,0]
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
    log.debug("Max imaginary element: %.3e", abs(j3c.imag).max())

    return j3c

if __name__ == "__main__":
    from pyscf.pbc import gto, scf
    from pyscf.pbc import df, tools

    cell = gto.Cell()
    cell.atom = '''
    He  0.  0.  0.
    He  1.  0.  1.
    '''
    #cell.basis = 'gth-dzvp'
    cell.basis = '6-31g'
    #cell.pseudo = 'gth-pade'
    cell.a = 3.0*np.eye(3)
    cell.verbose = 4
    cell.precision = 1e-6
    cell.build()


    for k in range(2, 20):
        kmesh = [1, 1, k]
        #kmesh = [1, k, k]
        #kmesh = [k, k, k]
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

        # Supercell
        t0 = timer()
        scell, phase = pyscf_k2gamma.get_phase(cell, kpts)
        j3c = gdf_k2gamma(cell, mf.with_df, kpts)
        t_pc_eri = timer() - t0
        print("Time for primitive cell ERI: %.6f" % (t_pc_eri))

        nao_sc = scell.nao_nr()
        mf_sc = scf.RHF(scell)
        mf_sc = mf_sc.density_fit()
        t0 = timer()
        mf_sc.with_df.build()
        t_sc_build = timer() - t0
        print("Time for supercell build: %.6f" % (t_sc_build))

        # TEST Correctness
        if True:
        #if False:
            eri_3c = np.einsum('Lpq,Lrs->pqrs', j3c.conj(), j3c)
            eri_sc = mf_sc.with_df.get_eri(compact=False).reshape([nao_sc]*4)
            err = abs(eri_3c - eri_sc).max()
            print("ERROR: %.8e" % err)
            assert err < 1e-8

        with open("j3c-timings.txt", "a") as f:
            f.write("%3d %.5f %.5f %.5f %.5f\n" % (len(kpts), t_pc_build, t_pc_eri, t_pc_build+t_pc_eri, t_sc_build))
