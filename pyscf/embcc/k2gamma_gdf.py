import logging
import ctypes
from timeit import default_timer as timer

import numpy as np

import pyscf.pbc
import pyscf.lib
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.tools import k2gamma as k2gamma

try:
    from .util import *
except (SystemError, ImportError):
    import functools
    einsum = functools.partial(np.einsum, optimize=True)

__all__ = ["k2gamma_gdf", "k2gamma_4c2e", "k2gamma_3c2e"]

log = logging.getLogger(__name__)

def k2gamma_gdf(mf, kpts, unfold_j3c=True):
    """kpts is the mesh!"""
    mf_sc = k2gamma.k2gamma(mf, kpts)
    ncells = np.product(kpts)
    assert ncells == (mf_sc.cell.natm // mf.cell.natm)
    # Scale total energy to supercell size
    mf_sc.e_tot = ncells * mf.e_tot
    mf_sc.converged = mf.converged
    # k2gamma GDF
    if isinstance(mf.with_df, pyscf.pbc.df.GDF):
        mf_sc = mf_sc.density_fit()
        # Store integrals in memory
        if unfold_j3c:
            j3c = k2gamma_3c2e(mf.cell, mf.with_df, mf.kpts)
            mf_sc.with_df._cderi = j3c

    return mf_sc

def k2gamma_4c2e(cell, gdf, kpts):
    """Unfold 4c2e integrals from a k-point sampled GDF to the supercell."""
    nao = cell.nao_nr()
    nk = len(kpts)
    scell, phase = k2gamma.get_phase(cell, kpts)
    phase = phase.T.copy()  # (R,k) -> (k,R)
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    eri_sc = np.zeros(4*[nk,nao], dtype=np.complex)
    for i in range(nk):
        for j in range(nk):
            for k in range(nk):
                l = kconserv[i,j,k]
                kijkl = (kpts[i], kpts[j], kpts[k], kpts[l])
                eri_kijkl = gdf.get_eri(kpts=kijkl, compact=False).reshape(4*[nao])
                # Which conjugation order?
                #eri_sc += einsum("ijkl,a,b,c,d->aibjckdl", eri_kijkl, phase[i].conj(), phase[j], phase[k].conj(), phase[l])
                eri_sc += einsum("ijkl,a,b,c,d->aibjckdl", eri_kijkl, phase[i], phase[j].conj(), phase[k], phase[l].conj())

    eri_sc = eri_sc.reshape(4*[nk*nao]) / nk

    maximag = abs(eri_sc.imag).max()
    if maximag > 1e-4:
        log.error("ERROR: Large imaginary part in unfolded 4c2e integrals= %.2e !!!", maximag)
    elif maximag > 1e-8:
        log.warning("WARNING: Large imaginary part in unfolded 4c2e integrals= %.2e !", maximag)
    return eri_sc.real

def k2gamma_3c2e(cell, gdf, kpts, compact=True, use_ksymm=True, version="C"):
    """Unfold 3c2e integrals from a k-point sampled GDF to the supercell."""
    nao = cell.nao_nr()
    nk = len(kpts)
    if gdf.auxcell is None:
        gdf.build(with_j3c=False)
    naux = gdf.auxcell.nao_nr()
    scell, phase = k2gamma.get_phase(cell, kpts)
    phase = phase.T.copy()  # (R,k) -> (k,R)
    kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0].copy()

    if version == "python":

        if compact:
            ncomp = nk*nao*(nk*nao+1)//2
            j3c_sc = np.zeros((nk, naux, ncomp), dtype=np.complex)
        else:
            j3c_sc = np.zeros((nk, naux, nk*nao, nk*nao), dtype=np.complex)
        log.info("Size of 3c-integrals= %s", memory_string(j3c_sc))

        t0 = timer()
        for ki in range(nk):
            jmax = (ki+1) if use_ksymm else nk
            for kj in range(jmax):
                j3c_ij = load_j3c(cell, gdf, (kpts[ki], kpts[kj]), compact=False)
                j3c_ij = einsum("Lab,i,j->Liajb", j3c_ij, phase[ki], phase[kj].conj()).reshape(naux, nk*nao, nk*nao)
                kl = kconserv[ki,kj]
                if compact:
                    j3c_sc[kl] += pyscf.lib.pack_tril(j3c_ij)
                else:
                    j3c_sc[kl] += j3c_ij

                if use_ksymm and kj < ki:
                    j3c_ij = j3c_ij.transpose(0, 2, 1)
                    kl = kconserv[kj,ki]
                    if compact:
                        j3c_sc[kl] += pyscf.lib.pack_tril(j3c_ij)
                    else:
                        j3c_sc[kl] += j3c_ij
        log.info("Time for unfolding AO basis= %.2g s", (timer()-t0))

        # Rotate auxiliary dimension to yield real integrals
        t0 = timer()
        j3c_sc = einsum("k...,kR->R...", j3c_sc, phase)
        log.info("Time for unfolding auxiliary basis= %.2g s", (timer()-t0))

        maximag = abs(j3c_sc.imag).max()
        log.info("Max imaginary part in unfolded 3c2e integrals= %.2e", maximag)
        if maximag > 1e-4:
            log.error("ERROR: Large imaginary part in unfolded 3c2e integrals!!!")
        elif maximag > 1e-8:
            log.warning("WARNING: Large imaginary part in unfolded 3c2e integrals!")
        j3c_sc = j3c_sc.real

    elif version == "C":
        # Precompute all j3c_kpts
        t0 = timer()
        log.info("Loading k-point sampled 3c-integrals into memory...")
        j3c_kpts = np.zeros((naux, nao, nao, nk, nk), dtype=np.complex)
        for ki in range(nk):
            kjmax = (ki+1) if use_ksymm else nk
            for kj in range(kjmax):
                j3c_kpts[...,ki,kj] = load_j3c(cell, gdf, (kpts[ki], kpts[kj]), compact=False)
                if use_ksymm and kj < ki:
                    # Transpose AO indices
                    j3c_kpts[...,kj,ki] = j3c_kpts[...,ki,kj].transpose(0,2,1).conj()
        log.info("Time to load k-sampled j3c= %.2g s", (timer()-t0))

        if compact:
            ncomp = nk*nao*(nk*nao+1)//2
            j3c_sc = np.zeros((nk, naux, ncomp))
        else:
            j3c_sc = np.zeros((nk, naux, nk*nao, nk*nao))
        log.info("Size of 3c-integrals= %s", memory_string(j3c_sc))

        t0 = timer()
        libpbc = pyscf.lib.load_library("libpbc")
        ierr = libpbc.j3c_k2gamma(
                ctypes.c_int64(nk),
                ctypes.c_int64(nao),
                ctypes.c_int64(naux),
                kconserv.ctypes.data_as(ctypes.c_void_p),
                phase.ctypes.data_as(ctypes.c_void_p),
                j3c_kpts.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_bool(compact),
                # Out
                j3c_sc.ctypes.data_as(ctypes.c_void_p),
                )
        assert (ierr == 0)
        log.info("Time for unfolding in C= %.2g s", (timer()-t0))

    j3c_sc = j3c_sc / np.sqrt(nk)

    if compact:
        j3c_sc = j3c_sc.reshape(nk*naux, ncomp)
    else:
        j3c_sc = j3c_sc.reshape(nk*naux, nk*nao, nk*nao)
        assert (np.allclose(j3c_sc, j3c_sc.transpose(0,2,1)))

    return j3c_sc

def load_j3c(cell, gdf, kij, compact=False):
    """Load 3c-integrals into memory"""
    nao = cell.nao_nr()
    if gdf.auxcell is None:
        gdf.build(with_j3c=False)
    naux = gdf.auxcell.nao_nr()
    if compact:
        # Untested code, raise error to be on safe side
        raise NotImplementedError
        ncomp = nao*(nao+1)//2
        j3c_kij = np.zeros((naux, ncomp), dtype=np.complex)
    else:
        j3c_kij = np.zeros((naux, nao, nao), dtype=np.complex)

    blkstart = 0
    for lr, li, sign in gdf.sr_loop(kij, compact=compact):
        assert (sign == 1)
        blksize = lr.shape[0]
        blk = np.s_[blkstart:blkstart+blksize]
        if compact:
            j3c_kij[blk] += (lr+1j*li)#.reshape(blksize, ncomp)
        else:
            j3c_kij[blk] += (lr+1j*li).reshape(blksize, nao, nao)
        blkstart += blksize

    assert (blkstart <= naux)

    return j3c_kij


if __name__ == "__main__":

    from pyscf.pbc import gto, scf
    from pyscf.pbc import df, tools

    logging.basicConfig(filename="k2gamma.log", level=logging.DEBUG)
    log = logging.getLogger("k2gamma.log")

    cell = gto.Cell()
    cell.atom = '''
    O  0.  0.  0.
    Ti  1.  0.  1.
    '''
    #cell.basis = '6-31g'
    cell.basis = "gth-dzvp-molopt-sr"
    cell.pseudo = "gth-pade"
    cell.a = 2.0*np.eye(3)
    cell.verbose = 4
    cell.precision = 1e-6
    cell.build()

    nd = 2
    compact = True
    #compact = False

    for i, k in enumerate(range(2, 20)):
    #for i, k in enumerate(range(4, 20)):
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
        scell, phase = k2gamma.get_phase(cell, kpts)
        j3c = k2gamma_3c2e(cell, mf.with_df, kpts, compact=compact)
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
