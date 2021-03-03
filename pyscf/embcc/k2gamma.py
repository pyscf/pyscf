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
#if __name__ != "__main__":
#    log = logging.getLogger(__name__)
#else:
#    logging.basicConfig(filename="k2gamma.log", level=logging.DEBUG)
#    log = logging.getLogger("k2gamma.log")



TEST_MODULE = False

# Note that the imaginary part of (L|ij) only contributes to the real part of (ij|kl)
# when squared with itself - a tolerance of 1e-7 means that the real part is only affected by ~1e-14
DEFAULT_IMAG_TOL = 1e-7

def k2gamma(mf, kpts, unfold_j3c=True):
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
        if unfold_j3c:
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

def get_j3c_from_gdf(cell, gdf, kpts, aos=None, use_kpts_sym=True, imag_tol=DEFAULT_IMAG_TOL):
    """
    If abs(imag).max() < imag_tol, only the real part of j3c will be saved
    """
    # Calculate all AOs
    if aos is None:
        nao = cell.nao_nr()
    # Calculate only AOs defined by aos
    else:
        nao = len(np.arange(cell.nao_nr())[aos])

    nk = len(kpts)
    if gdf.auxcell is None:
        gdf.build(with_j3c=False)
    naux = gdf.auxcell.nao_nr()
    dtype = np.complex
    j3c_k = np.zeros((naux, nk, nao, nk, nao), dtype=dtype)

    if use_kpts_sym:
        for i, ki in enumerate(kpts):
            for j, kj in enumerate(kpts[:i+1]):
                for lr, li, sign in gdf.sr_loop([ki, kj], compact=False):
                    assert (sign == 1)
                    nauxk = lr.shape[0]
                    assert nauxk <= naux
                    if aos is not None:
                        lr = lr[:,aos][:,:,aos]
                        li = li[:,aos][:,:,aos]
                    tmp = (lr+1j*li).reshape(nauxk, nao, nao)
                    j3c_k[:nauxk,i,:,j,:] += tmp
                    if i != j:
                        j3c_k[:nauxk,j,:,i,:] += tmp.transpose([0,2,1]).conj()
    else:
        for i, ki in enumerate(kpts):
            for j, kj in enumerate(kpts):
                for lr, li, sign in gdf.sr_loop([ki, kj], compact=False):
                    assert (sign == 1)
                    nauxk = lr.shape[0]
                    assert nauxk <= naux
                    if aos is not None:
                        lr = lr[:,aos][:,:,aos]
                        li = li[:,aos][:,:,aos]
                    j3c_k[:nauxk,i,:,j,:] += (lr+1j*li).reshape(nauxk, nao, nao)

    # Discard imaginary part, if close to zero
    if imag_tol and abs(j3c_k.imag).max() < imag_tol:
        j3c_k = j3c_k.real

    return j3c_k

#def gdf_k2gamma_new(cell, gdf, kpts, compact=True, use_kpts_sym=True, imag_tol=DEFAULT_IMAG_TOL):
#    """Unfold k-point sampled three center-integrals (l|ki,qj) into supercell integrals (L|IJ).
#
#    Sizes:
#        K:      Number of k-points
#        L:      Size of auxiliary basis in primitive cell
#        N:      Number of AOs in primitive cell.
#        K*L:    Size of auxiliary basis in supercell
#        K*N:    Number of AOs in supercell.
#
#    Parameters
#    ----------
#    cell : pyscf.pbc.gto.Cell
#        Cell object of the primitive cell.
#    gdf : pyscf.pbc.df.GDF
#        Gaussian-density fitting object of the primitive cell or 3c-integrals.
#    kpts : ndarray(K, 3)
#        K-point array.
#    compact : bool, optional
#        Store only lower-triangular part of unfolded supercell AOs.
#
#    Returns
#    -------
#    j3c : ndarray(K*L, K*N, K*N) or ndarray(K*L, (K*N)*(K*N+1)/2)
#        Three-center integrals of the supercell.
#    """
#
#    nao = cell.nao_nr()
#    nk = len(kpts)
#    dtype = np.complex
#
#    #if gdf._cderi is not None:
#    if isinstance(gdf._cderi, np.ndarray):
#        j3c_k = gdf._cderi.reshape(-1, nk, nao, nk, nao)
#    elif isinstance(gdf, pyscf.pbc.df.GDF):
#        t0 = timer()
#        j3c_k = get_j3c_from_gdf(cell, gdf, kpts, aos=aos, use_kpts_sym=use_kpts_sym)
#        log.debug("Time to evaluate k-point sampled three-center integrals: %.3f", timer()-t0)
#    else:
#        raise ValueError("Invalid type for gdf: %s" % type(gdf))
#    naux = j3c_k.shape[0]
#    log.debug("dtype of k-point sampled j3c: %r", j3c_k.dtype)
#
#    scell, phase = pyscf_k2gamma.get_phase(cell, kpts)
#
#    t0 = timer()
#    # Check that first k-point is Gamma point
#    assert np.all(kpts[0] == 0)
#    #kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0].copy()
#    kconserv = kpts_helper.get_kconserv(cell, kpts)
#    if compact:
#        ncomp = nk*nao*(nk*nao+1)//2
#        j3c_l = np.zeros((nk, naux, ncomp), dtype=dtype)
#    else:
#        j3c_l = np.zeros((nk, naux, nk*nao, nk*nao), dtype=dtype)
#
#    j3c_r = np.zeros_like(j3c_l)
#
#    # Bottlenecking unfolding step O(N^4):
#    for i in range(nk):
#        for j in range(nk):
#            j3c_l[i] = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao, nk*nao)
#
#    for i in range(nk):
#        for j in range(nk):
#            for k in range(nk):
#                l = kconserv[i,j,k]
#                tmp = j3c_
#
#                j3c_r[i] += einsum("Lab,R,S->LRaSb", j3c_k[:,k,:,l], phase[:,k], phase[:,l].conj()).reshape(naux, nk*nao, nk*nao)
#
#
#
#
#    if use_kpts_sym:
#        for i in range(nk):
#            for j in range(i+1):
#                l = kconserv[i,j]
#                tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao, nk*nao)
#                #tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i].conj(), phase[:,j]).reshape(naux, nk*nao_out, nk*nao_out)
#                if compact:
#                    j3c[l]+= pyscf.lib.pack_tril(tmp)
#                else:
#                    j3c[l] += tmp
#
#                if i != j:
#                    l = kconserv[j,i]
#                    if compact:
#                        j3c[l] += pyscf.lib.pack_tril(tmp.transpose([0,2,1]).conj())
#                    else:
#                        j3c[l] += tmp.transpose([0,2,1]).conj()
#    else:
#        for i in range(nk):
#            for j in range(nk):
#                l = kconserv[i,j]
#                tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao_out, nk*nao_out)
#                if compact:
#                    j3c[l]+= pyscf.lib.pack_tril(tmp)
#                else:
#                    j3c[l] += tmp
#
#    # Discard imaginary part if close to zero
#    if imag_tol:
#        if abs(j3c_l.imag).max() < imag_tol:
#            j3c_l = j3c_l.real
#        if abs(j3c_r.imag).max() < imag_tol:
#            j3c_r = j3c_r.real
#
#    if compact:
#        j3c_l = j3c_l.reshape(nk*naux, ncomp) / np.sqrt(nk)
#        j3c_r = j3c_r.reshape(nk*naux, ncomp) / np.sqrt(nk)
#    else:
#        j3c_l = j3c_l.reshape(nk*naux, nk*nao_out, nk*nao_out) / np.sqrt(nk)
#        j3c_r = j3c_r.reshape(nk*naux, nk*nao_out, nk*nao_out) / np.sqrt(nk)
#
#    log.debug("Memory for 3c-integrals: %.3f GB", 2*j3c_l.nbytes/1e9)
#    log.debug("Time to unfold three-center integrals: %.3f", timer()-t0)
#    log.debug("Max imaginary element of j3c_l= %.3e , j3c_r= %.3e" , abs(j3c_l.imag).max(), abs(j3c_r.imag).max())
#
#    #check
#    #if check_imag:
#    #    im_norm = 0.0
#    #    im_max = 0.0
#    #    j3c_full = pyscf.lib.unpack_tril(j3c) if compact else j3c
#    #    for i in range(nk*nao_out):
#    #        eri = einsum("Lj,Lkl->jkl", j3c_full[:,i].conj(), j3c_full)
#    #        im_max = max(im_max, abs(eri.imag).max())
#    #        im_norm += np.linalg.norm(eri.imag)**2
#    #        log.info("i= %3d norm= %.5g max= %.5g", i, np.sqrt(im_norm), im_max)
#    #    im_norm = np.sqrt(im_norm)
#    #    log.info("Imaginary part of (ab|cd) in AO basis: norm= %.5g max= %.5g", im_norm, im_max)
#
#    return j3c_l, j3c_r


def gdf_k2gamma(cell, gdf, kpts, aos=None, compact=True, use_kpts_sym=True, imag_tol=DEFAULT_IMAG_TOL, check_imag=False):
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
        Gaussian-density fitting object of the primitive cell or 3c-integrals.
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
    nao_out = nao if aos is None else len(np.arange(nao)[aos])
    nk = len(kpts)
    dtype = np.complex

    #if gdf._cderi is not None:
    if isinstance(gdf._cderi, np.ndarray):
        j3c_k = gdf._cderi
    #    elif isinstance(gdf._cderi, str):
    #        j3c_k = np.load(gdf._cderi)
    #    #j3c_k = lib.unpack_tril(j3c_k).reshape(-1, nk, nao, nk, nao)
    #    j3c_k = j3c_k.reshape(-1, nk, nao, nk, nao)
    # Evaluate k-point sampled primitive three-center integrals
    #if isinstance(gdf, np.ndarray):
        j3c_k = j3c_k.reshape(-1, nk, nao, nk, nao)
        if aos is not None:
            j3c_k = j3c_k[:,:,aos][:,:,:,:,aos]
    elif isinstance(gdf, pyscf.pbc.df.GDF):
        t0 = timer()
        j3c_k = get_j3c_from_gdf(cell, gdf, kpts, aos=aos, use_kpts_sym=use_kpts_sym)
        log.debug("Time to evaluate k-point sampled three-center integrals: %.3f", timer()-t0)
    else:
        raise ValueError("Invalid type for gdf: %s" % type(gdf))
    naux = j3c_k.shape[0]
    log.debug("dtype of k-point sampled j3c: %r", j3c_k.dtype)

    scell, phase = pyscf_k2gamma.get_phase(cell, kpts)

    t0 = timer()
    # Check that first k-point is Gamma point
    assert np.all(kpts[0] == 0)
    kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0].copy()
    if compact:
        ncomp = nk*nao_out*(nk*nao_out+1)//2
        j3c = np.zeros((nk, naux, ncomp), dtype=dtype)
    else:
        j3c = np.zeros((nk, naux, nk*nao_out, nk*nao_out), dtype=dtype)

    #fast = True
    fast = False

    # Bottlenecking unfolding step O(N^4):
    if not fast:
        if use_kpts_sym:
            for i in range(nk):
                for j in range(i+1):
                    l = kconserv[i,j]
                    tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao_out, nk*nao_out)
                    #tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i].conj(), phase[:,j]).reshape(naux, nk*nao_out, nk*nao_out)
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
        else:
            for i in range(nk):
                for j in range(nk):
                    l = kconserv[i,j]
                    tmp = einsum("Lab,R,S->LRaSb", j3c_k[:,i,:,j], phase[:,i], phase[:,j].conj()).reshape(naux, nk*nao_out, nk*nao_out)
                    if compact:
                        j3c[l]+= pyscf.lib.pack_tril(tmp)
                    else:
                        j3c[l] += tmp
    else:
        raise NotImplementedError()
        import ctypes
        log.debug("Loading library libbc")
        libpbc = pyscf.lib.load_library("libpbc")
        log.debug("Library loaded")
        log.debug("First element before= %.3e", j3c[0,0,0])
        log.debug("Passed data from Python:")
        log.debug("nk= %d, nao= %d, naux= %d", nk, nao, naux)
        log.debug("kconserv= %d %d ... %d", kconserv[0,0], kconserv[0,1], kconserv[-1,-1])
        log.debug("phase.real= %e %e ... %e", phase.real[0,0], phase.real[0,1], phase.real[-1,-1])
        log.debug("phase.imag= %e %e ... %e", phase.imag[0,0], phase.imag[0,1], phase.imag[-1,-1])
        log.debug("j3c_k.real= %e %e ... %e", j3c_k.real[0,0,0,0,0], j3c_k.real[0,0,0,0,1], j3c_k.real[-1,-1,-1,-1,-1])
        log.debug("j3c_k.imag= %e %e ... %e", j3c_k.imag[0,0,0,0,0], j3c_k.imag[0,0,0,0,1], j3c_k.imag[-1,-1,-1,-1,-1])
        log.debug("j3c.real= %e %e ... %e", j3c.real[0,0,0], j3c.real[0,0,1], j3c.real[-1,-1,-1])
        log.debug("j3c.imag= %e %e ... %e", j3c.imag[0,0,0], j3c.imag[0,0,1], j3c.imag[-1,-1,-1])
        log.debug("kconserv.flags: %r", kconserv.flags)
        log.debug("j3c_k.flags: %r", j3c_k.flags)
        log.debug("j3c.flags: %r", j3c.flags)

        libpbc.j3c_k2gamma(
                ctypes.c_int64(nk),
                ctypes.c_int64(nao),
                ctypes.c_int64(naux),
                kconserv.ctypes.data_as(ctypes.c_void_p),
                phase.ctypes.data_as(ctypes.c_void_p),
                j3c_k.ctypes.data_as(ctypes.c_void_p),
                # Out
                j3c.ctypes.data_as(ctypes.c_void_p),
                )
        log.debug("Done.")
        log.debug("j3c.real= %e %e ... %e", j3c.real[0,0,0], j3c.real[0,0,1], j3c.real[-1,-1,-1])
        log.debug("j3c.imag= %e %e ... %e", j3c.imag[0,0,0], j3c.imag[0,0,1], j3c.imag[-1,-1,-1])
        log.debug("j3c.flags: %r", j3c.flags)
        log.debug("j3c.shape: %r", list(j3c.shape))

    # Discard imaginary part if close to zero
    if imag_tol and abs(j3c.imag).max() < imag_tol:
        j3c = j3c.real

    if compact:
        j3c = j3c.reshape(nk*naux, ncomp) / np.sqrt(nk)
    else:
        j3c = j3c.reshape(nk*naux, nk*nao_out, nk*nao_out) / np.sqrt(nk)

    log.debug("Memory for 3c-integrals: %.3f GB", j3c.nbytes/1e9)
    log.debug("Time to unfold three-center integrals: %.3f", timer()-t0)
    log.debug("Max imaginary element of j3c: %.3e", abs(j3c.imag).max())

    #check_imag = True
    if check_imag:
        im_norm = 0.0
        im_max = 0.0
        j3c_full = pyscf.lib.unpack_tril(j3c) if compact else j3c
        for i in range(nk*nao_out):
            eri = einsum("Lj,Lkl->jkl", j3c_full[:,i].conj(), j3c_full)
            im_max = max(im_max, abs(eri.imag).max())
            im_norm += np.linalg.norm(eri.imag)**2
            log.info("i= %3d norm= %.5g max= %.5g", i, np.sqrt(im_norm), im_max)
        im_norm = np.sqrt(im_norm)
        log.info("Imaginary part of (ab|cd) in AO basis: norm= %.5g max= %.5g", im_norm, im_max)

    return j3c

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
