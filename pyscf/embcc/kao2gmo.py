"""AO to MO transformation from k-space (AOs) to supercell real space (MOs)

Author: Max Nusspickel
Email:  max.nusspickel@gmail.com
"""

# Standard
from timeit import default_timer as timer
import ctypes
import logging
# External
import numpy as np
# PySCF
import pyscf
import pyscf.lib
from pyscf.mp.mp2 import _mo_without_core
import pyscf.pbc
import pyscf.pbc.df
import pyscf.pbc.tools
from pyscf.pbc.lib import kpts_helper
# Package
from .util import einsum, memory_string

log = logging.getLogger(__name__)

def gdf_to_pyscf_eris(mf, gdf, cm, fock=None):
    """Get supercell MO eris from k-point sampled GDF.

    This folds the MO back into k-space
    and contracts with the k-space three-center integrals..

    Arguments
    ---------
    mf: pyscf.scf.hf.RHF
        Supercell mean-field object.
    gdf: pyscf.pbc.df.GDF
        Gaussian density-fit object of primitive cell (with k-points)
    cm: pyscf.mp.mp2.MP2 or pyscf.cc.ccsd.CCSD
        Correlated method, must have mo_coeff set.
    fock: (N,N) array, optional
        Fock matrix. If None, calculated via mf.get_fock().

    Returns
    -------
    eris: pyscf.mp.mp2._ChemistsERIs or pyscf.cc.rccsd._ChemistsERIs
        ERIs which can be used for the respective correlated method.
    """
    #log.debug("Correlated method in eris_kao2gmo= %s", type(cm))

    if fock is None: fock = mf.get_fock()

    # MP2 ERIS
    if isinstance(cm, pyscf.mp.mp2.MP2):
        from pyscf.mp.mp2 import _ChemistsERIs
        eris = _ChemistsERIs()
        only_ovov = True
    # Coupled-cluster ERIS
    elif isinstance(cm, pyscf.cc.ccsd.CCSD):
        from pyscf.cc.rccsd import _ChemistsERIs
        eris = _ChemistsERIs()
        only_ovov = False
    else:
        raise NotImplementedError("Unknown correlated method= %s" % type(cm))

    mo_coeff = _mo_without_core(cm, cm.mo_coeff)
    eris.mo_coeff = mo_coeff
    eris.nocc = cm.nocc
    eris.e_hf = cm._scf.e_tot
    eris.fock = np.linalg.multi_dot((mo_coeff.T, fock, mo_coeff))
    eris.mo_energy = eris.fock.diagonal().copy()

    # Remove EXXDIV correction from Fock matrix (necessary for CCSD)
    if mf.exxdiv and isinstance(cm, pyscf.cc.ccsd.CCSD):
        madelung = pyscf.pbc.tools.madelung(mf.mol, mf.kpt)
        for i in range(eris.nocc):
            eris.fock[i,i] += madelung

    g = gdf_to_eris(gdf, mo_coeff, cm.nocc, only_ovov=only_ovov)
    for key, val in g.items():
        setattr(eris, key, val)

    return eris


def gdf_to_eris(gdf, mo_coeff, nocc, only_ovov=False, real_j3c=False):
    """Make supercell ERIs from k-point sampled, density-fitted three-center integrals.

    Arguments
    ---------
    cell : pyscf.pbc.gto.Cell
        Primitive unit cell, not supercell!
    gdf : pyscf.pbc.df.GDF
        Density fitting object of primitive cell, with k-points.
    mo_coeff : (Nr*Naux, Nmo) array
        MO coefficients in supercell. The AOs in the supercell must be ordered
        in the same way as the k-points in the primitive cell!
    nocc : int
        Number of occupied orbitals.
    kpts : (Nk, 3)
        k-point coordinates.
    only_ovov : bool, optional
        Only calculate (occ,vir|occ,vir)-type ERIs (for MP2). Default=False.

    Returns
    -------
    eris : dict
        Dict of supercell ERIs. Has elements "ovov" `if only_ovov == True`
        and "oooo", "ovoo", "oovv", "ovov", "ovvo", "ovvv", and "vvvv" otherwise.
    """
    # DF compatiblity layer
    ints3c = ThreeCenterInts.init_from_gdf(gdf)

    # If GDF was loaded from hdf5 file:

    #if gdf.auxcell is None: gdf.build(with_j3c=False)
    #cell = gdf.cell
    #kpts = gdf.kpts
    cell, kpts, nk, nao, naux = ints3c.cell, ints3c.kpts, ints3c.nk, ints3c.nao, ints3c.naux

    phase = pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]
    #nk = len(kpts)
    nmo = mo_coeff.shape[-1]
    #nao = cell.nao_nr()         # Atomic orbitals in primitive cell
    #naux = gdf.auxcell.nao_nr()  # Auxiliary size in primitive cell
    nvir = nmo - nocc
    o, v = np.s_[:nocc], np.s_[nocc:]

    if only_ovov:
        mem_j3c = nk*naux*nocc*nvir * 16
        mem_eris = nocc**2*nvir**2 * 8
    else:
        mem_j3c = nk*naux*(nocc*nvir + nocc*nocc + nvir*nvir) * 16
        mem_eris = (nocc**4 + nocc**3*nvir + 3*nocc**2*nvir**2 + nocc*nvir**3 + nvir**4)*8
    log.debug("Memory needed for kAO->GMO: temporary= %s final= %s",
            memory_string(max(mem_j3c, mem_j3c/2+mem_eris)), memory_string(mem_eris))

    # Transform: (l|ka,qb) -> (Rl|i,j)
    mo_coeff = mo_coeff.reshape(nk, nao, nmo)
    ck_o = einsum("Rk,Rai->kai", phase.conj(), mo_coeff[:,:,o]) / np.power(nk, 0.25)
    ck_v = einsum("Rk,Rai->kai", phase.conj(), mo_coeff[:,:,v]) / np.power(nk, 0.25)
    t0 = timer()
    j3c_ov, j3c_oo, j3c_vv = j3c_kao2gmo(ints3c, ck_o, ck_v, only_ov=only_ovov, make_real=real_j3c)
    t_trafo = (timer()-t0)
    # Composite auxiliary index: R,l -> L
    j3c_ov = j3c_ov.reshape(nk*naux, nocc, nvir)
    if not only_ovov:
        j3c_oo = j3c_oo.reshape(nk*naux, nocc, nocc)
        j3c_vv = j3c_vv.reshape(nk*naux, nvir, nvir)

    # Contract Lij,Lkl->ijkl
    # TODO: Parallelize in C / numba?
    t0 = timer()
    eris = {}
    contract = lambda l, r : np.tensordot(l.conj(), r, axes=(0, 0))
    if not only_ovov:
        eris["vvvv"] = contract(j3c_vv, j3c_vv)
        eris["ovvv"] = contract(j3c_ov, j3c_vv)
        eris["oovv"] = contract(j3c_oo, j3c_vv)
        del j3c_vv
    eris["ovov"] = contract(j3c_ov, j3c_ov)
    if not only_ovov:
        eris["ovvo"] = contract(j3c_ov, j3c_ov.transpose(0, 2, 1))
        eris["ovoo"] = contract(j3c_ov, j3c_oo)
        del j3c_ov
        eris["oooo"] = contract(j3c_oo.conj(), j3c_oo)
        del j3c_oo
    t_contract = (timer()-t0)

    if not real_j3c:
        for key in list(eris.keys()):
            val = eris[key]
            inorm = np.linalg.norm(val.imag)
            imax = abs(val.imag).max()
            if max(inorm, imax) > 1e-5:
                log.warning("Im(%2s|%2s):  ||x||= %.2e  max(|x|)= %.2e !", key[:2], key[2:], inorm, imax)
            else:
                log.debug("Im(%2s|%2s):  ||x||= %.2e  max(|x|)= %.2e", key[:2], key[2:], inorm, imax)
            eris[key] = val.real

    log.debug("Timings for kAO->GMO [s]: transform= %.2f  contract= %.2f", t_trafo, t_contract)

    return eris

class ThreeCenterInts:
    """Interface class for DF classes."""

    def __init__(self, cell, kpts, naux):
        self.cell = cell
        self.kpts = kpts
        self.naux = naux
        self.values = None
        self.df = None

    @property
    def nk(self):
        return len(self.kpts)

    @property
    def nao(self):
        return self.cell.nao_nr()

    #def memory_ov(self, nocc, nvir):
    #    return self.nk*self.naux*nocc*nvir * 16

    #def memory(self, nocc, nvir):
    #    return self.nk*self.naux*(nocc*nvir + nocc*nocc + nvir*nvir) * 16

    @classmethod
    def init_from_gdf(cls, gdf):
        if gdf.auxcell is None:
            gdf.build(with_j3c=False)
        ints3c = cls(gdf.cell, gdf.kpts, gdf.auxcell.nao_nr())
        ints3c.df = gdf
        return ints3c

    def sr_loop(self, *args, **kwargs):
        return self.df.sr_loop(*args, **kwargs)

    def get_array(self, kptsym=True):

        if self.values is not None:
            return self.values, None

        elif isinstance(self.df._cderi, str):
            #import h5py
            #with h5py.File(self.df._cderi) as f:
            #    log.info("h5py keys: %r", list(f.keys()))
            #    #for key in list(f.keys()):
            #    log.info("h5py keys of j3c: %r", list(f["j3c"].keys()))
            #    log.info("h5py kptij: %r", list(f["j3c-kptij"].shape))

            if kptsym:
                nkij = self.nk*(self.nk+1)//2
                j3c = np.zeros((nkij, self.naux, self.nao, self.nao), dtype=complex)
                kuniq_map = np.zeros((self.nk, self.nk), dtype=np.int)
                kij = 0
                for ki in range(self.nk):
                    for kj in range(ki+1):
                        kuniq_map[ki,kj] = kij
                        kpts_ij = (self.kpts[ki], self.kpts[kj])
                        blk0 = 0
                        for lr, li, sign in self.df.sr_loop(kpts_ij, compact=False, blksize=int(1e9)):
                            assert (sign == 1)
                            blksize = lr.shape[0]
                            blk = np.s_[blk0:blk0+blksize]
                            blk0 += blksize
                            j3c[kij,blk] = (lr+1j*li).reshape(blksize, self.nao, self.nao) #* factor
                        if blk0 != self.naux:
                            log.info("Naux(ki= %3d, kj= %3d)= %4d", ki, kj, blk0)
                        kij += 1

                # At this point, all kj <= ki are set
                # Here we loop over kj > ki
                for ki in range(self.nk):
                    for kj in range(ki+1, self.nk):
                        kuniq_map[ki,kj] = -kuniq_map[kj,ki]
                assert np.all(kuniq_map < nkij)
                assert np.all(kuniq_map > -nkij)

            else:
                j3c = np.zeros((self.nk, self.nk, self.naux, self.nao, self.nao), dtype=complex)
                kuniq_map = None
                for ki in range(self.nk):
                    for kj in range(self.nk):
                        kij = (self.kpts[ki], self.kpts[kj])
                        blk0 = 0
                        for lr, li, sign in self.df.sr_loop(kij, compact=False, blksize=int(1e9)):
                            assert (sign == 1)
                            blksize = lr.shape[0]
                            blk = np.s_[blk0:blk0+blksize]
                            blk0 += blksize
                            j3c[ki,kj,blk] = (lr+1j*li).reshape(blksize, self.nao, self.nao) #* factor
                        if blk0 != self.naux:
                            log.info("Naux(ki= %3d, kj= %3d)= %4d", ki, kj, blk0)

        # Can access array directly
        #if isinstance(self.df, pyscf.pbc.df.df_incore.IncoreGDF):
        elif isinstance(self.df, pyscf.pbc.df.df_incore.IncoreGDF):
            #j3c = self.df._cderi["j3c"].reshape(-1, self.naux, self.nao, self.nao).copy()
            j3c = self.df._cderi["j3c"].reshape(-1, self.naux, self.nao, self.nao)
            #log.info("N uniq(ki-kj)= %d", j3c.shape[0])
            nkuniq = j3c.shape[0]
            log.info("Nkuniq= %3d", nkuniq)
            # Check map
            _get_kpt_hash = pyscf.pbc.df.df_incore._get_kpt_hash
            kuniq_map = np.zeros((self.nk, self.nk), dtype=np.int)
            for ki in range(self.nk):
                for kj in range(self.nk):
                    kij = np.asarray((self.kpts[ki], self.kpts[kj]))
                    kij_id = self.df._cderi['j3c-kptij-hash'].get(_get_kpt_hash(kij), [None])
                    assert len(kij_id) == 1
                    kij_id = kij_id[0]
                    if kij_id is None:
                        kij_id = self.df._cderi['j3c-kptij-hash'][_get_kpt_hash(kij[[1,0]])]
                        assert len(kij_id) == 1
                        # negative to indicate transpose needed
                        kij_id = -kij_id[0]
                    assert (abs(kij_id) < nkuniq)

                    #tmp = j3c2[abs(kij_id)].reshape(self.naux,self.nao,self.nao)
                    #if kij_id < 0:
                    #    tmp = tmp.transpose(0,2,1).conj()

                    #log.debug("ki= %d kj= %d kij= %d", ki, kj, kij_id)
                    #assert np.allclose(j3c[ki,kj], tmp)

                    kuniq_map[ki,kj] = kij_id

            #for ki in range(self.nk):
            #    for kj in range(self.nk):
            #        log.info("ki= %3d kj= %3d -> kij= %3d", ki, kj, kuniq_map[ki,kj])

        return j3c, kuniq_map



#def j3c_kao2gmo(gdf, cocc, cvir, only_ov=False, make_real=True, driver='c', factor=1):
#def j3c_kao2gmo(ints3c, cocc, cvir, only_ov=False, make_real=True, driver='c', factor=1):
def j3c_kao2gmo(ints3c, cocc, cvir, only_ov=False, make_real=True, driver='c'):
    #cell = gdf.cell
    #kpts = gdf.kpts
    #nk = len(kpts)
    nocc = cocc.shape[-1]
    nvir = cvir.shape[-1]
    #naux = gdf.auxcell.nao_nr()
    cell, kpts, nk, naux = ints3c.cell, ints3c.kpts, ints3c.nk, ints3c.naux
    nao = cell.nao_nr()
    t0 = timer()
    #kconserv = kpts_helper.get_kconserv(cell, kpts)[:,:,0].copy()
    kconserv = kpts_helper.get_kconserv(cell, kpts, n=2)
    log.debug("Time to make kconserv: %.2f", timer()-t0)

    j3c_ov = np.zeros((nk, naux, nocc, nvir), dtype=complex)
    if not only_ov:
        j3c_oo = np.zeros((nk, naux, nocc, nocc), dtype=complex)
        j3c_vv = np.zeros((nk, naux, nvir, nvir), dtype=complex)
    else:
        j3c_oo = j3c_vv = None

    if driver.lower() == 'python':
        for ki in range(nk):
            for kj in range(nk):
                kij = (kpts[ki], kpts[kj])
                kk = kconserv[ki,kj]
                blk0 = 0
                #for lr, li, sign in gdf.sr_loop(kij, compact=False):
                for lr, li, sign in ints3c.sr_loop(kij, compact=False):
                    assert (sign == 1)
                    blksize = lr.shape[0]
                    blk = np.s_[blk0:blk0+blksize]
                    blk0 += blksize

                    j3c_kij = (lr+1j*li).reshape(blksize, nao, nao) #* factor

                    j3c_ov[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cocc[ki].conj(), cvir[kj])      # O(Nk^2 * Nocc * Nvir)
                    if only_ov: continue
                    j3c_oo[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cocc[ki].conj(), cocc[kj])      # O(Nk^2 * Nocc * Nocc)
                    j3c_vv[kk,blk] += einsum("Lab,ai,bj->Lij", j3c_kij, cvir[ki].conj(), cvir[kj])      # O(Nk^2 * Nvir * Nvir)

    elif driver.lower() == 'c':
        # Load j3c into memory
        t0 = timer()
        #j3c, kunique = load_j3c(gdf, factor=factor)
        #j3c = ints3c.get_array(factor=factor)
        j3c, kunique = ints3c.get_array()

        # Mapping from (ki,kj) -> unique(ki-kj)
        if kunique is None:
            kunique_pt = ctypes.POINTER(ctypes.c_void_p)()
        else:
            kunique_pt = kunique.ctypes.data_as(ctypes.c_void_p)
        if only_ov:
            j3c_oo_pt = j3c_vv_pt = ctypes.POINTER(ctypes.c_void_p)()
        else:
            j3c_oo_pt = j3c_oo.ctypes.data_as(ctypes.c_void_p)
            j3c_vv_pt = j3c_vv.ctypes.data_as(ctypes.c_void_p)
        log.debug("Time to load j3c from file: %.2f", timer()-t0)

        cocc = cocc.copy()
        cvir = cvir.copy()

        libpbc = pyscf.lib.load_library("libpbc")
        t0 = timer()
        ierr = libpbc.j3c_kao2gmo(
                ctypes.c_int64(nk),
                ctypes.c_int64(nao),
                ctypes.c_int64(nocc),
                ctypes.c_int64(nvir),
                ctypes.c_int64(naux),
                kconserv.ctypes.data_as(ctypes.c_void_p),
                kunique_pt,
                cocc.ctypes.data_as(ctypes.c_void_p),
                cvir.ctypes.data_as(ctypes.c_void_p),
                j3c.ctypes.data_as(ctypes.c_void_p),
                # Out
                j3c_ov.ctypes.data_as(ctypes.c_void_p),
                j3c_oo_pt,
                j3c_vv_pt)
        log.debug("Time in j3c_kao2gamo in C: %.2f", timer()-t0)
        assert (ierr == 0)

    if make_real:
        t0 = timer()
        phase = pyscf.pbc.tools.k2gamma.get_phase(cell, kpts)[1]
        j3c_ov = np.tensordot(phase, j3c_ov, axes=1)
        if j3c_ov.size > 0:
            inorm = np.linalg.norm(j3c_ov.imag)
            imax = abs(j3c_ov.imag).max()
            if max(inorm, imax) > 1e-5:
                log.warning("WARNING: Im(L|ov):  norm= %.2e  max= %.2e", inorm, imax)
            else:
                log.debug("Im(L|ov):  norm= %.2e  max= %.2e", inorm, imax)
        j3c_ov = j3c_ov.real
        if not only_ov:
            j3c_oo = np.tensordot(phase, j3c_oo, axes=1)
            j3c_vv = np.tensordot(phase, j3c_vv, axes=1)
            inorm = np.linalg.norm(j3c_oo.imag)
            imax = abs(j3c_oo.imag).max()
            if max(inorm, imax) > 1e-5:
                log.warning("WARNING: Im(L|oo):  norm= %.2e  max= %.2e", inorm, imax)
            else:
                log.debug("Im(L|oo):  norm= %.2e  max= %.2e", inorm, imax)
            inorm = np.linalg.norm(j3c_vv.imag)
            imax = abs(j3c_vv.imag).max()
            if max(inorm, imax) > 1e-5:
                log.warning("WARNING: Im(L|vv):  norm= %.2e  max= %.2e", inorm, imax)
            else:
                log.debug("Im(L|vv):  norm= %.2e  max= %.2e", inorm, imax)
            j3c_oo = j3c_oo.real
            j3c_vv = j3c_vv.real
        log.debug("Time to rotate to real: %.2f", timer()-t0)

    return j3c_ov, j3c_oo, j3c_vv

#def load_j3c(gdf, factor=1):
#    kpts = gdf.kpts
#    nk = len(kpts)
#    naux = gdf.auxcell.nao_nr()
#    nao = gdf.cell.nao_nr()
#
#    #if isinstance(gdf._cderi, np.ndarray):
#    #    return gdf._cderi
#
#    j3c = np.zeros((nk, nk, naux, nao, nao), dtype=complex)
#    for ki in range(nk):
#        for kj in range(nk):
#            kij = (kpts[ki], kpts[kj])
#            blk0 = 0
#            for lr, li, sign in gdf.sr_loop(kij, compact=False, blksize=int(1e9)):
#                assert (sign == 1)
#                blksize = lr.shape[0]
#                blk = np.s_[blk0:blk0+blksize]
#                blk0 += blksize
#                j3c[ki,kj,blk] = (lr+1j*li).reshape(blksize, nao, nao) * factor
#    kuniq_map = None
#
#    if False and isinstance(gdf, pyscf.pbc.df.df_incore.IncoreGDF):
#        log.debug("IncoreGDF detected - trying to access j3c directly")
#        j3c2 = (gdf._cderi["j3c"]*factor).reshape(-1, naux, nao, nao).copy()
#        log.debug("j3c.flags= %r", j3c2.flags)
#        log.debug("j3c.shape=  %r", tuple(j3c.shape))
#        log.debug("j3c2.shape= %r", tuple(j3c2.shape))
#
#        # This is wrong?
#        #kdij = []
#        #for ki in range(nk):
#        #    for kj in range(nk):
#        #        kdij.append(kpts[ki]-kpts[kj])
#        #kdij = np.asarray(kdij)
#        #kuniq_map = pyscf.pbc.lib.kpts_helper.unique(kdij)[2].reshape(nk, nk)
#        #log.debug("K-map:\n%r", kuniq_map)
#
#        # Check map
#        _get_kpt_hash = pyscf.pbc.df.df_incore._get_kpt_hash
#        kuniq_map = np.zeros((nk, nk), dtype=np.int)
#        for ki in range(nk):
#            for kj in range(nk):
#                kij = np.asarray((kpts[ki], kpts[kj]))
#                kij_id = gdf._cderi['j3c-kptij-hash'].get(_get_kpt_hash(kij), [None])
#                assert len(kij_id) == 1
#                kij_id = kij_id[0]
#                if kij_id is None:
#                    kij_id = gdf._cderi['j3c-kptij-hash'][_get_kpt_hash(kij[[1,0]])]
#                    assert len(kij_id) == 1
#                    # negative to indicate transpose needed
#                    kij_id = -kij_id[0]
#
#                tmp = j3c2[abs(kij_id)].reshape(naux,nao,nao)
#                if kij_id < 0:
#                    tmp = tmp.transpose(0,2,1).conj()
#
#                log.debug("ki= %d kj= %d kij= %d", ki, kj, kij_id)
#                assert np.allclose(j3c[ki,kj], tmp)
#
#                kuniq_map[ki,kj] = kij_id
#
#        for ki in range(nk):
#            for kj in range(nk):
#                kij = abs(kuniq_map[ki,kj])
#                log.debug("First elements of ki= %d kj= %d kij= %d -> %f %f vs %f %f", ki, kj, kij, j3c[ki,kj,0,0,0].real, j3c[ki,kj,0,0,0].imag, j3c2[kij,0,0,0].real, j3c2[kij,0,0,0].imag)
#
#        #log.debug("K-map:\n%r", kuniq_map)
#
#        #log.debug("j3c[0,0]:\n%r", j3c[0,0])
#        #log.debug("j3c2[0]:\n%r", j3c2[0].reshape(naux,nao,nao))
#
#        #log.debug("j3c[0,1]:\n%r", j3c[0,1])
#        #idx = gdf._cderi['j3c-kptij-hash'].get(_get_kpt_hash((kpts[0], kpts[1])), None)
#        #if idx is None:
#        #    idx = gdf._cderi['j3c-kptij-hash'].get(_get_kpt_hash((kpts[1], kpts[0])), None)
#        #idx = idx[0]
#        #log.debug("j3c2[1]:\n%r", j3c2[idx].reshape(naux,nao,nao).transpose(0, 2, 1).conj())
#
#    else:
#        pass
#        #j3c = np.zeros((nk, nk, naux, nao, nao), dtype=complex)
#        #for ki in range(nk):
#        #    for kj in range(nk):
#        #        kij = (kpts[ki], kpts[kj])
#        #        blk0 = 0
#        #        for lr, li, sign in gdf.sr_loop(kij, compact=False, blksize=int(1e9)):
#        #            assert (sign == 1)
#        #            blksize = lr.shape[0]
#        #            blk = np.s_[blk0:blk0+blksize]
#        #            blk0 += blksize
#        #            j3c[ki,kj,blk] = (lr+1j*li).reshape(blksize, nao, nao) * factor
#        #kuniq_map = None
#
#    return j3c, kuniq_map
#
#
#
#if __name__ == "__main__":
#    from test_systems import graphite
#    cell = graphite(basis="gth-dzvp")
#    kmesh = [6, 6, 3]
#    #kmesh = [2, 2, 1]
#    kpts = cell.make_kpts(kmesh)
#    nk = len(kpts)
#    nao = cell.nao_nr()
#    naux = 336
#
#    ints3c = ThreeCenterInts(cell, kpts, naux)
#    print("Random values...")
#    #ints3c.values = (np.random.rand(nk, nk, naux, nao, nao)
#    #               + np.random.rand(nk, nk, naux, nao, nao)*1j)
#    ints3c.values = np.zeros((nk, nk, naux, nao, nao), dtype=complex)
#    ints3c.values[:] = 2.0
#    print("Size of ints3c: %.2f GB" % (ints3c.values.nbytes/1e9))
#
#    #nocc = 860
#    #nvir = 4
#    nocc = 4
#    nvir = 4748
#    ck_o = np.random.rand(nk, nao, nocc) + np.random.rand(nk, nao, nocc)*1j
#    ck_v = np.random.rand(nk, nao, nvir) + np.random.rand(nk, nao, nvir)*1j
#
#    #print("Occupieds")
#    print("Virtuals")
#    j3c_ov, j3c_oo, j3c_vv = j3c_kao2gmo(ints3c, ck_o, ck_v, only_ov=True)
#    print("Done!")
