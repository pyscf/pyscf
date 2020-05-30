import logging
import os.path
import functools

import numpy as np
import scipy
import scipy.linalg
from mpi4py import MPI

import pyscf
import pyscf.lo
import pyscf.cc
import pyscf.ci
import pyscf.mcscf
import pyscf.fci

__all__ = [
        "EmbCC",
        ]

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

einsum = functools.partial(np.einsum, optimize=True)

class Cluster:

    def __init__(self, base, name, indices, solver="CCSD", bath_type=None, tol_bath=1e-3, tol_dmet_bath=1e-8):
        """
        Parameters
        ----------
        name :
            Name of cluster.
        indices:
            Atomic orbital indices of cluster.
        """

        self.base = base
        self.name = name
        self.indices = indices
        # Optional
        self.solver = solver
        self.bath_type = bath_type
        self.tol_bath = tol_bath
        self.tol_dmet_bath = tol_dmet_bath

        # Indices which are NOT in the cluster
        self.rest = np.asarray([i for i in np.arange(self.mol.nao_nr()) if i not in self.indices])

        # Output attributes
        self.converged = True
        self.e_corr = 0.0

        self.e_cl_ccsd = 0.0
        self.e_ccsd = 0.0
        self.e_pt = 0.0
        self.nbath0 = 0
        self.nbath = 0
        self.nfrozen = 0

        self.e_ccsd_v = 0.0
        self.e_ccsd_w = 0.0
        self.e_ccsd_z = 0.0

    def __len__(self):
        return len(self.indices)

    @property
    def mf(self):
        return self.base.mf

    @property
    def mol(self):
        return self.base.mol


    def make_projector(self):
        """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
        S1 = self.mf.get_ovlp()
        nao = self.mol.nao_nr()
        S2 = S1[np.ix_(self.indices, self.indices)]
        S21 = S1[self.indices]
        #s2_inv = np.linalg.inv(s2)
        #p_21 = np.dot(s2_inv, s21)
        # Better: solve with Cholesky decomposition
        # Solve: S2 * p_21 = S21 for p_21
        p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        p_12 = np.eye(nao)[:,self.indices]
        p = np.dot(p_12, p_21)
        return p

    def make_projector_s121(self):
        """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
        S1 = self.mf.get_ovlp()
        nao = self.mol.nao_nr()
        S2 = S1[np.ix_(self.indices, self.indices)]
        S21 = S1[self.indices]
        #s2_inv = np.linalg.inv(s2)
        #p_21 = np.dot(s2_inv, s21)
        # Better: solve with Cholesky decomposition
        # Solve: S2 * p_21 = S21 for p_21
        p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        #p_12 = np.eye(nao)[:,self.indices]
        p = np.dot(S21.T, p_21)
        return p


    #def make_local_orbitals(self, tol=1e-9):
    #    S = self.mf.get_ovlp()
    #    #S_inv = np.linalg.inv(S)
    #    C = self.mf.mo_coeff
    #    S_inv = np.dot(C, C.T)
    #    P = self.make_projector()

    #    D_loc = np.linalg.multi_dot((P, S_inv, P.T))
    #    C = self.mf.mo_coeff.copy()
    #    SC = np.dot(S, C)

    #    # Transform to C
    #    D_loc = np.linalg.multi_dot((SC.T, D_loc, SC))
    #    e, r = np.linalg.eigh(D_loc)
    #    rev = np.s_[::-1]
    #    e = e[rev]
    #    r = r[:,rev]

    #    nloc = len(e[e>tol])
    #    assert nloc == len(self), "Error finding local orbitals: %s" % e
    #    #C_loc = np.dot(C, r[:,:nimp])
    #    C = np.dot(C, r)

    #    return C

    def make_local_orbitals(self):
        S = self.mf.get_ovlp()
        norb = S.shape[-1]

        S121 = self.make_projector_s121()
        assert np.allclose(S121, S121.T)
        e, C = scipy.linalg.eigh(S121, b=S)
        rev = np.s_[::-1]
        e = e[rev]
        C = C[:,rev]

        nloc = len(e[e>1e-9])
        assert nloc == len(self), "Error finding local orbitals: %s" % e
        assert np.allclose(np.linalg.multi_dot((C.T, S, C)), np.eye(C.shape[-1]))

        return C

    def make_dmet_bath_orbitals(self, C, tol=None):
        if tol is None:
            tol = self.tol_dmet_bath
        C = C.copy()
        env = np.s_[len(self):]
        S = self.mf.get_ovlp()
        D = np.linalg.multi_dot((C[:,env].T, S, self.mf.make_rdm1(), S, C[:,env])) / 2
        e, v = np.linalg.eigh(D)
        reverse = np.s_[::-1]
        e = e[reverse]
        v = v[:,reverse]
        mask_bath = np.fmin(abs(e), abs(e-1)) >= tol

        sort = np.argsort(np.invert(mask_bath), kind="mergesort")
        e = e[sort]
        v = v[:,sort]

        nbath0 = sum(mask_bath)
        nenvocc = sum(e[nbath0:] > 0.5)

        log.debug("DMET bath eigenvalues:\n%s\nFollowing eigenvalues:\n%s", e[:nbath0], e[nbath0:nbath0+3])

        assert nbath0 <= len(self)

        C[:,env] = np.dot(C[:,env], v)

        return C, nbath0, nenvocc

    def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=None, normalize=False):
    #def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=None, normalize=True):
        if tol is None:
            tol = self.tol_bath

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        b = np.einsum("xi,i,ai->xa", csc[non_local], e**power, csc[loc], optimize=True)

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of kind=%s, power=%d bath, tolerance=%e", kind, power, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def make_uncontracted_dmet_orbitals(self, C, kind, non_local, tol=None, normalize=False):
    #def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=None, normalize=True):
        if tol is None:
            tol = self.tol_bath

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        b = np.einsum("xi,ai->xia", csc[non_local], csc[loc], optimize=True)
        b = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of uncontracted DMET bath of kind=%s, tolerance=%e", kind, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=None, normalize=False):
        if tol is None:
            tol = self.tol_bath

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        b = np.einsum("xi,i,ai->xa", csc[non_local], e**power, csc[loc], optimize=True)

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of kind=%s, power=%d bath, tolerance=%e", kind, power, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath


    def make_matsubara_bath_orbitals(self, C, kind, non_local, npoints=1000, beta=100.0, tol=None, normalize=False):

        if kind == "occ":
            mask = self.mf.mo_occ > 0
        elif kind == "vir":
            mask = self.mf.mo_occ == 0
        else:
            raise ValueError()

        S = self.mf.get_ovlp()
        csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
        e = self.mf.mo_energy[mask]

        loc = np.s_[:len(self)]

        # Matsubara points
        wn = (2*np.arange(npoints)+1)*np.pi/beta
        kernel = wn[np.newaxis,:] / np.add.outer(self.mf.mo_energy[mask]**2, wn**2)

        b = np.einsum("xi,iw,ai->xaw", csc[non_local], kernel, csc[loc], optimize=True)
        b = b.reshape(b.shape[0], b.shape[1]*b.shape[2])

        if normalize:
            b /= np.linalg.norm(b, axis=1, keepdims=True)
            assert np.allclose(np.linalg.norm(b, axis=1), 1)

        p = np.dot(b, b.T)
        e, v = np.linalg.eigh(p)
        assert np.all(e > -1e-13)
        rev = np.s_[::-1]
        e = e[rev]
        v = v[:,rev]

        nbath = sum(e >= tol)
        #log.debug("Eigenvalues of kind=%s, power=%d bath:\n%r", kind, power, e)
        log.debug("Eigenvalues of kind=%s Matsubara bath, tolerance=%e", kind, tol)
        log.debug("%d eigenvalues above threshold:\n%r", nbath, e[:nbath])
        log.debug("%d eigenvalues below threshold:\n%r", len(e)-nbath, e[nbath:])

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath



    #def make_power_bath_orbitals_2(self, C, kind, non_local, power=1, tol=None, normalize=False):
    #    if tol is None:
    #        tol = self.tol_bath

    #    if kind == "occ":
    #        mask = self.mf.mo_occ > 0
    #    elif kind == "vir":
    #        mask = self.mf.mo_occ == 0
    #    else:
    #        raise ValueError()

    #    C = C.copy()
    #    #env = np.s_[len(self):]
    #    S = self.mf.get_ovlp()
    #    dm = np.einsum("ai,i,bi->ab",
    #            self.mf.mo_coeff[:,mask], self.mf.mo_energy[mask]**power, self.mf.mo_coeff[:,mask])
    #    D = np.linalg.multi_dot((C[:,non_local].T, S, dm, S, C[:,non_local]))
    #    e, v = np.linalg.eigh(D)
    #    reverse = np.s_[::-1]
    #    e = e[reverse]
    #    v = v[:,reverse]

    #    print("new")
    #    print(e)
    #    print(self.mf.mo_energy[self.mf.mo_energy>0])
    #    1/0


    #    S = self.mf.get_ovlp()
    #    csc = np.linalg.multi_dot((C.T, S, self.mf.mo_coeff[:,mask]))
    #    e = self.mf.mo_energy[mask]

    #    loc = np.s_[:len(self)]

    #    b = np.einsum("xi,i,ai->xa", csc[non_local], e**power, csc[loc], optimize=True)

    #    if normalize:
    #        b /= np.linalg.norm(b, axis=1, keepdims=True)
    #        assert np.allclose(np.linalg.norm(b, axis=1), 1)

    #    p = np.dot(b, b.T)
    #    e, v = np.linalg.eigh(p)
    #    assert np.all(e > -1e-13)
    #    rev = np.s_[::-1]
    #    e = e[rev]
    #    v = v[:,rev]

    #    nbath = sum(e >= tol)

    #    C = C.copy()
    #    C[:,non_local] = np.dot(C[:,non_local], v)

    #    return C, nbath

    def run_solver(self, solver=None, max_power=0, pertT=False, diagonalize_fock=True, cc_verbose=4):

        if solver is None:
            solver = self.solver

        C = self.make_local_orbitals()
        C, nbath0, nenvocc = self.make_dmet_bath_orbitals(C)
        nbath = nbath0

        # Add additional power bath orbitals
        nbathpocc = 0
        nbathpvir = 0
        # Power orbitals
        for power in range(1, max_power+1):
            occ_space = np.s_[len(self)+nbath0+nbathpocc:len(self)+nbath0+nenvocc]
            C, nbo = self.make_power_bath_orbitals(C, "occ", occ_space, power=power)
            #C, nbo = self.make_power_bath_orbitals_2(C, "occ", occ_space, power=power)
            #1/0
            vir_space = np.s_[len(self)+nbath0+nenvocc+nbathpvir:]
            C, nbv = self.make_power_bath_orbitals(C, "vir", vir_space, power=power)
            #C, nbv = self.make_power_bath_orbitals_2(C, "vir", vir_space, power=power)
            nbathpocc += nbo
            nbathpvir += nbv
        # Uncontracted DMET
        if self.bath_type == "uncontracted":
            occ_space = np.s_[len(self)+nbath0+nbathpocc:len(self)+nbath0+nenvocc]
            C, nbo = self.make_uncontracted_dmet_orbitals(C, "occ", occ_space, tol=self.tol_bath)
            vir_space = np.s_[len(self)+nbath0+nenvocc+nbathpvir:]
            C, nbv = self.make_uncontracted_dmet_orbitals(C, "vir", vir_space, tol=self.tol_bath)
            nbathpocc += nbo
            nbathpvir += nbv
        # Matsubara
        elif self.bath_type == "matsubara":
            occ_space = np.s_[len(self)+nbath0+nbathpocc:len(self)+nbath0+nenvocc]
            C, nbo = self.make_matsubara_bath_orbitals(C, "occ", occ_space, tol=self.tol_bath)
            vir_space = np.s_[len(self)+nbath0+nenvocc+nbathpvir:]
            C, nbv = self.make_matsubara_bath_orbitals(C, "vir", vir_space, tol=self.tol_bath)
            nbathpocc += nbo
            nbathpvir += nbv

        nbath += nbathpocc
        nbath += nbathpvir

        S = self.mf.get_ovlp()
        SDS_hf = np.linalg.multi_dot((S, self.mf.make_rdm1(), S))

        # Diagonalize cluster DM
        ncl = len(self) + nbath0
        cl = np.s_[:ncl]
        D = np.linalg.multi_dot((C[:,cl].T, SDS_hf, C[:,cl])) / 2
        e, v = np.linalg.eigh(D)
        reverse = np.s_[::-1]
        e = e[reverse]
        v = v[:,reverse]
        C_cc = C.copy()
        C_cc[:,cl] = np.dot(C_cc[:,cl], v)

        # Sort occupancy
        occ = np.einsum("ai,ab,bi->i", C_cc, SDS_hf, C_cc, optimize=True)
        assert np.allclose(np.fmin(abs(occ), abs(occ-2)), 0, atol=1e-6, rtol=0), "Error in occupancy: %s" % occ
        occ = np.asarray([2 if occ > 1 else 0 for occ in occ])
        sort = np.argsort(-occ, kind="mergesort") # mergesort is stable (keeps relative order)
        rank = np.argsort(sort)
        C_cc = C_cc[:,sort]
        occ = occ[sort]
        nocc = sum(occ > 0)
        nactive = len(self) + nbath
        frozen = rank[nactive:]
        active = rank[:nactive]
        nocc_active = sum(occ[active] > 0)

        # Accelerates convergence
        if diagonalize_fock:
            F = np.linalg.multi_dot((C_cc.T, self.mf.get_fock(), C_cc))
            # Occupied active
            o = np.nonzero(occ > 0)[0]
            o = np.asarray([i for i in o if i in active])
            if len(o) > 0:
                e, r = np.linalg.eigh(F[np.ix_(o, o)])
                C_cc[:,o] = np.dot(C_cc[:,o], r)
            # Virtual active
            v = np.nonzero(occ == 0)[0]
            v = np.asarray([i for i in v if i in active])
            if len(v) > 0:
                e, r = np.linalg.eigh(F[np.ix_(v, v)])
                C_cc[:,v] = np.dot(C_cc[:,v], r)

        if solver == "CCSD":
            ccsd = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            ccsd.max_cycle = 100
            ccsd.verbose = cc_verbose
            log.debug("Running CCSD...")
            ccsd.kernel()
            log.debug("CCSD done. converged: %r", ccsd.converged)
            C1 = ccsd.t1
            C2 = ccsd.t2 + einsum('ia,jb->ijab', ccsd.t1, ccsd.t1)

        elif solver == "CISD":
            cisd = pyscf.ci.CISD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            cisd.max_cycle = 100
            cisd.verbose = cc_verbose
            log.debug("Running CISD...")
            cisd.kernel()
            log.debug("CISD done. converged: %r", cisd.converged)
            C0, C1, C2 = cisd.cisdvec_to_amplitudes(cisd.ci)
            # Intermediate normalization
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm

        elif solver == "FCI":
            casci = pyscf.mcscf.CASCI(self.mol, nactive, 2*nocc_active)
            casci.canonicalization = False
            C_cas = pyscf.mcscf.addons.sort_mo(casci, mo_coeff=C_cc, active, base=0)
            log.debug("Running FCI...")
            e_tot, e_cas, wf, mo_coeff, mo_energy = casci.kernel(mo_coeff=C_cas)
            log.debug("FCI done. converged: %r", casci.converged)
            assert np.allclose(mo_coeff, C_cas)
            cisdvec = pyscf.ci.cisd.from_fcivec(wf, nactive, 2*nocc_active)
            C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, nactive, nocc_active)
            # Intermediate normalization
            renorm = 1/C0
            C1 *= renorm
            C2 *= renorm

        else:
            raise ValueError("Unknown solver: %s" % solver)

        self.converged = cc.converged
        self.nbath0 = nbath0
        self.nbath = nbath
        self.nfrozen = len(frozen)

        log.debug("Calculating local energy...")

        if solver == "CCSD":
            self.e_ccsd, self.e_pt = self.get_local_energy(cc, pertT=pertT)
            self.e_ccsd_v, _ = self.get_local_energy(cc, projector="vir", pertT=pertT)
            #self.e_ccsd_w, _ = self.get_local_energy(cc, projector="occ", symmetrize=True, pertT=pertT)
            #self.e_ccsd_z, _ = self.get_local_energy(cc, projector="vir", symmetrize=True, pertT=pertT)

            self.e_ccsd_w, _ = self.get_local_energy(cc, projector="occ-2")
            self.e_ccsd_z = self.get_local_energy_most_indices(cc)

            e_test = self.get_local_energy_new(ccsd, C1, C2)
            assert np.isclose(e_test, self.e_ccsd)

            self.e_corr = self.e_ccsd

            # CCSD energy of whole cluster
            self.e_cl_ccsd = cc.e_corr

        elif solver == "CISD":
            self.e_cisd = self.get_local_energy_new(cisd, C1, C2)
            self.e_corr = self.e_cisd
        elif solver == "FCI":
            # Fake CISD
            cisd = pyscf.ci.CISD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
            self.e_fci = self.get_local_energy_new(cisd, C1, C2)
            self.e_corr = self.e_fci

        log.debug("Calculating local energy done.")

        return int(self.converged)

    def get_local_energy_most_indices(self, cc):

        a = cc.get_frozen_mask()
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]
        CTS = np.dot(C.T, S)

        # Project one index of T amplitudes
        l= self.indices
        r = self.rest
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0

        eris = cc.ao2mo()

        def get_projectors(aos):
            Po = np.dot(CTS[o][:,aos], C[aos][:,o])
            Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
            return Po, Pv

        Lo, Lv = get_projectors(l)
        Ro, Rv = get_projectors(r)

        # Nomenclature:
        # old occupied: i,j
        # old virtual: a,b
        # new occupied: p,q
        # new virtual: s,t
        T1_ll = einsum("pi,ia,sa->ps", Lo, cc.t1, Lv)
        T1_lr = einsum("pi,ia,sa->ps", Lo, cc.t1, Rv)
        T1_rl = einsum("pi,ia,sa->ps", Ro, cc.t1, Lv)
        T1 = T1_ll + (T1_lr + T1_rl)/2

        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * T1)
        if not np.isclose(e1, 0):
            log.warning("Warning: large E1 component: %.8e" % e1)


        tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
        def project_T2(P1, P2, P3, P4):
            T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, tau, P3, P4)
            return T2p

        T2_3l1x = (project_T2(Lo, Lo, Lv, Lv)
                 + project_T2(Lo, Lo, Lv, Rv)
                 + project_T2(Lo, Lo, Rv, Lv)
                 + project_T2(Lo, Ro, Lv, Lv)
                 + project_T2(Ro, Lo, Lv, Lv))

        # Change notation back to i,j and a,b
        e2 = 0.0
        # 4:0 and 3:1 (The easy part)
        e2 += 2*einsum('ijab,iabj', T2_3l1x, eris.ovvo)
        e2 -=   einsum('ijab,jabi', T2_3l1x, eris.ovvo)
        # Loop over other fragments
        for x, cx in enumerate(self.base.clusters):
            if cx == self:
                log.debug("self=%s, cx=%s, skip", self.name, cx.name)
                continue
            # These should be democratic between L and X (factor=0.5)
            Xo, Xv = get_projectors(cx.indices)
            T2_2l2x = 0.5*(project_T2(Lo, Lo, Xv, Xv)
                         + project_T2(Lo, Xo, Lv, Xv)
                         + project_T2(Lo, Xo, Xv, Lv)
                         + project_T2(Xo, Lo, Lv, Xv)
                         + project_T2(Xo, Lo, Xv, Lv)
                         + project_T2(Xo, Xo, Lv, Lv))
            e2 += 2*einsum('ijab,iabj', T2_2l2x, eris.ovvo)
            e2 -=   einsum('ijab,jabi', T2_2l2x, eris.ovvo)

            for y, cy in enumerate(self.base.clusters):
                if (cy == self) or (cy == cx):
                    continue
                # These should contribute to L
                # (we can neglect the interchange x <-> y, since both x and y are unrestricted (except x != y)
                Yo, Yv = get_projectors(cy.indices)
                T2_2lxy = (project_T2(Lo, Lo, Xv, Yv)
                         + project_T2(Lo, Xo, Lv, Yv)
                         + project_T2(Lo, Xo, Yv, Lv)
                         + project_T2(Xo, Lo, Lv, Yv)
                         + project_T2(Xo, Lo, Yv, Lv)
                         + project_T2(Xo, Yo, Lv, Lv))
                e2 += 2*einsum('ijab,iabj', T2_2lxy, eris.ovvo)
                e2 -=   einsum('ijab,jabi', T2_2lxy, eris.ovvo)

                for z, cz in enumerate(self.base.clusters):
                    if (cz == self) or (cz == cx) or (cz == cy):
                        continue
                    # We can neglect interchange between x, y, z (see above)
                    Zo, Zv = get_projectors(cz.indices)
                    T2_lxyz = 0.25*(project_T2(Lo, Xo, Yv, Zv)
                                  + project_T2(Xo, Lo, Yv, Zv)
                                  + project_T2(Xo, Yo, Lv, Zv)
                                  + project_T2(Xo, Yo, Zv, Lv))
                    e2 += 2*einsum('ijab,iabj', T2_lxyz, eris.ovvo)
                    e2 -=   einsum('ijab,jabi', T2_lxyz, eris.ovvo)

        e_loc = e2

        return e_loc

    def get_local_energy_new(self, cc, C1, C2):

        a = cc.get_frozen_mask()
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]

        # Project one index of amplitudes
        l = self.indices
        r = self.rest
        P = np.linalg.multi_dot((C[:,o].T, S[:,l], C[l][:,o]))
        #S_121 = self.make_projector_s121()
        #P = np.linalg.multi_dot((C_cc[:,o].T, S_121, C_cc[:,o]))
        C1 = einsum("xi,ia->xa", P, C1)
        C2 = einsum("xi,ijab->xjab", P, C2)

        eris = cc.ao2mo()
        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * C1)
        if abs(e1) > 1e-7:
            log.warning("Warning: large E1 component: %.8e" % e1)

        e2 = 2*einsum('ijab,iabj', C2, eris.ovvo)
        e2 -=  einsum('ijab,jabi', C2, eris.ovvo)

        e_loc = e1 + e2

        return e_loc

    def get_local_energy(self, cc, projector="occ", pertT=False):

        a = cc.get_frozen_mask()
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]
        CTS = np.dot(C.T, S)

        # Project one index of T amplitudes
        l = self.indices
        r = self.rest
        if projector == "occ":
            Lo = np.linalg.multi_dot((C[:,o].T, S[:,l], C[l][:,o]))
            #S_121 = self.make_projector_s121()
            #P = np.linalg.multi_dot((C_cc[:,o].T, S_121, C_cc[:,o]))
            T1 = einsum("xi,ia->xa", Lo, cc.t1)
            T2 = einsum("xi,ijab->xjab", Lo, cc.t2)
            T21 = T2 + einsum('xa,jb->xjab', T1, cc.t1)
        elif projector == "vir":
            Lv = np.linalg.multi_dot((C[:,v].T, S[:,l], C[l][:,v]))
            T1 = einsum("xa,ia->ix", Lv, cc.t1)
            T2 = einsum("xa,ijab->ijxb", Lv, cc.t2)
            T21 = T2 + einsum('ix,jb->ijxb', T1, cc.t1)
        elif projector == "occ-2":
            Lo = np.linalg.multi_dot((CTS[o][:,l], C[l][:,o]))
            #Lv = np.linalg.multi_dot((CTS[v][:,l], C[l][:,v]))
            Ro = np.linalg.multi_dot((CTS[o][:,r], C[r][:,o]))
            #Rv = np.linalg.multi_dot((CTS[v][:,r], C[r][:,v]))

            T1 = einsum("pi,ia->pa", Lo, cc.t1)

            tau = cc.t2 + einsum("ia,jb->ijab", cc.t1, cc.t1)
            T2_ll = einsum("pi,qj,ijab->pqab", Lo, Lo, tau)
            T2_lr = einsum("pi,qj,ijab->pqab", Lo, Ro, tau)
            T2_rl = einsum("pi,qj,ijab->pqab", Ro, Lo, tau)
            T21 = T2_ll + (T2_lr + T2_rl)/2


        #elif projector == "weighted":
        #    #symmetrize= True
        #    # MF occ of AOs (population analysis)
        #    occ = np.einsum("ab,ba->a", self.mf.make_rdm1(), S) / 2
        #    # This may not be true?
        #    #assert np.all(0 < occ)
        #    #assert np.all(occ < 1)
        #    for idx, ao in enumerate(self.mol.ao_labels()):
        #        log.debug("%s: %f", ao, occ[idx])

        #    norm = sum(occ)
        #    log.debug("sum: %.8f" % sum(occ))
        #    occ = np.clip(occ, 0, 1)
        #    log.debug("sum: %.8f" % sum(occ))
        #    occ *= norm / sum(occ)
        #    log.debug("sum: %.8f" % sum(occ))

        #    w = occ[l]
        #    Po = np.einsum("ai,ab,b,bj->ij", C[:,o], S[:,l], w, C[l][:,o])
        #    Pv = np.einsum("ai,ab,b,bj->ij", C[:,v], S[:,l], (1-w), C[l][:,v])
        #    if symmetrize:
        #        Po = (Po + Po.T)/2
        #        Pv = (Pv + Pv.T)/2

        #    T1o = np.einsum("xi,ia->xa", Po, cc.t1, optimize=True)
        #    T1v = np.einsum("xa,ia->ix", Pv, cc.t1, optimize=True)
        #    T1 = T1o + T1v

        #    T2 = (np.einsum("xi,ijab->xjab", Po, cc.t2, optimize=True)
        #        + np.einsum("xa,ijab->ijxb", Pv, cc.t2, optimize=True))
        #    T21 = T2 + (np.einsum('xa,jb->xjab', T1o, cc.t1, optimize=True)
        #              + np.einsum('ix,jb->ijxb', T1v, cc.t1, optimize=True))

        #elif projector == "weighted-inv":
        #    #symmetrize= True
        #    # MF occ of AOs (population analysis)
        #    occ = np.einsum("ab,ba->a", self.mf.make_rdm1(), S) / 2
        #    # This may not be true?
        #    #assert np.all(0 < occ)
        #    #assert np.all(occ < 1)
        #    for idx, ao in enumerate(self.mol.ao_labels()):
        #        log.debug("%s: %f", ao, occ[idx])

        #    norm = sum(occ)
        #    log.debug("sum: %.8f" % sum(occ))
        #    occ = np.clip(occ, 0, 1)
        #    log.debug("sum: %.8f" % sum(occ))
        #    occ *= norm / sum(occ)
        #    log.debug("sum: %.8f" % sum(occ))

        #    w = occ[l]
        #    Po = np.einsum("ai,ab,b,bj->ij", C[:,o], S[:,l], (1-w), C[l][:,o])
        #    Pv = np.einsum("ai,ab,b,bj->ij", C[:,v], S[:,l], w, C[l][:,v])
        #    if symmetrize:
        #        Po = (Po + Po.T)/2
        #        Pv = (Pv + Pv.T)/2

        #    T1o = np.einsum("xi,ia->xa", Po, cc.t1, optimize=True)
        #    T1v = np.einsum("xa,ia->ix", Pv, cc.t1, optimize=True)
        #    T1 = T1o + T1v

        #    T2 = (np.einsum("xi,ijab->xjab", Po, cc.t2, optimize=True)
        #        + np.einsum("xa,ijab->ijxb", Pv, cc.t2, optimize=True))
        #    T21 = T2 + (np.einsum('xa,jb->xjab', T1o, cc.t1, optimize=True)
        #              + np.einsum('ix,jb->ijxb', T1v, cc.t1, optimize=True))

        else:
            raise ValueError()

        # Energy
        eris = cc.ao2mo()
        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * T1)
        #assert np.isclose(e1, 0)
        if not np.isclose(e1, 0):
            log.warning("Warning: large E1 component: %.8e" % e1)

        #e2 = 2*np.einsum('xjab,xabj', T21, eris.ovvo, optimize=True)
        #e2 -=  np.einsum('xjab,jabx', T21, eris.ovvo, optimize=True)
        e2 = 2*einsum('ijab,iabj', T21, eris.ovvo)
        e2 -=  einsum('ijab,jabi', T21, eris.ovvo)

        e_loc = e1 + e2

        if pertT:
            raise NotImplementedError()
            T1 = np.ascontiguousarray(T1)
            T2 = np.ascontiguousarray(T2)
            e_pertT = cc.ccsd_t(T1, T2, eris)
        else:
            e_pertT = 0

        return e_loc, e_pertT

# ===== #

class EmbCC:

    def __init__(self, mf, solver="CCSD", bath_type=None, tol_bath=1e-3, tol_dmet_bath=1e-8):
        self.mf = mf
        self.mol = mf.mol

        if solver not in ("CCSD", "CISD", "FCI"):
            raise ValueError()
        if bath_type not in (None, "matsubara", "uncontracted"))
            raise ValueError()
        self.solver = solver
        self.bath_type = bath_type
        self.tol_bath = tol_bath
        self.tol_dmet_bath = tol_dmet_bath

        self.clusters = []

    def make_cluster(self, name, ao_indices, **kwargs):
        kwargs["solver"] = kwargs.get("solver", self.solver)
        kwargs["bath_type"] = kwargs.get("bath_type", self.bath_type)
        kwargs["tol_bath"] = kwargs.get("tol_bath", self.tol_bath)
        kwargs["tol_dmet_bath"] = kwargs.get("tol_dmet_bath", self.tol_dmet_bath)

        cluster = Cluster(self, name, ao_indices, **kwargs)
        return cluster

    def make_atom_clusters(self):
        """Divide atomic orbitals into clusters according to their base atom."""

        # base atom for each AO
        base_atoms = np.asarray([ao[0] for ao in self.mol.ao_labels(None)])

        self.clear_clusters()
        ncluster = self.mol.natm
        for atomid in range(ncluster):
            ao_indices = np.nonzero(base_atoms == atomid)[0]
            name = self.mol.atom_symbol(atomid)
            c = self.make_cluster(name, ao_indices)
            self.clusters.append(c)
        return self.clusters

    def make_rest_cluster(self, name, **kwargs):
        """Combine all AOs which are not part of a cluster, into a rest cluster."""

        rest_aos = list(range(self.mol.nao_nr()))
        for c in self.clusters:
            rest_aos = [i for i in rest_aos if i not in c.indices]
        c = self.make_cluster(name, ao_indices, **kwargs)
        self.clusters.append(c)
        return c

    def make_custom_cluster(self, ao_symbols, name=None):
        """Make custom clusters in terms of AOs.

        Parameters
        ----------
        ao_symbols : iterable
            List of atomic orbital symbols for cluster.
        """
        if isinstance(ao_symbols, str):
            ao_symbols = [ao_symbols]

        if name is None:
            name = ",".join(ao_symbols)

        ao_indices = []
        for ao_idx, ao_label in enumerate(self.mol.ao_labels()):
            for ao_symbol in ao_symbols:
                if ao_symbol in ao_label:
                    log.debug("AO symbol %s found in %s", ao_symbol, ao_label)
                    ao_indices.append(ao_idx)
                    break
        c = self.make_cluster(name, ao_indices)
        self.clusters.append(c)
        return c

    def make_custom_atom_cluster(self, atoms, name=None):
        """Make custom clusters in terms of atoms..

        Parameters
        ----------
        atoms : iterable
            List of atom symbols for cluster.
        """

        if name is None:
            name = ",".join(atoms)
        # base atom for each AO
        ao2atomlbl = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
        ao_indices = np.nonzero(np.isin(ao2atomlbl, atoms))[0]
        c = self.make_cluster(name, ao_indices)
        self.clusters.append(c)
        return c

    def merge_clusters(self, clusters, name=None, **kwargs):
        """Attributes solver, bath_type, tol_bath, and tol_dmet_bath will be taken from first cluster,
        unless specified in **kwargs.
        name will be auto generated, unless specified.

        Parameters
        ----------
        clusters : iterable
            List of clusters to merge.
        """
        clusters_out = []
        merged = []
        for c in self.clusters:
            if c.name in clusters:
                merged.append(c)
            else:
                clusters_out.append()

        if len(merged) < 2:
            raise ValueError("Not enough clusters (%d) found to merge." % len(merged))

        if name is None:
            name = "+".join([c.name for c in merged])
        merged_indices = np.hstack([c.indices for c in merged])
        kwargs["solver"] = kwargs.get("solver", merged[0].solver)
        kwargs["bath_type"] = kwargs.get("bath_type", merged[0].bath_type)
        kwargs["tol_bath"] = kwargs.get("tol_bath", merged[0].tol_bath)
        kwargs["tol_dmet_bath"] = kwargs.get("tol_dmet_bath", merged[0].tol_dmet_bath)
        #merged_cluster = Cluster(merged_name, self.mf, merged_indices,
        #        tol_dmet_bath=tol_dmet_bath, tol_bath=tol_bath)
        c = self.make_cluster(name, ao_indices, **kwargs)
        clusters_out.append(c)
        self.clusters = clusters_out
        return c

    def clear_clusters(self):
        """Clear all previously defined clusters."""
        self.clusters = []

    def print_clusters(self, clusters=None, file=None):
        """Print clusters to logging or file."""
        if clusters is None:
            clusters = self.clusters

        ao_labels = self.mol.ao_labels(None)

        end = "\n" if file else ""
        headfmt = "Cluster %3d: %s with %3d local orbitals:" + end
        linefmt = "%4d %5s %3s %10s" + end

        if file is None:
            for cidx, c in enumerate(clusters):
                log.info(headfmt, cidx, c.name, len(c))
                for ao in c.indices:
                    log.info(linefmt, *ao_labels[ao])
        else:
            with open(file, "w") as f:
                for cidx, c in enumerate(clusters):
                    f.write(headfmt % (cidx, c.name, len(c)))
                    for ao in c.indices:
                        f.write(linefmt % ao_labels[ao])

    def get_cluster(self, name):
        for c in self.clusters:
            if c.name == name:
                return c
        else:
            raise ValueError()

    def run(self, clusters=None, max_power=0, pertT=False):
        if pertT:
            raise NotImplementedError("Perturbative triplet correction currently not implemented.")
        if clusters is None:
            clusters = self.clusters
        if not clusters:
            raise ValueError("No clusters defined for EmbCC calculation.")

        MPI_comm.Barrier()
        t_start = MPI.Wtime()

        for idx, c in enumerate(clusters):
            if MPI_rank != (idx % MPI_size):
                continue
            log.debug("Running cluster %s on rank %d", c.name, MPI_rank)
            c.run_solver(max_power=max_power, pertT=pertT)
            log.debug("Cluster %s on rank %d is done.", c.name, MPI_rank)

        all_conv = self.collect_results()

        MPI_comm.Barrier()
        wtime = MPI.Wtime() - t_start
        log.info("Total wall time for EmbCCSD: %.0f min %.2g s", *divmod(wtime, 60.0))

        return all_conv

    def collect_results(self):
        log.debug("Communicating results.")
        clusters = self.clusters

        # Communicate
        def mpi_reduce(attribute, op=MPI.SUM, root=0):
            res = MPI_comm.reduce(np.asarray([getattr(c, attribute) for c in clusters]), op=op, root=root)
            return res

        converged = mpi_reduce("converged", op=MPI.PROD)
        nbath0 = mpi_reduce("nbath0")
        nbath = mpi_reduce("nbath")
        nfrozen = mpi_reduce("nfrozen")
        e_cl_ccsd = mpi_reduce("e_cl_ccsd")

        e_corr = mpi_reduce("e_corr")
        #e_ccsd = mpi_reduce("e_ccsd")
        #e_pt = mpi_reduce("e_pt")

        e_ccsd_v = mpi_reduce("e_ccsd_v")
        e_ccsd_w = mpi_reduce("e_ccsd_w")
        e_ccsd_z = mpi_reduce("e_ccsd_z")

        #converged = MPI_comm.reduce(np.asarray([c.converged for c in clusters]), op=MPI.PROD, root=0)
        #nbath0 = MPI_comm.reduce(np.asarray([c.nbath0 for c in clusters]), op=MPI.SUM, root=0)
        #nbath = MPI_comm.reduce(np.asarray([c.nbath for c in clusters]), op=MPI.SUM, root=0)
        #nfrozen = MPI_comm.reduce(np.asarray([c.nfrozen for c in clusters]), op=MPI.SUM, root=0)
        #e_cl_ccsd = MPI_comm.reduce(np.asarray([c.e_cl_ccsd for c in clusters]), op=MPI.SUM, root=0)
        #e_ccsd = MPI_comm.reduce(np.asarray([c.e_ccsd for c in clusters]), op=MPI.SUM, root=0)
        #e_pt = MPI_comm.reduce(np.asarray([c.e_pt for c in clusters]), op=MPI.SUM, root=0)

        #e_ccsd_v = MPI_comm.reduce(np.asarray([c.e_ccsd_v for c in clusters]), op=MPI.SUM, root=0)
        #e_ccsd_w = MPI_comm.reduce(np.asarray([c.e_ccsd_w for c in clusters]), op=MPI.SUM, root=0)
        #e_ccsd_z = MPI_comm.reduce(np.asarray([c.e_ccsd_z for c in clusters]), op=MPI.SUM, root=0)

        if MPI_rank == 0:
            for cidx, c in enumerate(self.clusters):
                c.converged = converged[cidx]
                c.nbath0 = nbath0[cidx]
                c.nbath = nbath[cidx]
                c.nfrozen = nfrozen[cidx]
                c.e_cl_ccsd = e_cl_ccsd[cidx]
                #c.e_ccsd = e_ccsd[cidx]
                c.e_corr = e_corr[cidx]
                #c.e_pt = e_pt[cidx]

                c.e_ccsd_v = e_ccsd_v[cidx]
                c.e_ccsd_w = e_ccsd_w[cidx]
                c.e_ccsd_z = e_ccsd_z[cidx]

            #self.e_ccsd = sum(e_ccsd)
            self.e_corr = sum(e_corr)
            #self.e_pt = sum(e_pt)

            #self.e_corr = self.e_ccsd + self.e_pt
            self.e_tot = self.mf.e_tot + self.e_corr

            self.e_ccsd_v = sum(e_ccsd_v)
            self.e_ccsd_w = sum(e_ccsd_w)
            self.e_ccsd_z = sum(e_ccsd_z)

        #if MPI_rank == 0:
        #    log.info("Local correlation energy contributions")
        #    log.info("--------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd, self.e_pt)


        #    log.info("Local correlation energy contributions (virtual)")
        #    log.info("------------------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd_v, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd_v, 0.0)


        #    log.info("Local correlation energy contributions (symmetrized)")
        #    log.info("------------------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd_w, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd_w, 0.0)


        #    log.info("Local correlation energy contributions (symmetrized, virtual)")
        #    log.info("------------------------------------------------")
        #    fmtstr_c = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    fmtstr_t = "%10s                           : CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
        #    for c in clusters:
        #        log.info(fmtstr_c, c.name, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_ccsd_z, c.e_pt)
        #    log.info(fmtstr_t, "Total", self.e_ccsd_z, 0.0)


        #    log.info("Full cluster CCSD energies")
        #    log.info("--------------------------")
        #    for c in clusters:
        #        log.info("%10s: CCSD=%+16.8g Eh", c.name, c.e_cl_ccsd)

        #    log.debug("Communicating results done.")

        return np.all(converged)

    def print_cluster_results(self):
        log.info("Energy contributions per cluster")
        log.info("------------------------0-------")
        # Name solver nactive (local, dmet bath, add bath) nfrozen E_corr_full E_corr
        linefmt = "%10s  %6s  %3d (%3d,%3d,%3d)  %3d: Full=%16.8g Eh Local=%16.8 Eh"
        totalfmt = "Total=%16.8 Eh"
        for c in clusters:
            log.info(linefmt, c.name, c.solver, len(c)+c.nbath, len(c), c.nbath0, c.nbath-c.nbath0, c.nfrozen, c.e_corr_full, c.e_corr)
        log.info(totalfmt, self.e_corr)
