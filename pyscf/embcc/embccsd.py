import logging

import numpy as np
import scipy
import scipy.linalg
from mpi4py import MPI

import pyscf
import pyscf.lo
import pyscf.cc

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

__all__ = [
        "EmbCCSD",
        ]

class Cluster:

    def __init__(self, mf, indices):
        """
        Parameters
        ----------
        mf :
            Mean-field object.
        indices:
            Atomic orbital indices of cluster.
        """

        self.mf = mf
        self.mol = mf.mol
        self.indices = indices

    def __len__(self):
        return len(self.indices)

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

    def make_local_orbitals(self, tol=1e-9):
        S = self.mf.get_ovlp()
        #S_inv = np.linalg.inv(S)
        C = self.mf.mo_coeff
        S_inv = np.dot(C, C.T)
        P = self.make_projector()

        D_loc = np.linalg.multi_dot((P, S_inv, P.T))
        C = self.mf.mo_coeff.copy()
        SC = np.dot(S, C)

        # Transform to C
        D_loc = np.linalg.multi_dot((SC.T, D_loc, SC))
        e, r = np.linalg.eigh(D_loc)
        rev = np.s_[::-1]
        e = e[rev]
        r = r[:,rev]

        nloc = len(e[e>tol])
        assert nloc == len(self), "Error finding local orbitals: %s" % e
        #C_loc = np.dot(C, r[:,:nimp])
        C = np.dot(C, r)

        return C

    def make_dmet_bath_orbitals(self, C, tol=1e-8):
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

        #print("DMET bath eigenvalues:\n%s\nFollowing eigenvalues:\n%s" % (e[:nbath], e[nbath:nbath+3]))

        assert nbath0 <= len(self)

        C[:,env] = np.dot(C[:,env], v)

        return C, nbath0, nenvocc

    def make_power_bath_orbitals(self, C, kind, non_local, power=1, tol=1e-8, normalize=False):

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

        print("Power %d kind %s eigenvalues:\n%s" % (power, kind, e))

        nbath = sum(e >= tol)

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def run_ccsd(self, max_power=0, diagonalize_fock=True):
        C = self.make_local_orbitals()
        C, nbath0, nenvocc = self.make_dmet_bath_orbitals(C)
        nbath = nbath0

        # Add additional power bath orbitals
        nbathpocc = 0
        nbathpvir = 0
        for power in range(1, max_power+1):
            occ_space = np.s_[len(self)+nbath0+nbathpocc:len(self)+nbath0+nenvocc]
            C, nbo = self.make_power_bath_orbitals(C, "occ", occ_space, power=power)
            vir_space = np.s_[len(self)+nbath0+nenvocc+nbathpvir:]
            C, nbv = self.make_power_bath_orbitals(C, "vir", vir_space, power=power)
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
        nactive = len(self) + nbath
        frozen = rank[nactive:]
        active = rank[:nactive]

        # Accelerates convergence
        if diagonalize_fock:
            F = np.linalg.multi_dot((C_cc.T, mf.get_fock(), C_cc))
            # Occupied
            o = np.nonzero(occ > 0)[0]
            o = np.asarray([i for i in o if i in active])
            e, r = np.linalg.eigh(F[np.ix_(o, o)])
            C_cc[:,o] = np.dot(C_cc[:,o], r)
            # Virtual
            v = np.nonzero(occ == 0)[0]
            v = np.asarray([i for i in v if i in active])
            e, r = np.linalg.eigh(F[np.ix_(v, v)])
            C_cc[:,v] = np.dot(C_cc[:,v], r)

        #print("Running CCSD with %3d local, %3d DMET bath, %3d other bath and %3d frozen orbitals" % (len(self), nbath0, nbath-nbath0, len(frozen)))
        cc = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)

        #cc.verbose = 4
        cc.kernel()
        assert cc.converged
        assert np.allclose(C_cc, cc.mo_coeff)

        e_loc = self.get_local_energy(cc)

        return e_loc

    def get_local_energy(self, cc):

        occ = cc.mo_occ
        eris = cc.ao2mo()
        S = self.mf.get_ovlp()

        a = cc.get_frozen_mask()

        C = cc.mo_coeff[:,a]
        SC = np.dot(S, C)

        l = self.indices
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0

        # Transform amplitudes
        T1 = np.einsum("xi,ia,ya->xy", C[:,o], cc.t1, C[:,v], optimize=True)
        # T2
        T21 = np.einsum("xi,yj,ijab,za,wb->xyzw", C[l][:,o], C[:,o], cc.t2, C[:,v], C[:,v], optimize=True)
        # Add connected T1
        T21 += np.einsum('xz,yw->xyzw', T1[l], T1, optimize=True)

        # Energy
        F = np.linalg.multi_dot((SC[:,o][l], eris.fock[o][:,v], SC[:,v].T))
        e1 = 2*np.sum(F * T1[l])
        eris_ovvo = np.einsum("xi,ya,iabj,zb,wj->xyzw",
                SC[:,o], SC[:,v], eris.ovvo, SC[:,v], SC[:,o], optimize=True)
        e2 = 2*np.einsum('ijab,iabj', T21, eris_ovvo[l], optimize=True)
        e2 -=  np.einsum('jiab,iabj', T21, eris_ovvo[:,:,:,l], optimize=True)

        e_loc = e1 + e2
        return e_loc

    #def make_local_orbitals_2(self):
    #    C = self.mf.mo_coeff[self.indices]
    #    C = pyscf.lo.vec_lowdin(C, self.S)
    #    return C


class EmbCCSD:

    def __init__(self, mf):
        self.mf = mf
        self.mol = mf.mol

    def create_atom_clusters(self):
        """Divide atomic orbitals into clusters."""

        # base atom for each AO
        base_atoms = np.asarray([ao[0] for ao in self.mol.ao_labels(None)])

        clusters = []
        for atomid in range(self.mol.natm):
            ao_indices = np.nonzero(base_atoms == atomid)[0]
            c = Cluster(self.mf, ao_indices)
            clusters.append(c)

        return clusters

    def run(self, max_power=0, cluster=None):
        clusters = self.create_atom_clusters()

        MPI_comm.Barrier()
        t_start = MPI.Wtime()

        e_cluster = np.zeros((len(clusters),))
        for idx, c in enumerate(clusters):
            if cluster is not None and idx not in cluster:
                continue
            if MPI_rank != (idx % MPI_size):
                continue
            e_cluster[idx] = c.run_ccsd(max_power=max_power)

        if MPI_rank == 0:
            e_corr = np.zeros((len(clusters),))
        else:
            e_corr = None

        MPI_comm.Reduce([e_cluster, MPI.DOUBLE], [e_corr, MPI.DOUBLE], op=MPI.SUM, root=0)

        MPI_comm.Barrier()
        wtime = MPI.Wtime() - t_start
        if MPI_rank == 0:
            print("Wall time: %.2g s" % wtime)

        if MPI_rank == 0:
            e_corr = sum(e_corr)
            self.e_corr = e_corr

        return e_corr

if __name__ == "__main__":
    import pyscf
    import pyscf.gto
    import pyscf.scf

    from molecules import *

    eps = 1e-14
    #dists = np.arange(0.8, 4.0+eps, 0.2)
    #dists = np.linspace(0, 2*np.pi, num=10, endpoint=False)
    dists = np.linspace(0, 360, num=10, endpoint=False)

    basis = "cc-pVDZ"
    output = "energies.txt"

    for d in dists:
        if MPI_rank == 0:
            print("Now calculating distance=%.3f" % d)

        #mol = build_EtOH(d, basis=basis)
        mol = build_biphenyl(d, basis=basis)

        mf = pyscf.scf.RHF(mol)
        mf.kernel()

        with open(output, "a") as f:
            f.write("%3f  %.8e\n" % (d, mf.e_tot))

        continue

        if MPI_rank == 0:
            print("Full space CCSD")
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        assert cc.converged
        if MPI_rank == 0:
            print("Done")

        ecc = EmbCCSD(mf)
        ecc.run(max_power=2)
        #ecc.run(max_power=2, cluster=[0])
        #ecc.e_corr *= natom

        if MPI_rank == 0:
            print("CCSD correlation energy:    %+g" % cc.e_corr)
            print("EmbCCSD correlation energy: %+g" % ecc.e_corr)
            print("Error:                      %+g" % (ecc.e_corr - cc.e_corr))
            print("%% Correlation:             %.3f %%" % (100*ecc.e_corr/cc.e_corr))

            with open(output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e\n" % (d, mf.e_tot, cc.e_tot, mf.e_tot + ecc.e_corr))
