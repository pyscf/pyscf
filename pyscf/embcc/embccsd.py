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

    def __init__(self, name, mf, indices):
        """
        Parameters
        ----------
        mf :
            Mean-field object.
        name :
            Name of cluster.
        indices:
            Atomic orbital indices of cluster.
        """

        self.name = name
        self.mf = mf
        self.indices = indices

        self.mol = mf.mol

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

        #print("Power %d kind %s eigenvalues:\n%s" % (power, kind, e))

        nbath = sum(e >= tol)

        C = C.copy()
        C[:,non_local] = np.dot(C[:,non_local], v)

        return C, nbath

    def run_ccsd(self, max_power=0, pertT=False, diagonalize_fock=True):
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

        print("Running CCSD with %3d local, %3d DMET bath, %3d other bath and %3d frozen orbitals" % (len(self), nbath0, nbath-nbath0, len(frozen)))
        cc = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)

        #cc.verbose = 4
        cc.kernel()
        assert cc.converged
        assert np.allclose(C_cc, cc.mo_coeff)

        e_loc, e_pertT = self.get_local_energy(cc, pertT=pertT)

        return e_loc, e_pertT

    def get_local_energy(self, cc, pertT=False):

        a = cc.get_frozen_mask()
        l = self.indices
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C = cc.mo_coeff[:,a]

        var = "pv2"
        # Project amplitudes
        if var == "po1":
            P = np.linalg.multi_dot((C[:,o].T, S[:,l], C[l][:,o]))
            T1 = np.einsum("xi,ia->xa", P, cc.t1, optimize=True)
            # T2
            T2 = np.einsum("xi,ijab->xjab", P, cc.t2, optimize=True)
            # Add connected T1
            T21 = T2 + np.einsum('xa,jb->xjab', T1, cc.t1, optimize=True)
        elif var == "pv1":
            Pv = np.linalg.multi_dot((C[:,v].T, S[:,l], C[l][:,v]))

            T1 = np.einsum("xa,ia->ix", Pv, cc.t1, optimize=True)
            # T2
            T2 = np.einsum("xa,ijab->ijxb", Pv, cc.t2, optimize=True)
            # Add connected T1
            T21 = T2 + np.einsum('ix,jb->ijxb', T1, cc.t1, optimize=True)

        elif var == "pv2":
            Pv = np.linalg.multi_dot((C[:,v].T, S[:,l], C[l][:,v])).T

            T1 = np.einsum("xa,ia->ix", Pv, cc.t1, optimize=True)
            # T2
            T2 = np.einsum("xb,ijab->ijax", Pv, cc.t2, optimize=True)
            # Add connected T1
            T21 = T2 + np.einsum('ia,jx->ijax', cc.t1, T1, optimize=True)


        # Energy
        eris = cc.ao2mo()
        F = eris.fock[o][:,v]
        e1 = 2*np.sum(F * T1)
        e2 = 2*np.einsum('xjab,xabj', T21, eris.ovvo, optimize=True)
        e2 -=  np.einsum('xjab,jabx', T21, eris.ovvo, optimize=True)

        e_loc = e1 + e2

        if pertT:
            T1 = np.ascontiguousarray(T1)
            T2 = np.ascontiguousarray(T2)
            e_pertT = cc.ccsd_t(T1, T2, eris)
        else:
            e_pertT = 0

        return e_loc, e_pertT

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
            name = self.mol.atom_symbol(atomid)
            c = Cluster(name, self.mf, ao_indices)
            clusters.append(c)

        return clusters

    def run(self, max_power=0, pertT=False, cluster=None):
        clusters = self.create_atom_clusters()

        MPI_comm.Barrier()
        t_start = MPI.Wtime()

        ecl_sd = np.zeros((len(clusters),))
        ecl_pt = np.zeros((len(clusters),))
        for idx, c in enumerate(clusters):
            if cluster is not None and idx not in cluster:
                continue
            if MPI_rank != (idx % MPI_size):
                continue
            ecl_sd[idx], ecl_pt[idx] = c.run_ccsd(max_power=max_power, pertT=pertT)

        if MPI_rank == 0:
            e_sd = np.zeros_like(ecl_sd)
            e_pt = np.zeros_like(ecl_pt)
        else:
            e_sd = None
            e_pt = None

        #MPI_comm.Reduce([e_cluster, MPI.DOUBLE], [e_corr, MPI.DOUBLE], op=MPI.SUM, root=0)
        MPI_comm.Reduce([ecl_sd, MPI.DOUBLE], [e_sd, MPI.DOUBLE], op=MPI.SUM, root=0)
        MPI_comm.Reduce([ecl_pt, MPI.DOUBLE], [e_pt, MPI.DOUBLE], op=MPI.SUM, root=0)

        MPI_comm.Barrier()
        wtime = MPI.Wtime() - t_start
        if MPI_rank == 0:
            print("Wall time: %.2g s" % wtime)

        if MPI_rank == 0:
            for cidx, cluster in enumerate(clusters):
                print("%10s: CCSD=%+16.8g Eh  (T)=%+16.8g Eh" % (cluster.name, e_sd[cidx], e_pt[cidx]))

            e_ccsd = sum(e_sd)
            e_pt = sum(e_pt)
            e_ccsdpt = e_ccsd + e_pt

            print("%10s: CCSD=%+16.8g Eh  (T)=%+16.8g Eh" % ("Total", e_ccsd, e_pt))

            self.e_corr = e_ccsdpt

        return e_ccsd, e_pt

if __name__ == "__main__":
    import pyscf
    import pyscf.gto
    import pyscf.scf

    from molecules import *

    eps = 1e-14
    dists = np.arange(0.8, 4.0+eps, 0.2)
    #dists = np.linspace(0, 2*np.pi, num=10, endpoint=False)
    #dists = np.linspace(0, 180, num=30, endpoint=False)

    #basis = "sto-3g"
    #basis = "tzvp"
    basis = "cc-pVDZ"
    output = "output.txt"

    pt = False

    for d in dists:
        if MPI_rank == 0:
            print("Now calculating distance=%.3f" % d)

        #mol = build_dimer(["N", "N"], d, basis=basis)
        mol = build_EtOH(d, basis=basis)
        #mol = build_biphenyl(d, basis=basis)

        try:
            mf = pyscf.scf.RHF(mol)
            mf.kernel()
        except Exception as e:
            print(e)
            continue

        if MPI_rank == 0:
            print("Full space CCSD")
        cc = pyscf.cc.CCSD(mf)
        try:
            cc.kernel()
            e_ccsd_ref = cc.e_corr
            assert cc.converged
            if pt:
                e_pt_ref = cc.ccsd_t()
        except Exception as e:
            print(e)
            e_cc = -1
        if MPI_rank == 0:
            print("Done")

        ecc = EmbCCSD(mf)
        #try:
        ecc_sd, ecc_pt = ecc.run(max_power=2, pertT=pt)
        #except Exception as e:
        #    print(e)
        #    e_ecc = -1

        if MPI_rank == 0:
            print("CCSD correlation energy:    %+g" % e_ccsd_ref)
            print("EmbCCSD correlation energy: %+g" % ecc_sd)
            print("Error:                      %+g" % (ecc_sd - e_ccsd_ref))
            print("%% Correlation:             %.3f %%" % (100*ecc_sd/e_ccsd_ref))

            if pt:
                print("CCSD(T) correlation energy:    %+g" % (e_ccsd_ref + e_pt_ref))
                print("EmbCCSD(T) correlation energy: %+g" % (ecc_sd + ecc_pt))
                print("Error:                         %+g" % (ecc_sd+ecc_pt - e_ccsd_ref-e_pt_ref))
                print("%% Correlation:                %.3f %%" % (100*(ecc_sd+ecc_pt)/(e_ccsd_ref+e_pt_ref)))

            if pt:
                with open(output, "a") as f:
                    f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (d, mf.e_tot, mf.e_tot+e_ccsd_ref, mf.e_tot+e_ccsd_ref+e_pt_ref, mf.e_tot+ecc_sd, mf.e_tot+ecc_sd+ecc_pt))
            else:
                with open(output, "a") as f:
                    f.write("%3f  %.8e  %.8e  %.8e\n" % (d, mf.e_tot, mf.e_tot+e_ccsd_ref, mf.e_tot+ecc_sd))
