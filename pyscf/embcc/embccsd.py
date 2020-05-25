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

        # Output
        self.converged = True
        self.e_ccsd = 0
        self.e_pt = 0
        self.nbath0 = 0
        self.nbath = 0
        self.nfrozen = 0

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

        #print("Running CCSD with %3d local, %3d DMET bath, %3d other bath and %3d frozen orbitals" % (len(self), nbath0, nbath-nbath0, len(frozen)))
        cc = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ, frozen=frozen)
        cc.max_cycle = 100
        #cc.verbose = 4
        cc.kernel()

        self.converged = cc.converged
        self.nbath0 = nbath0
        self.nbath = nbath
        self.nfrozen = len(frozen)

        self.e_ccsd, self.e_pt = self.get_local_energy(cc, C, pertT=pertT)

        return int(self.converged)

    def get_local_energy(self, cc, C, pertT=False):

        a = cc.get_frozen_mask()
        #nactive = sum(a)
        #print("active orbitals: %d" % nactive)
        l = self.indices
        o = cc.mo_occ[a] > 0
        v = cc.mo_occ[a] == 0
        # Projector to local, occupied region
        S = self.mf.get_ovlp()
        C_cc = cc.mo_coeff[:,a]
        #C = C[:,np.s_[:nactive]]

        var = "po"
        if var == "po":
            #P = np.linalg.multi_dot((C_cc[:,o].T, S[:,l], C_cc[l][:,o]))
            #P = (P + P.T)/2

            S_121 = self.make_projector_s121()
            P = np.linalg.multi_dot((C_cc[:,o].T, S_121, C_cc[:,o]))


            T1 = np.einsum("xi,ia->xa", P, cc.t1, optimize=True)
            # T2
            T2 = np.einsum("xi,ijab->xjab", P, cc.t2, optimize=True)
            # Add connected T1
            T21 = T2 + np.einsum('xa,jb->xjab', T1, cc.t1, optimize=True)

        elif var == "pv":
            P = np.linalg.multi_dot((C_cc[:,v].T, S[:,l], C_cc[l][:,v]))
            #P = (P + P.T)/2

            T1 = np.einsum("xa,ia->ix", P, cc.t1, optimize=True)
            # T2
            T2 = np.einsum("xa,ijab->ijxb", P, cc.t2, optimize=True)
            # Add connected T1
            T21 = T2 + np.einsum('ix,jb->ijxb', T1, cc.t1, optimize=True)

        elif var == "average":
            P = np.linalg.multi_dot((C_cc.T, S[:,l], C_cc[l]))
            Po = P[np.ix_(o, o)]
            Pv = P[np.ix_(v, v)]

            # Occ
            T1o = np.einsum("xi,ia->xa", Po, cc.t1, optimize=True)
            T2o = np.einsum("xi,ijab->xjab", Po, cc.t2, optimize=True)
            T21o = T2o + np.einsum('xa,jb->xjab', T1o, cc.t1, optimize=True)

            # Vir
            T1v = np.einsum("xa,ia->ix", Pv, cc.t1, optimize=True)
            T2v = np.einsum("xa,ijab->ijxb", Pv, cc.t2, optimize=True)
            T21v = T2v + np.einsum('ix,jb->ijxb', T1v, cc.t1, optimize=True)

            T1 = (T1o + T1v)/2
            T21 = (T21o + T21v)/2

        #elif var == "new-v":
        #    loc = np.s_[:len(self)]
        #    csc = np.linalg.multi_dot((C_cc[:,v].T, S, C[:,loc]))
        #    P = np.dot(csc, csc.T)

        #    S_121 = self.make_projector_s121()
        #    P2 = np.linalg.multi_dot((C_cc[:,v].T, S_121, C_cc[:,v]))
        #    assert np.allclose(P, P2)

        #    T1 = np.einsum("xa,ia->ix", P, cc.t1, optimize=True)
        #    # T2
        #    T2 = np.einsum("xa,ijab->ijxb", P, cc.t2, optimize=True)
        #    # Add connected T1
        #    T21 = T2 + np.einsum('ix,jb->ijxb', T1, cc.t1, optimize=True)

        #elif var == "Po":
        #    S_121 = self.make_projector_s121()

        #    P = np.linalg.multi_dot((C[:,o].T, S_121, C[:,o]))

        #    T1 = np.einsum("xi,ia->xa", P, cc.t1, optimize=True)
        #    # T2
        #    T2 = np.einsum("xi,ijab->xjab", P, cc.t2, optimize=True)
        #    # Add connected T1
        #    T21 = T2 + np.einsum('xa,jb->xjab', T1, cc.t1, optimize=True)

        #elif var == "Pv":
        #    S_121 = self.make_projector_s121()
        #    P = np.linalg.multi_dot((C[:,v].T, S_121, C[:,v]))

        #    T1 = np.einsum("xa,ia->ix", P, cc.t1, optimize=True)
        #    # T2
        #    T2 = np.einsum("xa,ijab->ijxb", P, cc.t2, optimize=True)
        #    # Add connected T1
        #    T21 = T2 + np.einsum('ix,jb->ijxb', T1, cc.t1, optimize=True)

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
        ncluster = self.mol.natm
        for atomid in range(ncluster):
            ao_indices = np.nonzero(base_atoms == atomid)[0]
            name = self.mol.atom_symbol(atomid)
            c = Cluster(name, self.mf, ao_indices)
            clusters.append(c)

        self.clusters = clusters

        return clusters

    def create_custom_clusters(self, clusters):
        """Divide atomic orbitals into clusters."""

        # base atom for each AO
        #ao_labels = [ao for ao in self.mol.ao_labels(None)]
        ao2atomlbl = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])

        # [("H1", "C2"), ("H3", "H4")]
        ncluster = len(clusters)

        clusters_out = []
        for cluster in clusters:
            ao_indices = np.nonzero(np.isin(ao2atomlbl, cluster))[0]
            name = ",".join(cluster)
            c = Cluster(name, self.mf, ao_indices)
            clusters_out.append(c)

        self.clusters = clusters_out

        return clusters_out


    def collect_results(self):
        clusters = self.clusters

        converged = MPI_comm.reduce(np.asarray([c.converged for c in clusters]), op=MPI.PROD, root=0)
        nbath0 = MPI_comm.reduce(np.asarray([c.nbath0 for c in clusters]), op=MPI.SUM, root=0)
        nbath = MPI_comm.reduce(np.asarray([c.nbath for c in clusters]), op=MPI.SUM, root=0)
        nfrozen = MPI_comm.reduce(np.asarray([c.nfrozen for c in clusters]), op=MPI.SUM, root=0)
        e_ccsd = MPI_comm.reduce(np.asarray([c.e_ccsd for c in clusters]), op=MPI.SUM, root=0)
        e_pt = MPI_comm.reduce(np.asarray([c.e_pt for c in clusters]), op=MPI.SUM, root=0)

        if MPI_rank == 0:
            print("Cluster results")
            print("---------------")

            fmtstr = "%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh"
            for i, c in enumerate(clusters):
                print(fmtstr % (c.name, len(c), nbath0[i], nbath[i]-nbath0[i], nfrozen[i], e_ccsd[i], e_pt[i]))

            self.e_ccsd = sum(e_ccsd)
            self.e_pt = sum(e_pt)

            print("%10s:                            CCSD=%+16.8g Eh  (T)=%+16.8g Eh" % ("Total", self.e_ccsd, self.e_pt))

        return np.all(converged)

    def run(self, max_power=0, pertT=False):
        clusters = self.clusters

        MPI_comm.Barrier()
        t_start = MPI.Wtime()

        for idx, c in enumerate(clusters):
            if MPI_rank != (idx % MPI_size):
                continue

            c.run_ccsd(max_power=max_power, pertT=pertT)

        all_conv = self.collect_results()

        MPI_comm.Barrier()
        wtime = MPI.Wtime() - t_start
        if MPI_rank == 0:
            print("Wall time: %.2g s" % wtime)

        return all_conv

        #if MPI_rank == 0:
        #    for cidx, cluster in enumerate(clusters):
        #        print("%10s [N=%3d,B0=%3d,B=%3d,F=%3d]: CCSD=%+16.8g Eh  (T)=%+16.8g Eh" % (cluster.name, *sizes[cidx], e_sd[cidx], e_pt[cidx]))

        #    e_ccsd = sum(e_sd)
        #    e_pt = sum(e_pt)
        #    e_ccsdpt = e_ccsd + e_pt

        #    print("%10s:                            CCSD=%+16.8g Eh  (T)=%+16.8g Eh" % ("Total", e_ccsd, e_pt))
        #    #self.e_corr = e_ccsdpt

        #else:
        #    e_ccsd, e_pt = 0, 0

        #return e_ccsd, e_pt


if __name__ == "__main__":
    import pyscf
    import pyscf.gto
    import pyscf.scf

    from molecules import *

    eps = 1e-14
    dists = np.arange(0.7, 4.0+eps, 0.1)
    #dists = np.linspace(0, 2*np.pi, num=10, endpoint=False)
    #dists = np.linspace(0, 180, num=30, endpoint=False)
    #dists = [2.0]

    #basis = "sto-3g"
    #basis = "tzvp"
    #basis = "cc-pVDZ"
    basis = "cc-pVTZ"
    output = "output.txt"

    pt = False

    max_power = 0
    full_ccsd = False
    emb_ccsd = True

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

        if full_ccsd:
            cc = pyscf.cc.CCSD(mf)
            cc.kernel()
            e_ccsd_ref = cc.e_corr
            assert cc.converged
            #if pt:
            #    e_pt_ref = cc.ccsd_t()

            if not emb_ccsd:
                with open(output, "a") as f:
                    f.write("%3f  %.8e  %.8e\n" % (d, mf.e_tot, mf.e_tot+e_ccsd_ref))

        else:
            e_ccsd_ref = 0

        if emb_ccsd:
            ecc = EmbCCSD(mf)
            #ecc.create_custom_clusters([("O1", "H3")])
            ecc.create_atom_clusters()
            conv = ecc.run(max_power=max_power, pertT=pt)
            if MPI_rank == 0:
                assert conv

            if MPI_rank == 0:
                print("CCSD correlation energy:    %+g" % e_ccsd_ref)
                print("EmbCCSD correlation energy: %+g" % ecc.e_ccsd)
                print("Error:                      %+g" % (ecc.e_ccsd - e_ccsd_ref))
                print("%% Correlation:             %.3f %%" % (100*ecc.e_ccsd/e_ccsd_ref))

                with open(output, "a") as f:
                    f.write("%3f  %.8e  %.8e  %.8e\n" % (d, mf.e_tot, mf.e_tot+e_ccsd_ref, mf.e_tot+ecc.e_ccsd))

                #if pt:
                #    print("CCSD(T) correlation energy:    %+g" % (e_ccsd_ref + e_pt_ref))
                #    print("EmbCCSD(T) correlation energy: %+g" % (ecc_sd + ecc_pt))
                #    print("Error:                         %+g" % (ecc_sd+ecc_pt - e_ccsd_ref-e_pt_ref))
                #    print("%% Correlation:                %.3f %%" % (100*(ecc_sd+ecc_pt)/(e_ccsd_ref+e_pt_ref)))

                #    with open(output, "a") as f:
                #        f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (d, mf.e_tot, mf.e_tot+e_ccsd_ref, mf.e_tot+e_ccsd_ref+e_pt_ref, mf.e_tot+ecc_sd, mf.e_tot+ecc_sd+ecc_pt))

