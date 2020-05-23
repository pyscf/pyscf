import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lo
import pyscf.cc

def create_atom_clusters(mf):
    """Divide atomic orbitals into clusters."""

    # base atom for each AO
    mol = mf.mol
    base_atoms = np.asarray([ao[0] for ao in mol.ao_labels(None)])

    clusters = []
    for atomid in range(mol.natm):
        ao_indices = np.nonzero(base_atoms == atomid)[0]
        c = Cluster(mf, ao_indices)
        clusters.append(c)

    return clusters

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

    @property
    def S(self):
        """Overlap matrix in cluster."""
        ovlp = self.mf.get_ovlp()
        S = ovlp[np.ix_(self.indices, self.indices)]
        return S

    def make_projector(self):
        """Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b"""
        S1 = self.mf.get_ovlp()
        nao = self.mol.nao_nr()
        S2 = self.S
        S21 = S1[self.indices]
        #s2_inv = np.linalg.inv(s2)
        #p_21 = np.dot(s2_inv, s21)
        # Better: solve with Cholesky decomposition
        # Solve: S2 * p_21 = S21 for p_21
        p_21 = scipy.linalg.solve(S2, S21, assume_a="pos")
        p_12 = np.eye(nao)[:,self.indices]
        p = np.dot(p_12, p_21)
        return p


    def make_local_orbitals(self, tol=1e-12):
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
        reverse = np.s_[::-1]
        e = e[reverse]
        r = r[:,reverse]

        nloc = len(e[e>tol])
        assert nloc == len(self)
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

        nbath = sum(mask_bath)
        assert nbath <= len(self)

        C[:,env] = np.dot(C[:,env], v)

        return C, nbath

    def run_ccsd(self, diagonalize_fock=True):
        C = self.make_local_orbitals()
        C, nbath = self.make_dmet_bath_orbitals(C)

        S = self.mf.get_ovlp()
        SDS_hf = np.linalg.multi_dot((S, self.mf.make_rdm1(), S))

        # Diagonalize cluster DM
        ncl = len(self) + nbath
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
        assert np.allclose(np.fmin(abs(occ), abs(occ-2)), 0, atol=1e-6, rtol=0)
        occ = np.asarray([2 if occ > 1 else 0 for occ in occ])
        sort = np.argsort(-occ, kind="mergesort") # mergesort is stable (keeps relative order)
        C_cc = C_cc[:,sort]
        occ = occ[sort]

        # Accelerates convergence
        if diagonalize_fock:
            F = np.linalg.multi_dot((C_cc.T, mf.get_fock(), C_cc))
            # Occupied
            o = occ > 0
            e, r = np.linalg.eigh(F[np.ix_(o, o)])
            C_cc[:,o] = np.dot(C_cc[:,o], r)
            # Virtual
            v = occ == 0
            e, r = np.linalg.eigh(F[np.ix_(v, v)])
            C_cc[:,v] = np.dot(C_cc[:,v], r)

        cc = pyscf.cc.CCSD(self.mf, mo_coeff=C_cc, mo_occ=occ)
        cc.verbose = 4
        cc.kernel()
        assert cc.converged
        assert np.allclose(C_cc, cc.mo_coeff)

        e_loc = self.get_local_energy(cc)

        return e_loc

    def get_local_energy(self, cc):

        occ = cc.mo_occ
        eris = cc.ao2mo()
        S = self.mf.get_ovlp()
        C = cc.mo_coeff
        SC = np.dot(S, C)

        l = self.indices
        o = cc.mo_occ > 0
        v = cc.mo_occ == 0

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

    #def make_power_bath_orbitals(self, C, tol=1e-8):
    #    S = self.mf.get_ovlp()
    #    csc = np.linalg.multi_dot(self.mf.mo_coeff.T, S, C)

    #def make_local_orbitals_2(self):
    #    C = self.mf.mo_coeff[self.indices]
    #    C = pyscf.lo.vec_lowdin(C, self.S)
    #    return C


if __name__ == "__main__":
    import pyscf
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.M(
            atom = "O 0 0 -1.5; C 0 0 0; O 0 0 1.5",
            basis = "ccpvdz"
            )

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    cc = pyscf.cc.CCSD(mf)
    cc.kernel()
    assert cc.converged
    e_cc = cc.e_tot
    print("CCSD energy: %e" % e_cc)

    clusters = create_atom_clusters(mf)

    e_corr = 0.0
    for c in clusters:
        e_cc = c.run_ccsd()
        e_corr += e_cc
        print("Local energy: %e" % e_cc)

    print("CCSD energy: %e" % e_corr)
