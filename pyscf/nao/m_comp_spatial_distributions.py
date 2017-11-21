from __future__ import division
import numpy as np
from pyscf.nao import scf
from pyscf.nao.m_tools import find_nearrest_index
import h5py

try:
    import numba as nb
    from pyscf.nao.m_comp_spatial_numba import get_spatial_density_numba, get_spatial_density_numba_parallel
    use_numba = True
except:
    use_numba = False


class spatial_distribution(scf):
    """
        class to calculate spatial distribution of
            * density change
            * induce potential
            * induce electric field
            * intensity of the induce Efield
    """

    def __init__(self, dn, freq, w0, box, **kw):
        """
            initialize the class aith scf.__init__, checking arguments
            and compute the spatial distribution of the density change
            necessary for the other quantities. All quantites must be given
            in a.u.

            dn (complex array calculated from tddft_iter, dim: [3, nfreq, nprod]): the induce density
                    in the product basis.
            freq (real array, dim: [nfreq]): the frequency range for which dn was computed
            w0 (real): frequency at which you want to computer the spatial quantities
            box (real array, dim: [3, 2]): spatial boundaries of the box in which you want
                        to compute the spatial quantities, first index run over x, y, z axis,
                        second index stand for lower and upper boundaries.
        """

        Eext = kw['Eext'] if 'Eext' in kw else np.array([1.0, 0.0, 0.0])
        self.dr = kw['dr'] if 'dr' in kw else np.array([0.3, 0.3, 0.3])
        self.nb_cache = kw["cache"] if 'cache' in kw else True
        self.nb_parallel = kw["parallel"] if 'parallel' in kw else False

        assert Eext.size == 3
        assert self.dr.size == 3
        assert box.shape == (3, 2)

        self.Eext = Eext/np.sqrt(np.dot(Eext, Eext))
        self.mesh = np.array([np.arange(box[0, 0], box[0, 1]+self.dr[0], self.dr[0]),
                              np.arange(box[1, 0], box[1, 1]+self.dr[1], self.dr[1]),
                              np.arange(box[2, 0], box[2, 1]+self.dr[2], self.dr[2])])

        scf.__init__(self, **kw)
        iw = find_nearrest_index(freq, w0)
        self.nprod = dn.shape[2]
        self.mu2dn = np.dot(self.Eext, dn[:, iw, :])

        self.get_spatial_density(self.pb.prod_log)


    def get_spatial_density(self, ao_log=None):

        from pyscf.nao.m_ao_matelem import ao_matelem_c
        from pyscf.nao.m_csphar import csphar
        from pyscf.nao.m_rsphar_libnao import rsphar
        from pyscf.nao.m_log_interp import comp_coeffs_

        aome = ao_matelem_c(self.ao_log.rr, self.ao_log.pp)
        me = ao_matelem_c(self.ao_log) if ao_log is None else aome.init_one_set(ao_log)
        atom2s = np.zeros((self.natm+1), dtype=np.int64)
        for atom,sp in enumerate(self.atom2sp):
            atom2s[atom+1]= atom2s[atom] + me.ao1.sp2norbs[sp]

        rr = self.ao_log.rr
        coeffs = np.zeros((6), dtype=np.float64)
        res = np.zeros((self.nprod), dtype = np.float64)
        self.dn_spatial = np.zeros((self.mesh[0].size, self.mesh[1].size, self.mesh[2].size),
                                        dtype=np.complex64)

        if use_numba:
            if self.nb_parallel:
                get_spatial = nb.jit(nopython=False, cache=self.nb_cache, parallel=self.nb_parallel)(get_spatial_density_numba_parallel)
            else:
                get_spatial = nb.jit(nopython=True, cache=self.nb_cache)(get_spatial_density_numba)
            get_spatial(self.dn_spatial, self.mu2dn, self.mesh, self.atom2sp, self.atom2coord,
                     atom2s, np.array(self.pb.prod_log.sp_mu2j), np.array(self.pb.prod_log.psi_log_rl), 
                     np.array(self.pb.prod_log.sp_mu2s), np.array(self.ao_log.sp2rcut), rr, res, coeffs,
                     self.pb.prod_log.interp_rr.gammin_jt, self.pb.prod_log.interp_rr.dg_jt,
                     self.pb.prod_log.interp_rr.nr)
        else:
            for ix, x in enumerate(self.mesh[0]):
                print(ix, self.mesh[0].shape)
                for iy, y in enumerate(self.mesh[1]):
                    for iz, z in enumerate(self.mesh[2]):

                        br = np.array([x, y, z])

                        for atm, sp in enumerate(self.atom2sp):
                            brp = br - self.atom2coord[atm, :]
                            r = np.sqrt(np.dot(brp, brp))
                            rcut = self.ao_log.sp2rcut[sp]

                            si = atom2s[atm]
                            fi = atom2s[atm+1]

                            if r>rcut: continue
                            jmx_sp = self.pb.prod_log.sp_mu2j[sp].max()
                            rsh = np.zeros((jmx_sp+1)**2)
                            rsphar(brp, jmx_sp, rsh)

                            ir = comp_coeffs_(self.pb.prod_log.interp_rr, r, coeffs)
                            for j,ff,s,f in zip(self.pb.prod_log.sp_mu2j[sp],
                                                self.pb.prod_log.psi_log_rl[sp],
                                                self.pb.prod_log.sp_mu2s[sp],
                                                self.pb.prod_log.sp_mu2s[sp][1:]):
                                fval = (ff[ir:ir+6]*coeffs).sum() if j==0 else (ff[ir:ir+6]*coeffs).sum()*r**j
                                res[si+s: si+f] = fval * rsh[j*(j+1)-j:j*(j+1)+j+1]

                        self.dn_spatial[ix, iy, iz] += np.sum(res*self.mu2dn)

