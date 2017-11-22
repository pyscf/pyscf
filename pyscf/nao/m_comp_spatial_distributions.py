from __future__ import division
import numpy as np
from pyscf.nao import scf
from pyscf.nao.m_tools import find_nearrest_index
import h5py
from pyscf.nao.m_libnao import libnao
from ctypes import POINTER, c_int, c_int32, c_int64, c_float, c_double

#try:
#    import numba as nb
#    from pyscf.nao.m_comp_spatial_numba import get_spatial_density_numba, get_spatial_density_numba_parallel
#    use_numba = True
#except:
#    use_numba = False


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
        self.mu2dn_re = np.dot(self.Eext, dn[:, iw, :].real)
        self.mu2dn_im = np.dot(self.Eext, dn[:, iw, :].imag)

        self.dn_spatial = self.get_spatial_density(self.pb.prod_log)


    def get_spatial_density(self, ao_log=None):
        """
            Convert the density change from product basis
            to cartesian coordinates
        """

        from pyscf.nao.m_ao_matelem import ao_matelem_c
        from pyscf.nao.m_csphar import csphar
        from pyscf.nao.m_rsphar_libnao import rsphar
        from pyscf.nao.m_log_interp import comp_coeffs_

        aome = ao_matelem_c(self.ao_log.rr, self.ao_log.pp)
        me = ao_matelem_c(self.ao_log) if ao_log is None else aome.init_one_set(ao_log)
        atom2s = np.zeros((self.natm+1), dtype=np.int64)
        for atom,sp in enumerate(self.atom2sp):
            atom2s[atom+1]= atom2s[atom] + me.ao1.sp2norbs[sp]

        print(atom2s)
        gammin_jt = float(self.pb.prod_log.interp_rr.gammin_jt)
        dg_jt = float(self.pb.prod_log.interp_rr.dg_jt)
        Nx, Ny, Nz = self.mesh[0].size, self.mesh[1].size, self.mesh[2].size 

        dn_spatial_re = np.zeros((Nx, Ny, Nz), dtype=np.float32)
        dn_spatial_im = np.zeros((Nx, Ny, Nz), dtype=np.float32)

        # Aligning the data is important to avoid troubles in the fortran side!
        atoms2sp = np.require(self.atom2sp, dtype=self.atom2sp.dtype, requirements=["A", "O"])

        libnao.get_spatial_density_parallel(dn_spatial_re.ctypes.data_as(POINTER(c_float)),
                dn_spatial_im.ctypes.data_as(POINTER(c_float)),
                self.mu2dn_re.ctypes.data_as(POINTER(c_double)),
                self.mu2dn_im.ctypes.data_as(POINTER(c_double)),
                self.mesh[0].ctypes.data_as(POINTER(c_double)), 
                self.mesh[1].ctypes.data_as(POINTER(c_double)), 
                self.mesh[2].ctypes.data_as(POINTER(c_double)), 
                atoms2sp.ctypes.data_as(POINTER(c_int64)), 
                atom2s.ctypes.data_as(POINTER(c_int64)), 
                c_double(gammin_jt), c_double(dg_jt), c_int(Nx), 
                c_int(Ny), c_int(Nz), c_int(self.nprod), c_int(self.natoms), 
                c_int(self.nspecies))

        # sum(dn) =  77.8299 63.8207
        # print(np.sum(abs(dn_spatial_re)), np.sum(abs(dn_spatial_im)))
        return dn_spatial_re + 1.0j*dn_spatial_im
