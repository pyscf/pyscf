from __future__ import division
import numpy as np
from pyscf.nao import mf
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


class spatial_distribution(mf):
    """
        class to calculate spatial distribution of
            * density change
            * induce potential
            * induce electric field
            * intensity of the induce Efield

        Example:

            from __future__ import print_function, division
            import numpy as np
            from pyscf.nao import tddft_iter
            from pyscf.nao.m_comp_spatial_distributions import spatial_distribution

            from ase.units import Ry, eV, Ha, Bohr


            # run tddft calculation
            td = tddft_iter(label="siesta", iter_broadening=0.15/Ha, xc_code='LDA,PZ')

            omegas = np.linspace(0.0, 10.0, 200)/Ha + 1j*td.eps
            td.comp_dens_inter_along_Eext(omegas, Eext=np.array([1.0, 0.0, 0.0]))

            box = np.array([[-10.0, 10.0],
                            [-10.0, 10.0],
                            [-10.0, 10.0]])/Bohr
            dr = np.array([0.5, 0.5, 0.5])/Bohr

            # initialize spatial calculations
            spd = spatial_distribution(td.dn, omegas, box, dr = dr, label="siesta")

            # compute spatial density change distribution
            spd.get_spatial_density(3.5/Ha, Eext=np.array([1.0, 0.0, 0.0]))

            # compute Efield
            Efield = spd.comp_induce_field()
            
            # compute potential
            pot = spd.comp_induce_potential()
            
            # compute intensity
            intensity = spd.comp_intensity_Efield(Efield)
    """

    def __init__(self, dn, freq, box, excitation = "light", **kw):
        """
            initialize the class aith scf.__init__, checking arguments
            All quantites must be given in a.u.

            Warning: the same kw parameters used for the densit calculations must be used

            dn (complex array calculated from tddft_iter, dim: [3, nfreq, nprod]): the induce density
                    in the product basis calulated from comp_dens_inter_along_Eext in tddft_iter.
            freq (real array, dim: [nfreq]): the frequency range for which dn was computed
            w0 (real): frequency at which you want to computer the spatial quantities
            box (real array, dim: [3, 2]): spatial boundaries of the box in which you want
                        to compute the spatial quantities, first index run over x, y, z axis,
                        second index stand for lower and upper boundaries.
            excitation (string): type of external perturbation, can be
                * light: system is pertubated with a constant external electric field
                        the density change has been computed with tddft_iter module
                * electron: system has been perturbated by a moving charge (EELS)
                        the density change has has been computed with the tddft_tem module
        """


        self.dr = kw['dr'] if 'dr' in kw else np.array([0.3, 0.3, 0.3])

        assert self.dr.size == 3
        assert box.shape == (3, 2)
        assert excitation in ["light", "electron"]

        
        self.box = box
        self.excitation = excitation
        self.mesh = np.array([np.arange(box[0, 0], box[0, 1]+self.dr[0], self.dr[0]),
                              np.arange(box[1, 0], box[1, 1]+self.dr[1], self.dr[1]),
                              np.arange(box[2, 0], box[2, 1]+self.dr[2], self.dr[2])])
        self.freq = freq
        self.dn = dn

        mf.__init__(self, **kw)

        if self.excitation == "light" and len(dn.shape) == 3:
            self.nprod = dn.shape[2]
        elif self.excitation == "light" and len(dn.shape) != 3:
            raise ValueError("Wrong dimension: len(dn.shape) = {0}, for the density change with light excitation".format(len(dn.shape)))
        elif self.excitation == "electron" and len(dn.shape) == 2:
            self.nprod = dn.shape[1]
        elif self.excitation == "electron" and len(dn.shape) != 2:
            raise ValueError("Wrong dimension: len(dn.shape) = {0}, for the density change with electron excitation".format(len(dn.shape)))
        else:
            raise ValueError("wrong excitation type??")

    def get_spatial_density(self, w0, ao_log=None, Eext = np.array([1.0, 1.0, 1.0])):
        """
            Compute the density change fromm the product basis
            to cartesian bais for the frequency w0 and the external
            field directed along Eext
        """

        assert Eext.size == 3

        from pyscf.nao.m_ao_matelem import ao_matelem_c
        from pyscf.nao.m_csphar import csphar
        from pyscf.nao.m_rsphar_libnao import rsphar
        from pyscf.nao.m_log_interp import comp_coeffs_

        iw = find_nearrest_index(self.freq, w0)
        if self.excitation == "light":
            self.Eext = Eext/np.sqrt(np.dot(Eext, Eext))

            self.mu2dn_re = np.dot(self.Eext, self.dn[:, iw, :].real)
            self.mu2dn_im = np.dot(self.Eext, self.dn[:, iw, :].imag)
        elif self.excitation == "electron":
            self.Eext = None
            self.mu2dn_re = self.dn[iw, :].real
            self.mu2dn_im = self.dn[iw, :].imag


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
                c_int(Nx), c_int(Ny), c_int(Nz), 
                c_int(self.nprod), c_int(self.natoms))

        # sum(dn) =  77.8299 63.8207
        # print(np.sum(abs(dn_spatial_re)), np.sum(abs(dn_spatial_im)))
        self.dn_spatial = dn_spatial_re + 1.0j*dn_spatial_im

    def comp_induce_potential(self):
        """
            Compute the induce potential corresponding to the density change
            calculated in get_spatial_density
        """

        from scipy.signal import convolve

        Nx, Ny, Nz = self.mesh[0].size, self.mesh[1].size, self.mesh[2].size

        grid = np.zeros((Nx, Ny, Nz), dtype = np.float64)
        factor = self.dr[0]*self.dr[1]*self.dr[2]/(np.sqrt(2*np.pi)**3)

        libnao.comp_spatial_grid_pot(
            self.dr.ctypes.data_as(POINTER(c_double)), 
            self.mesh[0].ctypes.data_as(POINTER(c_double)),
            self.mesh[1].ctypes.data_as(POINTER(c_double)),
            self.mesh[2].ctypes.data_as(POINTER(c_double)),
            grid.ctypes.data_as(POINTER(c_double)),
            c_int(Nx), c_int(Ny), c_int(Nz))

        return convolve(grid, self.dn_spatial, mode="same", method="fft")*factor


    def comp_induce_field(self):
        """
            Compute the induce Electric field corresponding to the density change
            calculated in get_spatial_density
        """
        
        from scipy.signal import convolve

        Nx, Ny, Nz = self.mesh[0].size, self.mesh[1].size, self.mesh[2].size

        Efield = np.zeros((3, Nx, Ny, Nz), dtype = np.complex64)
        grid = np.zeros((Nx, Ny, Nz), dtype = np.float64)
        factor = self.dr[0]*self.dr[1]*self.dr[2]/(np.sqrt(2*np.pi)**3)

        for xyz in range(3):
            grid.fill(0.0)
            libnao.comp_spatial_grid(
                self.dr.ctypes.data_as(POINTER(c_double)), 
                self.mesh[0].ctypes.data_as(POINTER(c_double)),
                self.mesh[1].ctypes.data_as(POINTER(c_double)),
                self.mesh[2].ctypes.data_as(POINTER(c_double)),
                c_int(xyz+1), 
                grid.ctypes.data_as(POINTER(c_double)),
                c_int(Nx), c_int(Ny), c_int(Nz))

            Efield[xyz, :, :, :] = convolve(grid, self.dn_spatial, 
                                            mode="same", method="fft")*factor

        return Efield

    def comp_intensity_Efield(self, Efield):
        """
            Compute the intesity of the induce electric field
        """
        return np.sum(Efield.real**2, axis=0) + np.sum(Efield.imag**2, axis=0)
