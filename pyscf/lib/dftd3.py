# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import ctypes
from pyscf import lib, gto

libdftd3 = np.ctypeslib.load_library('libs-dftd3',  os.path.abspath(os.path.join(__file__, '..', 'deps', 'lib')))

_load_damping_param = {
    "d3bj":   libdftd3.dftd3_load_rational_damping,      #RationalDampingParam,
    "d3zero": libdftd3.dftd3_load_zero_damping,          #ZeroDampingParam,
    "d3bjm":  libdftd3.dftd3_load_mrational_damping,     #ModifiedRationalDampingParam,
    "d3mbj":  libdftd3.dftd3_load_mrational_damping,     #ModifiedRationalDampingParam,
    "d3zerom":libdftd3.dftd3_load_mzero_damping,         #ModifiedZeroDampingParam,
    "d3mzero":libdftd3.dftd3_load_mzero_damping,         #ModifiedZeroDampingParam,
    "d3op":   libdftd3.dftd3_load_optimizedpower_damping #OptimizedPowerDampingParam,
}

class _d3_restype(ctypes.Structure):
    pass

_d3_p = ctypes.POINTER(_d3_restype)

libdftd3.dftd3_new_error.restype                   = _d3_p
libdftd3.dftd3_new_structure.restype               = _d3_p
libdftd3.dftd3_load_optimizedpower_damping.restype = _d3_p
libdftd3.dftd3_load_mzero_damping.restype          = _d3_p
libdftd3.dftd3_load_mrational_damping.restype      = _d3_p
libdftd3.dftd3_load_zero_damping.restype           = _d3_p
libdftd3.dftd3_load_rational_damping.restype       = _d3_p
libdftd3.dftd3_new_d3_model.restype                = _d3_p

class DFTD3Dispersion(lib.StreamObject):
    def __init__(self, mol, xc, version='d3bj', atm=False):
        coords = np.asarray(mol.atom_coords(), dtype=np.double, order='C')
        nuc_types = [gto.charge(mol.atom_symbol(ia))
                     for ia in range(mol.natm)]
        nuc_types = np.asarray(nuc_types, dtype=np.int32)
        self.natm = mol.natm
        self._lattice = lib.c_null_ptr()
        self._periodic = lib.c_null_ptr()

        err = libdftd3.dftd3_new_error()
        self._mol = libdftd3.dftd3_new_structure(
            err,
            ctypes.c_int(mol.natm),
            nuc_types.ctypes.data_as(ctypes.c_void_p),
            coords.ctypes.data_as(ctypes.c_void_p),
            self._lattice,
            self._periodic,
        )

        self._disp = libdftd3.dftd3_new_d3_model(err, self._mol)
        self._param = _load_damping_param[version](
            err,
            ctypes.create_string_buffer(xc.encode(), size=50),
            ctypes.c_bool(atm))

        libdftd3.dftd3_delete_error(ctypes.byref(err))

    def __del__(self):
        err = libdftd3.dftd3_new_error()
        libdftd3.dftd3_delete_param(ctypes.byref(self._param))
        libdftd3.dftd3_delete_structure(err, ctypes.byref(self._mol))
        libdftd3.dftd3_delete_model(err, ctypes.byref(self._disp))
        libdftd3.dftd3_delete_error(ctypes.byref(err))

    def get_dispersion(self, grad=False):
        res = {}
        _energy = np.array(0.0, dtype=np.double)
        if grad:
            _gradient = np.zeros((self.natm,3))
            _sigma = np.zeros((3,3))
            _gradient_str = _gradient.ctypes.data_as(ctypes.c_void_p)
            _sigma_str = _sigma.ctypes.data_as(ctypes.c_void_p)
        else:
            _gradient = None
            _sigma = None
            _gradient_str = lib.c_null_ptr()
            _sigma_str = lib.c_null_ptr()

        err = libdftd3.dftd3_new_error()
        libdftd3.dftd3_get_dispersion(
            err,
            self._mol,
            self._disp,
            self._param,
            _energy.ctypes.data_as(ctypes.c_void_p),
            _gradient_str,
            _sigma_str)
        res = dict(energy=_energy)
        if _gradient is not None:
            res.update(gradient=_gradient)
        if _sigma is not None:
            res.update(virial=_sigma)

        libdftd3.dftd3_delete_error(ctypes.byref(err))
        
        return res