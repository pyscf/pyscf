#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the TREXIO files written by pyscf.tools.trexio with trexio-validate.

trexio-validate (https://github.com/TREX-CoE/trexio-validate) recomputes the
contents of a TREXIO file from the basis set stored in that same file, using
libcint, and compares the result against the stored data.  It checks the
conversion in the conventions of the TREXIO specification rather than those of
PySCF, so it covers the AO ordering, the solid-harmonic convention, the three
normalization factors and the index order of the stored integrals -- none of
which a round trip through PySCF's own reader can detect.

The validator is not on PyPI; build it from source::

    git clone https://github.com/TREX-CoE/trexio-validate
    cmake -S trexio-validate -B build \\
          -DLIBCINT_INCLUDE_DIR=$PWD/pyscf/lib/deps/include \\
          -DLIBCINT_LIBRARY=$PWD/pyscf/lib/deps/lib/libcint.so
    cmake --build build
    cmake --install build --prefix <prefix>

and put either the installed ``trexio_validate`` Python module on PYTHONPATH or
the ``trexio-validate`` executable on PATH.  These tests are skipped when
neither is found.

The file is validated on disk rather than through
``trexio_validate.validate(handle)``: the ``trexio`` wheels on PyPI compile
their own copy of libtrexio into the extension module, so a ``trexio.File``
handle they produce does not match the libtrexio the validator is linked to.
"""

import os
import shutil
import subprocess
import tempfile

import pytest

from pyscf import gto, scf, dft

try:
    import trexio as trexio_lib
    from pyscf.tools import trexio
except ImportError:
    trexio_lib = None
    trexio = None

try:
    import trexio_validate
except ImportError:
    trexio_validate = None

_EXE = shutil.which("trexio-validate")

pytestmark = pytest.mark.skipif(
    trexio_lib is None or (trexio_validate is None and _EXE is None),
    reason="trexio and trexio-validate are both needed",
)

# Exit status of the executable when no check could run at all.
_NOTHING_CHECKED = 77

# Checks that need only the geometry, the basis set and the MO coefficients.
BASE = ("basis", "nucleus_repulsion", "electron_count", "mo_orthonormality")
# Written by write_ao_eri=True.  The core Hamiltonian is listed separately
# because the validator cannot recompute it when an ECP is present.
AO_1E = ("ao_1e_int_overlap", "ao_1e_int_kinetic", "ao_1e_int_potential_n_e")
AO = AO_1E + ("ao_1e_int_core_hamiltonian", "ao_2e_int_eri")
# Written by write_mo_eri=True.
MO = ("mo_1e_int_overlap", "mo_1e_int_kinetic", "mo_1e_int_potential_n_e",
      "mo_1e_int_core_hamiltonian", "mo_2e_int_eri")


def _hdf5_available():
    with tempfile.TemporaryDirectory() as d:
        try:
            with trexio_lib.File(os.path.join(d, "probe"), "w",
                                 back_end=trexio_lib.TREXIO_HDF5):
                pass
            return True
        except Exception:
            return False


backend = "h5" if trexio_lib is not None and _hdf5_available() else "text"


def _validate(path, require):
    """Run the validator on *path*, requiring *require*; (ok, report text).

    ``require`` turns a check whose data is absent from a skip into a failure,
    so that a regression which stops writing a quantity is caught as well as
    one that writes it wrongly.
    """
    if trexio_validate is not None:
        report = trexio_validate.validate_file(path, require=list(require))
        return report.ok, str(report)

    run = subprocess.run([_EXE, "--require", ",".join(require), path],
                         capture_output=True, text=True)
    if run.returncode == _NOTHING_CHECKED:
        pytest.skip("trexio-validate could not check the file:\n" + run.stdout)
    if run.returncode not in (0, 1):
        raise RuntimeError("trexio-validate failed to run (exit %d):\n%s%s"
                           % (run.returncode, run.stdout, run.stderr))
    return run.returncode == 0, run.stdout + run.stderr


def _check(mf, require, **kwargs):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test." + ("h5" if backend == "h5" else "text"))
        trexio.to_trexio(mf, path, backend=backend, **kwargs)
        ok, report = _validate(path, require)
    assert ok, "trexio-validate rejected the file:\n" + report


H2O = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def _rhf(**kwargs):
    kwargs.setdefault("atom", H2O)
    return scf.RHF(gto.M(verbose=0, **kwargs)).run()


@pytest.mark.parametrize("eri_sym", ["s1", "s4", "s8"])
def test_ao_integrals_spherical(eri_sym):
    """AO integrals of a spherical basis, for each ERI packing.

    Regression test for the AO ordering: the AO integrals must be stored in
    TREXIO's AO order, the one mo.coefficient already used.
    """
    _check(_rhf(basis="6-31g*"), BASE + AO,
           write_ao_eri=True, eri_sym=eri_sym)


def test_ao_integrals_cartesian():
    _check(_rhf(basis="6-31g*", cart=True), BASE + AO,
           write_ao_eri=True, eri_sym="s8")


def test_ao_integrals_high_angular_momentum():
    """cc-pVTZ on oxygen has d and f shells, where the ordering differs most."""
    _check(_rhf(basis="cc-pvtz"), BASE + AO, write_ao_eri=True, eri_sym="s8")


def test_mo_integrals():
    _check(_rhf(basis="6-31g*"), BASE + MO, write_mo_eri=True, eri_sym="s4")


def test_ao_and_mo_integrals():
    _check(_rhf(basis="6-31g*"), BASE + AO + MO,
           write_ao_eri=True, write_mo_eri=True, eri_sym="s4")


def test_uhf():
    mol = gto.M(atom="O 0 0 0", basis="6-31g*", spin=2, verbose=0)
    _check(scf.UHF(mol).run(), BASE + AO, write_ao_eri=True, eri_sym="s8")


def test_rohf():
    mol = gto.M(atom="O 0 0 0", basis="6-31g*", spin=2, verbose=0)
    _check(scf.ROHF(mol).run(), BASE + AO, write_ao_eri=True, eri_sym="s8")


def test_rks():
    mol = gto.M(atom=H2O, basis="6-31g*", verbose=0)
    _check(dft.RKS(mol).run(xc="pbe"), BASE + AO,
           write_ao_eri=True, eri_sym="s8")


# The validator cannot recompute a core Hamiltonian that contains ECP terms,
# so that check is left out of the requirements here; it reports it as skipped.
ECP_CHECKS = BASE + AO_1E + ("ao_2e_int_eri",)


def test_ecp():
    mol = gto.M(atom="I 0 0 0; I 0 0 2.67", basis="lanl2dz", ecp="lanl2dz",
                verbose=0)
    _check(scf.RHF(mol).run(), ECP_CHECKS, write_ao_eri=True, eri_sym="s8")


def test_ecp_on_some_atoms_only():
    """Only the heavy atom carries an ECP, as in most applications."""
    mol = gto.M(atom="I 0 0 0; H 0 0 1.6", basis="lanl2dz", ecp="lanl2dz",
                verbose=0)
    _check(scf.RHF(mol).run(), ECP_CHECKS, write_ao_eri=True, eri_sym="s8")


def test_mole_only():
    """A file holding just the geometry and the basis set."""
    mol = gto.M(atom=H2O, basis="6-31g*", verbose=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mol." + ("h5" if backend == "h5" else "text"))
        trexio.to_trexio(mol, path, backend=backend)
        ok, report = _validate(path, ("basis", "nucleus_repulsion"))
    assert ok, "trexio-validate rejected the file:\n" + report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
