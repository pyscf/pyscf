#!/usr/bin/env python

"""
TREX-IO interface

References:
    https://trex-coe.github.io/trexio/trex.html

Installation instruction:
    https://github.com/TREX-CoE/trexio/blob/master/python/README.md
"""

import os
import re
import math
import shutil
import numpy as np
from collections import defaultdict

import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf import pbc
from pyscf import mcscf
from pyscf import fci
from pyscf import ao2mo
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import df as pbcdf, tools as pbctools
from pyscf.mcscf import mc1step, mc2step, casci

try:
    import trexio
except ImportError:
    raise ImportError(
        'Unable to import the trexio library, which this module requires.\n'
        'Install it with `pip install trexio` (or `pip install pyscf[trexio]`).\n'
        'See https://github.com/TREX-CoE/trexio/blob/master/python/README.md'
    )


def to_trexio(
    obj, filename, backend="h5",
    write_ao_eri=False,
    write_mo_eri=False,
    eri_sym='s1',
    write_mo_rdm=False,
    write_mcscf_eri=True,
    write_mcscf_rdm=True,
    ci_threshold=None,
    chunk_size=None,
):
    """Convert a PySCF object to a TREXIO file.

    The function dispatches on the type of *obj* and writes the appropriate
    quantities to *filename*.  Any file or directory already present at
    *filename* is **deleted and recreated** before writing (i.e. the function
    always produces a fresh file).

    Dispatch summary
    ----------------
    ``Mole`` / ``Cell``
        Nuclear geometry, basis set, ECP (if present), symmetry, and PBC
        cell vectors.  No orbital or integral data.
    ``SCF`` (RHF, UHF, ROHF, RKS, UKS, ROKS, and PBC counterparts)
        All ``Mole``/``Cell`` data plus MO coefficients, energies, occupations,
        and spin labels.  AO and/or MO integrals and density matrices are
        written only when explicitly requested via *write_ao_eri*,
        *write_mo_eri*, and *write_mo_rdm*.
    ``CASCI`` / ``CASSCF`` / ``UCASCI`` / ``UCASSCF``
        All ``Mole``/``Cell`` and SCF data plus CI determinants and natural
        occupations.  By default the active-space effective Hamiltonian
        (h1eff / h2eff) and the active-space reduced density matrices are
        also written; each can be suppressed independently.

    Parameters
    ----------
    obj : gto.Mole, pbc.gto.Cell, scf.hf.SCF, \
          mcscf.casci.CASCI, mcscf.mc1step.CASSCF, \
          mcscf.ucasci.UCASCI, or mcscf.umc1step.UCASSCF
        PySCF object to export.
    filename : str
        Path to the output TREXIO file.  An existing file (HDF5 backend) or
        directory (text backend) at this path is removed before writing.
    backend : {'h5', 'hdf5', 'text', 'txt', 'auto'}, optional
        TREXIO backend.  ``'h5'``/``'hdf5'`` produces a single HDF5 binary
        file (default).  ``'text'``/``'txt'`` produces a directory of plain
        text files.  ``'auto'`` lets TREXIO detect the format from *filename*.
    write_ao_eri : bool, optional
        *SCF only.*  When ``True``, write the AO-basis one-electron integrals
        (overlap, kinetic, nuclear attraction, core Hamiltonian stored as
        ``ao_1e_int_*``) **and** the AO-basis two-electron repulsion integrals
        (``ao_2e_int_eri``) with the symmetry specified by *eri_sym*.
        Default ``False``.
    write_mo_eri : bool, optional
        *SCF only.*  When ``True``, write the MO-basis one-electron integrals
        (``mo_1e_int_*``) **and** the MO-basis two-electron repulsion integrals
        (``mo_2e_int_eri``) with the symmetry specified by *eri_sym*.
        For UHF/UKS the alpha and beta MO coefficient blocks are concatenated
        so that all spin blocks of the ERI are covered.
        Default ``False``.
    eri_sym : {'s1', 's4', 's8'}, optional
        Integral permutation symmetry applied to both AO and MO two-electron
        integrals when *write_ao_eri* or *write_mo_eri* is ``True``.

        * ``'s1'`` — no symmetry, all :math:`N^4` elements stored.
        * ``'s4'`` — 4-fold symmetry :math:`(ij|kl)=(ji|kl)=(ij|lk)=(ji|lk)`,
          roughly :math:`N^4/8` elements.
        * ``'s8'`` — 8-fold symmetry (real AO integrals only),
          roughly :math:`N^4/8` with the additional
          :math:`(ij|kl)=(kl|ij)` relation; supported for AO basis only
          (``write_ao_eri=True``).

        ``write_mo_eri=True`` supports ``'s1'`` and ``'s4'`` only.
        Default ``'s1'``.
    write_mo_rdm : bool, optional
        *SCF only.*  When ``True``, write the 1-body reduced density matrix
        (``rdm_1e`` / ``rdm_1e_up`` + ``rdm_1e_dn`` for UHF) and the
        2-body reduced density matrix (``rdm_2e`` / spin-block ``rdm_2e_*``
        for UHF) in the MO basis.  Default ``False``.
    write_mcscf_eri : bool, optional
        *MCSCF only.*  When ``True`` (default), write the active-space
        effective one-electron integrals (``mo_1e_int_core_hamiltonian``
        containing h1eff) and the active-space two-electron effective
        integrals (``mo_2e_int_eri`` containing h2eff) in the MO basis.
        Set to ``False`` to skip these potentially large arrays.
    write_mcscf_rdm : bool, optional
        *MCSCF only.*  When ``True`` (default), write the active-space
        1-RDM (``rdm_1e`` or spin-block ``rdm_1e_up``/``rdm_1e_dn``) and
        2-RDM (``rdm_2e`` or spin-block ``rdm_2e_upup``/``rdm_2e_dndn``/
        ``rdm_2e_updn``) in the MO basis.  Set to ``False`` to skip.
    ci_threshold : float, optional
        *MCSCF only.*  CI coefficient magnitude threshold below which
        determinants are omitted from the TREXIO output.  ``0.0`` (default)
        writes all determinants.
    chunk_size : int, optional
        *MCSCF only.*  Number of sparse ERI or determinant entries written
        per TREXIO call.  Larger values reduce I/O round-trips at the cost
        of memory.  Default ``100000``.

    Raises
    ------
    NotImplementedError
        When *obj* is not one of the supported PySCF types.

    Notes
    -----
    * PBC calculations must be Gamma-point only.  k-point objects raise
      ``NotImplementedError`` inside the underlying write routines.
    * Complex-valued integrals are not supported; a ``ValueError`` or
      ``NotImplementedError`` is raised if complex data is encountered.
    * For MCSCF objects the SCF mean-field attributes are written before
      the CASSCF/CASCI-specific data; the complete MO coefficient matrix
      (core + active + virtual) is stored.
    * *write_ao_eri* and *write_mo_eri* write to different TREXIO slots
      (``ao_2e_int_eri`` vs ``mo_2e_int_eri``) and can both be ``True``
      in the same call; the file will contain both AO and MO integrals.

    Examples
    --------
    **1. Export only molecular geometry and basis (no SCF)**

    >>> from pyscf import gto
    >>> from pyscf.tools import trexio
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
    >>> trexio.to_trexio(mol, 'hf_mol.h5')

    **2. Export SCF results without integrals or density matrices**

    >>> from pyscf import gto, scf
    >>> from pyscf.tools import trexio
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
    >>> mf = scf.RHF(mol).run()
    >>> trexio.to_trexio(mf, 'hf_scf.h5',
    ...     write_ao_eri=False, write_mo_eri=False, eri_sym='s1', write_mo_rdm=False)

    **3. Export SCF results with MO integrals and density matrices**

    >>> from pyscf import gto, scf
    >>> from pyscf.tools import trexio
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
    >>> mf = scf.RHF(mol).run()
    >>> trexio.to_trexio(
    ...     mf, 'hf_full.h5',
    ...     write_ao_eri=False, write_mo_eri=True, eri_sym='s4',
    ...     write_mo_rdm=True,
    ... )

    **4. Export CASSCF results with active-space integrals and density matrices**

    >>> from pyscf import gto, scf, mcscf
    >>> from pyscf.tools import trexio
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
    >>> mf = scf.RHF(mol).run()
    >>> mc = mcscf.CASSCF(mf, 6, 6).run()
    >>> trexio.to_trexio(
    ...     mc, 'cas.h5',
    ...     write_mcscf_eri=True,
    ...     write_mcscf_rdm=True,
    ...     ci_threshold=1e-3,
    ... )
    """
    if os.path.isfile(filename):
        os.remove(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)
    back_end = _trexio_backend_const(backend)
    if isinstance(obj, gto.Mole) or isinstance(obj, pbcgto.Cell):
        with trexio.File(filename, "u", back_end=back_end) as tf:
            _mol_to_trexio(obj, tf)
    elif isinstance(obj, scf.hf.SCF):
        with trexio.File(filename, "u", back_end=back_end) as tf:
            _scf_to_trexio(obj, tf)
            if write_ao_eri:
                _write_2e_eri(obj, tf, basis='AO', sym=eri_sym)
                _write_1e_eri(obj, tf, basis='AO')
            if write_mo_eri:
                _write_2e_eri(obj, tf, basis='MO', sym=eri_sym)
                _write_1e_eri(obj, tf, basis='MO')
            if write_mo_rdm:
                _write_1b_rdm(obj, tf)
                _write_2b_rdm(obj, tf)
    elif isinstance(obj, (casci.CASCI, mc1step.CASSCF, mcscf.ucasci.UCASCI, mcscf.umc1step.UCASSCF)):
        ci_threshold = ci_threshold if ci_threshold is not None else 0.
        chunk_size = chunk_size if chunk_size is not None else 100000
        with trexio.File(filename, "u", back_end=back_end) as tf:
            _mcscf_to_trexio(obj, tf, ci_threshold=ci_threshold, chunk_size=chunk_size,
                              write_eri=write_mcscf_eri, write_rdm=write_mcscf_rdm)
    else:
        raise NotImplementedError(f"Conversion function for {obj.__class__}")


def from_trexio(filename):
    """Reconstruct a PySCF molecule or crystal object from a TREXIO file.

    Reads nuclear geometry, basis set, spin, symmetry, ECP (if present), and
    — for periodic systems — the lattice vectors from *filename*, and returns
    a fully built :class:`pyscf.gto.Mole` or :class:`pyscf.pbc.gto.Cell`
    object.  The basis shells are restored in the original ordering so that
    the AO index convention matches the one used when the file was written.

    Parameters
    ----------
    filename : str
        Path to the TREXIO file (HDF5 binary) or directory (text backend).
        The backend is detected automatically from the file format.

    Returns
    -------
    mol : gto.Mole or pbc.gto.Cell
        A built PySCF object.  The unit is set to ``'Bohr'``, matching the
        TREXIO convention.  For periodic systems the lattice vectors are read
        from the ``cell_a/b/c`` fields and stored in ``mol.a``.

    Raises
    ------
    AssertionError
        If the ``basis_type`` stored in the file is not ``"Gaussian"``.
    trexio.Error
        If a required TREXIO field is missing or the file cannot be opened.

    Notes
    -----
    * Coordinates are read in Bohr and stored with ``mol.unit = 'Bohr'``.
    * Each nucleus is assigned a unique label of the form ``<symbol><index>``
      (e.g. ``"H0"``, ``"H1"``, ``"O2"``) so that atoms of the same element
      can carry different basis sets or ECPs.
    * The function sets ``mol._basis`` directly and bypasses the standard
      :meth:`~pyscf.gto.Mole.build` basis-sorting step, preserving the shell
      ordering encoded in the TREXIO file.
    * ECP data, if present, is decoded from the TREXIO storage convention
      (``power``  = :math:`r`-exponent − 2, local channel encoded as
      ``max_ang_mom_plus_1``) back to PySCF format.

    Examples
    --------
    **1. Round-trip a molecular RHF calculation**

    >>> from pyscf import gto, scf
    >>> from pyscf.tools import trexio
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.8', basis='cc-pvdz', verbose=0)
    >>> mf = scf.RHF(mol).run()
    >>> trexio.to_trexio(mf, 'hf.h5')
    >>> mol2 = trexio.from_trexio('hf.h5')
    >>> mol2.natm
    2

    **2. Round-trip a periodic RHF calculation**

    >>> import numpy as np
    >>> from pyscf.pbc import gto as pbcgto, scf as pbcscf
    >>> from pyscf.tools import trexio
    >>> cell = pbcgto.Cell()
    >>> cell.build(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g',
    ...            a=np.diag([3.0, 3.0, 5.0]), verbose=0)
    >>> mf = pbcscf.RHF(cell).run()
    >>> trexio.to_trexio(mf, 'cell.h5')
    >>> cell2 = trexio.from_trexio('cell.h5')
    >>> isinstance(cell2, pbcgto.Cell)
    True
    """
    with trexio.File(filename, "r", back_end=trexio.TREXIO_AUTO) as tf:
        assert trexio.read_basis_type(tf) == "Gaussian"
        pbc_periodic = trexio.read_pbc_periodic(tf)
        labels = trexio.read_nucleus_label(tf)
        coords = trexio.read_nucleus_coord(tf)
        elements = [s + str(i) for i, s in enumerate(labels)]

        if pbc_periodic:
            mol = pbcgto.Cell()
            mol.unit = "Bohr"
            a = np.asarray(trexio.read_cell_a(tf), dtype=float)
            b = np.asarray(trexio.read_cell_b(tf), dtype=float)
            c = np.asarray(trexio.read_cell_c(tf), dtype=float)
            mol.a = np.vstack([a, b, c])
        else:
            mol = gto.Mole()
            mol.unit = "Bohr"

        mol.atom = list(zip(elements, coords))
        up_num = trexio.read_electron_up_num(tf)
        dn_num = trexio.read_electron_dn_num(tf)
        spin = up_num - dn_num
        mol.spin = spin

        if trexio.has_ecp(tf):
            # --- read TREXIO ECP arrays ---
            z_core = trexio.read_ecp_z_core(tf)  # shape (natm,)
            max_l1_arr = trexio.read_ecp_max_ang_mom_plus_1(tf)  # shape (natm,)
            nuc_idx = trexio.read_ecp_nucleus_index(tf)  # shape (ecp_num,)
            ang_mom_enc = trexio.read_ecp_ang_mom(tf)  # shape (ecp_num,)
            coeff_arr = trexio.read_ecp_coefficient(tf)  # shape (ecp_num,)
            exp_arr = trexio.read_ecp_exponent(tf)  # shape (ecp_num,)
            power_arr = trexio.read_ecp_power(tf)  # shape (ecp_num,)

            # --- aggregate primitives per (nucleus, l); decode local channel to l = -1 ---
            per_atom = defaultdict(
                lambda: defaultdict(list)
            )  # per_atom[n][l] -> [(power, exp, coeff), ...]
            for k in range(len(nuc_idx)):
                n = int(nuc_idx[k])
                l_e = int(ang_mom_enc[k])
                # local channel was encoded as max_l1_arr[n]; decode to l = -1
                l_d = -1 if l_e == int(max_l1_arr[n]) else l_e
                p = int(power_arr[k])  # stored as r-2 on write
                e = float(exp_arr[k])
                c = float(coeff_arr[k])
                per_atom[n][l_d].append((p, e, c))

            def _is_dummy_ul_s(items):
                # H/He dummy record injected at write time: (power=0, exp=1.0, coeff=0.0)
                return (
                    len(items) == 1
                    and items[0][0] == 0
                    and abs(items[0][1] - 1.0) < 1e-12
                    and abs(items[0][2]) < 1e-300
                )

            ecp_dict = {}
            for n, raw_sym in enumerate(labels):
                # skip ghosts: ECP not applied to 'X-*' atoms
                if re.match(r"X-.*", raw_sym):
                    continue

                zc = int(z_core[n]) if n < len(z_core) else 0
                lmap = dict(per_atom.get(n, {}))

                # drop intentional dummy ul-s for H/He if present
                if 0 in lmap and _is_dummy_ul_s(lmap[0]):
                    lmap.pop(0)

                # nothing to assign if no channels and no core reduction
                if not lmap and zc == 0:
                    continue

                # Build PySCF format for this atom:
                # for each l, buckets[r] holds (exp, coeff), where r = power + 2
                at_list = []
                # ordering is not strictly required, but keep nonlocal (l>=0) then local (l=-1) last
                for l in sorted(lmap.keys(), key=lambda x: (x == -1, x)):
                    items = lmap[l]
                    if not items:
                        continue
                    r_list = [p + 2 for p, _, _ in items]
                    max_r = max(r_list)
                    buckets = [[] for _ in range(max_r + 1)]  # r in [0..max_r]
                    for (p, e, c), r in zip(items, r_list):
                        if r < 0:
                            # should not happen if power = r-2 and r>=0
                            continue
                        buckets[r].append((e, c))
                    at_list.append([int(l), buckets])

                # ★ Use the exact same per-atom label as in mol.atom / mol._basis
                key = elements[n]
                ecp_dict[key] = (zc, at_list)

            if ecp_dict:
                mol.ecp = ecp_dict

        if trexio.has_ao_cartesian(tf):
            mol.cart = trexio.read_ao_cartesian(tf) == 1

        if trexio.has_nucleus_point_group(tf):
            mol.symmetry = trexio.read_nucleus_point_group(tf)

        nuc_idx = trexio.read_basis_nucleus_index(tf).tolist()
        ls = trexio.read_basis_shell_ang_mom(tf).tolist()
        prim2sh = trexio.read_basis_shell_index(tf).tolist()
        exps = trexio.read_basis_exponent(tf).tolist()
        coef = trexio.read_basis_coefficient(tf).tolist()

        basis = {}
        exps = _group_by(exps, prim2sh)
        coef = _group_by(coef, prim2sh)
        p1 = 0
        for ia, at_ls in enumerate(_group_by(ls, nuc_idx)):
            p0, p1 = p1, p1 + at_ls.size
            at_basis = [
                [l, *zip(e, c)] for l, e, c in zip(ls[p0:p1], exps[p0:p1], coef[p0:p1])
            ]
            basis[elements[ia]] = at_basis

        # To avoid the mol.build() sort the basis, disable mol.basis and set the
        # internal data _basis directly.
        mol.basis = {}
        mol._basis = basis
        return mol.build()


def _trexio_backend_const(backend='h5'):
    if isinstance(backend, int):
        return backend

    key = 'h5' if backend is None else str(backend).strip().lower()
    if key in ('h5', 'hdf5'):
        return trexio.TREXIO_HDF5
    if key in ('text', 'txt'):
        return trexio.TREXIO_TEXT
    if key == 'auto':
        return trexio.TREXIO_AUTO
    raise ValueError("backend must be one of 'h5', 'hdf5', 'text', 'txt', or 'auto'")


def _mol_to_trexio(mol, trexio_file):
    # 1 Metadata
    trexio.write_metadata_code_num(trexio_file, 1)
    trexio.write_metadata_code(trexio_file, [f"PySCF-v{pyscf.__version__}"])

    # 2 System
    trexio.write_nucleus_num(trexio_file, mol.natm)
    nucleus_charge = [mol.atom_charge(i) for i in range(mol.natm)]
    trexio.write_nucleus_charge(trexio_file, nucleus_charge)
    trexio.write_nucleus_coord(trexio_file, mol.atom_coords())
    labels = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
    trexio.write_nucleus_label(trexio_file, labels)
    if mol.symmetry:
        trexio.write_nucleus_point_group(trexio_file, mol.groupname)

    # 2.3 Periodic boundary calculations
    if isinstance(mol, pbc.gto.Cell):
        # 2.3 Periodic boundary calculations (pbc group)
        trexio.write_pbc_periodic(trexio_file, True)
        # 2.2 Cell
        lattice = mol.lattice_vectors()
        a = lattice[0]  # unit is Bohr
        b = lattice[1]  # unit is Bohr
        c = lattice[2]  # unit is Bohr
        trexio.write_cell_a(trexio_file, a)
        trexio.write_cell_b(trexio_file, b)
        trexio.write_cell_c(trexio_file, c)
    else:
        # 2.3 Periodic boundary calculations (pbc group)
        trexio.write_pbc_periodic(trexio_file, False)

    # 2.4 Electron (electron group)
    electron_up_num, electron_dn_num = mol.nelec
    trexio.write_electron_num(trexio_file, electron_up_num + electron_dn_num)
    trexio.write_electron_up_num(trexio_file, electron_up_num)
    trexio.write_electron_dn_num(trexio_file, electron_dn_num)

    # 2.5 Nuclear repulsion energy
    try:
        trexio.write_nucleus_repulsion(trexio_file, mol.energy_nuc())
    except trexio.Error:
        # TREXIO ver 2.6.0 bug. See #231. A negative value is not accepted.
        trexio.write_nucleus_repulsion(trexio_file, 0.0)

    # 3.1 Basis set
    trexio.write_basis_type(trexio_file, "Gaussian")
    if any(mol._bas[:, gto.NCTR_OF] > 1):
        mol = _to_segment_contraction(mol)
    trexio.write_basis_shell_num(trexio_file, mol.nbas)
    trexio.write_basis_prim_num(trexio_file, int(mol._bas[:, gto.NPRIM_OF].sum()))
    trexio.write_basis_nucleus_index(trexio_file, mol._bas[:, gto.ATOM_OF])
    trexio.write_basis_shell_ang_mom(trexio_file, mol._bas[:, gto.ANG_OF])
    trexio.write_basis_shell_factor(trexio_file, np.ones(mol.nbas))
    prim2sh = [[ib] * nprim for ib, nprim in enumerate(mol._bas[:, gto.NPRIM_OF])]
    trexio.write_basis_shell_index(trexio_file, np.hstack(prim2sh))
    trexio.write_basis_exponent(trexio_file, np.hstack(mol.bas_exps()))
    coef = [mol.bas_ctr_coeff(i).ravel() for i in range(mol.nbas)]
    trexio.write_basis_coefficient(trexio_file, np.hstack(coef))
    prim_norms = [
        gto.gto_norm(mol.bas_angular(i), mol.bas_exp(i)) for i in range(mol.nbas)
    ]
    trexio.write_basis_prim_factor(trexio_file, np.hstack(prim_norms))

    # 3.2 Effective core potentials (ecp group)
    if len(mol._pseudo) > 0:
        raise NotImplementedError(
            "TREXIO does not support 'pseudo' format. Please use 'ecp'"
        )

    if mol._ecp:
        # internal format of pyscf is described in
        # https://pyscf.org/pyscf_api_docs/pyscf.gto.html?highlight=ecp#module-pyscf.gto.ecp

        ecp_num = 0
        ecp_max_ang_mom_plus_1 = []
        ecp_z_core = []
        ecp_nucleus_index = []
        ecp_ang_mom = []
        ecp_coefficient = []
        ecp_exponent = []
        ecp_power = []

        chemical_symbol_list = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        atom_symbol_list = [mol.atom_symbol(i) for i in range(mol.natm)]

        for nuc_index, (chemical_symbol, atom_symbol) in enumerate(
            zip(chemical_symbol_list, atom_symbol_list)
        ):
            if re.match(r"X-.*", chemical_symbol) or re.match(r"X-.*", atom_symbol):
                ecp_num += 1
                ecp_max_ang_mom_plus_1.append(1)
                ecp_z_core.append(0)
                ecp_nucleus_index.append(nuc_index)
                ecp_ang_mom.append(0)
                ecp_coefficient.append(0.0)
                ecp_exponent.append(1.0)
                ecp_power.append(0)
                continue

            # atom_symbol is superior to atom_pure_symbol
            try:
                z_core, ecp_list = mol._ecp[atom_symbol]
            except KeyError:
                z_core, ecp_list = mol._ecp[chemical_symbol]

            # ecp zcore
            ecp_z_core.append(z_core)

            # max_ang_mom
            max_ang_mom = max([ecp[0] for ecp in ecp_list])
            if max_ang_mom == -1:
                # Special treatments are needed for H and He.
                # PySCF database does not define the ul-s part for these elements.
                dummy_ul_s_part = True
                max_ang_mom = 0
                max_ang_mom_plus_1 = 1
            else:
                dummy_ul_s_part = False
                max_ang_mom_plus_1 = max_ang_mom + 1

            ecp_max_ang_mom_plus_1.append(max_ang_mom_plus_1)

            for ecp in ecp_list:
                ang_mom = ecp[0]
                if ang_mom == -1:
                    ang_mom = max_ang_mom_plus_1
                for r, exp_coeff_list in enumerate(ecp[1]):
                    for exp_coeff in exp_coeff_list:
                        exp, coeff = exp_coeff
                        ecp_num += 1
                        ecp_nucleus_index.append(nuc_index)
                        ecp_ang_mom.append(ang_mom)
                        ecp_coefficient.append(coeff)
                        ecp_exponent.append(exp)
                        ecp_power.append(r - 2)

            if dummy_ul_s_part:
                # A dummy ECP is put for the ul-s part here for H and He atoms.
                ecp_num += 1
                ecp_nucleus_index.append(nuc_index)
                ecp_ang_mom.append(0)
                ecp_coefficient.append(0.0)
                ecp_exponent.append(1.0)
                ecp_power.append(0)

        trexio.write_ecp_num(trexio_file, ecp_num)
        trexio.write_ecp_max_ang_mom_plus_1(trexio_file, ecp_max_ang_mom_plus_1)
        trexio.write_ecp_z_core(trexio_file, ecp_z_core)
        trexio.write_ecp_nucleus_index(trexio_file, ecp_nucleus_index)
        trexio.write_ecp_ang_mom(trexio_file, ecp_ang_mom)
        trexio.write_ecp_coefficient(trexio_file, ecp_coefficient)
        trexio.write_ecp_exponent(trexio_file, ecp_exponent)
        trexio.write_ecp_power(trexio_file, ecp_power)

    # 4.1 Atomic orbitals (ao group)
    if mol.cart:
        trexio.write_ao_cartesian(trexio_file, 1)
        ao_shell = []
        ao_normalization = []
        for i, ang_mom in enumerate(mol._bas[:, gto.ANG_OF]):
            ao_shell += [i] * int((ang_mom + 1) * (ang_mom + 2) / 2)
            # note: PySCF(libintc) normalizes s and p only.
            if ang_mom == 0 or ang_mom == 1:
                ao_normalization += [
                    float(np.sqrt((2 * ang_mom + 1) / (4 * np.pi)))
                ] * int((ang_mom + 1) * (ang_mom + 2) / 2)
            else:
                ao_normalization += [1.0] * int((ang_mom + 1) * (ang_mom + 2) / 2)
    else:
        trexio.write_ao_cartesian(trexio_file, 0)
        ao_shell = []
        ao_normalization = []
        for i, ang_mom in enumerate(mol._bas[:, gto.ANG_OF]):
            ao_shell += [i] * (2 * ang_mom + 1)
            # note: TREXIO employs the solid harmonics notation,; thus, we need these factors.
            ao_normalization += [float(np.sqrt((2 * ang_mom + 1) / (4 * np.pi)))] * (
                2 * ang_mom + 1
            )

    trexio.write_ao_num(trexio_file, int(mol.nao))
    trexio.write_ao_shell(trexio_file, ao_shell)
    trexio.write_ao_normalization(trexio_file, ao_normalization)


def _scf_to_trexio(mf, trexio_file):
    mol = mf.mol
    _mol_to_trexio(mol, trexio_file)

    # PBC
    if isinstance(mol, pbc.gto.Cell):
        kpts = np.array(mf.kpts)
        kpts = mol.get_scaled_kpts(kpts)
        nk = len(mf.kpts)
        weights = np.full(nk, 1.0/nk)
        madelung = pbctools.pbc.madelung(mol, kpts)

        if nk == 1:
            # 2.3 Periodic boundary calculations (pbc group)
            trexio.write_pbc_k_point_num(trexio_file, 1)
            trexio.write_pbc_k_point(trexio_file, kpts)
            trexio.write_pbc_k_point_weight(trexio_file, weights[np.newaxis])
            trexio.write_pbc_madelung(trexio_file, madelung)

            if _trexio_is_uhf_uks_mf(mf):
                mo_type = "UHF"
                mo_energy = np.ravel(mf.mo_energy)
                mo_num = mo_energy.size
                mo_up, mo_dn = mf.mo_coeff
                if np.ndim(mo_up) == 3:
                    mo_up = mo_up[0]
                    mo_dn = mo_dn[0]
                idx = _order_ao_index(mf.mol)
                mo_up = mo_up[idx].T
                mo_dn = mo_dn[idx].T
                num_mo_up = len(mo_up)
                num_mo_dn = len(mo_dn)
                assert num_mo_up + num_mo_dn == mo_num
                mo = np.concatenate(
                    [mo_up, mo_dn], axis=0
                )  # dim (num_mo, num_ao) but it is f-contiguous
                mo_coefficient = np.ascontiguousarray(
                    mo
                )  # dim (num_mo, num_ao) and it is c-contiguous
                if np.all(np.isreal(mo_coefficient)):
                    mo_coefficient_real = mo_coefficient
                    mo_coefficient_imag = None
                else:
                    mo_coefficient_real = mo_coefficient.real
                    mo_coefficient_imag = mo_coefficient.imag
                mo_occ = np.ravel(mf.mo_occ)
                mo_spin = np.zeros(num_mo_up + num_mo_dn, dtype=int)
                mo_spin[:num_mo_up] = 0
                mo_spin[num_mo_up:] = 1

            elif _trexio_is_rohf_roks_mf(mf) or _trexio_is_rhf_rks_mf(mf):
                mo_type = "ROHF" if _trexio_is_rohf_roks_mf(mf) else "RHF"
                mo_energy = np.ravel(mf.mo_energy)
                mo_num = mo_energy.size
                mo = mf.mo_coeff
                if np.ndim(mo) == 3:
                    mo = mo[0]
                idx = _order_ao_index(mf.mol)
                mo = mo[idx].T  # dim (num_mo, num_ao) but it is f-contiguous
                mo_coefficient = np.ascontiguousarray(
                    mo
                )  # dim (num_mo, num_ao) and it is c-contiguous
                if np.all(np.isreal(mo_coefficient)):
                    mo_coefficient_real = mo_coefficient
                    mo_coefficient_imag = None
                else:
                    mo_coefficient_real = mo_coefficient.real
                    mo_coefficient_imag = mo_coefficient.imag
                mo_occ = np.ravel(mf.mo_occ)
                mo_spin = np.zeros(mo_num, dtype=int)
            else:
                raise NotImplementedError(f"Conversion function for {mf.__class__}")

            # 4.2 Molecular orbitals (mo group)
            trexio.write_mo_type(trexio_file, mo_type)
            trexio.write_mo_num(trexio_file, mo_num)
            trexio.write_mo_k_point(trexio_file, [0] * mo_num)
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)
            if mo_coefficient_imag is not None:
                trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag)
            trexio.write_mo_energy(trexio_file, mo_energy)
            trexio.write_mo_occupation(trexio_file, mo_occ)
            trexio.write_mo_spin(trexio_file, mo_spin)

        else:
            # 2.3 Periodic boundary calculations (pbc group)
            trexio.write_pbc_k_point_num(trexio_file, nk)
            trexio.write_pbc_k_point(trexio_file, kpts)
            trexio.write_pbc_k_point_weight(trexio_file, weights)
            trexio.write_pbc_madelung(trexio_file, madelung)

            # stack k-dependent molecular orbitals
            mo_k_point_pbc = []
            mo_num_pbc = 0
            mo_energy_pbc = []
            mo_coefficient_real_pbc = []
            mo_coefficient_imag_pbc = []
            mo_occ_pbc = []
            mo_spin_pbc = []

            if _trexio_is_uhf_uks_mf(mf):
                mo_type = "UHF"
                # Check for split structure (common in KUKS/KDF): ([up...], [dn...])
                is_split_spin = (mf.mo_coeff.ndim==4) and (mf.mo_coeff.shape[0]==2)

                for i_k, _ in enumerate(kpts):
                    if is_split_spin:
                        e_up = mf.mo_energy[0][i_k]
                        e_dn = mf.mo_energy[1][i_k]
                        mo_energy = np.concatenate([np.ravel(e_up), np.ravel(e_dn)])

                        mo_up = mf.mo_coeff[0][i_k]
                        mo_dn = mf.mo_coeff[1][i_k]

                        occ_up = mf.mo_occ[0][i_k]
                        occ_dn = mf.mo_occ[1][i_k]
                        mo_occ = np.concatenate([np.ravel(occ_up), np.ravel(occ_dn)])
                    else:
                        mo_energy = np.ravel(mf.mo_energy[i_k])
                        mo_up, mo_dn = mf.mo_coeff[i_k]
                        mo_occ = np.ravel(mf.mo_occ[i_k])

                    mo_num = mo_energy.size
                    idx = _order_ao_index(mf.mol)
                    mo_up = mo_up[idx].T
                    mo_dn = mo_dn[idx].T
                    mo = np.concatenate(
                        [mo_up, mo_dn], axis=0
                    )  # dim (num_mo, num_ao) but it is f-contiguous
                    mo_coefficient_real = np.ascontiguousarray(
                        mo.real
                    )  # dim (num_mo, num_ao) and it is c-contiguous
                    mo_coefficient_imag = np.ascontiguousarray(
                        mo.imag
                    )  # dim (num_mo, num_ao) and it is c-contiguous

                    mo_spin = np.zeros(mo_energy.size, dtype=int)
                    # Use size of up component to split spin
                    if is_split_spin:
                        mo_spin[e_up.size :] = 1
                    else:
                        # Assumes mo_energy was flattened from (up, dn)
                        # If mo_energy[i_k] is tuple (up, dn), ravel joins them.
                        # We need size of up.
                        if isinstance(mf.mo_energy[i_k], (tuple, list)):
                            mo_spin[mf.mo_energy[i_k][0].size :] = 1
                        else:
                            # If somehow not tuple?
                            # Fallback to half?
                            mo_spin[mo_energy.size // 2 :] = 1

                    mo_k_point_pbc += [i_k] * mo_energy.size
                    mo_num_pbc += mo_energy.size
                    mo_energy_pbc.append(mo_energy)
                    mo_coefficient_real_pbc.append(mo_coefficient_real)
                    mo_coefficient_imag_pbc.append(mo_coefficient_imag)
                    mo_occ_pbc.append(mo_occ)
                    mo_spin_pbc.append(mo_spin)

            elif _trexio_is_rohf_roks_mf(mf) or _trexio_is_rhf_rks_mf(mf):
                mo_type = "ROHF" if _trexio_is_rohf_roks_mf(mf) else "RHF"
                for i_k, _ in enumerate(kpts):
                    mo_energy = mf.mo_energy[i_k]
                    mo = mf.mo_coeff[i_k]
                    idx = _order_ao_index(mf.mol)
                    mo = mo[idx].T
                    mo_coefficient_real = np.ascontiguousarray(
                        mo.real
                    )  # dim (num_mo, num_ao) and it is c-contiguous
                    mo_coefficient_imag = np.ascontiguousarray(
                        mo.imag
                    )  # dim (num_mo, num_ao) and it is c-contiguous
                    mo_occ = np.ravel(mf.mo_occ[i_k])
                    mo_spin = np.zeros(mo_energy.size, dtype=int)

                    mo_k_point_pbc += [i_k] * mo_energy.size
                    mo_num_pbc += mo_energy.size
                    mo_energy_pbc.append(mo_energy)
                    mo_coefficient_real_pbc.append(mo_coefficient_real)
                    mo_coefficient_imag_pbc.append(mo_coefficient_imag)
                    mo_occ_pbc.append(mo_occ)
                    mo_spin_pbc.append(mo_spin)

            else:
                raise NotImplementedError(f"Conversion function for {mf.__class__}")

            # stack the results
            mo_k_point_pbc = np.array(mo_k_point_pbc)
            mo_energy_pbc = np.concatenate(mo_energy_pbc)
            mo_coefficient_real_pbc = np.ascontiguousarray(
                np.vstack(mo_coefficient_real_pbc)
            )  # it is c-contiguous
            mo_coefficient_imag_pbc = np.ascontiguousarray(
                np.vstack(mo_coefficient_imag_pbc)
            )  # it is c-contiguous
            mo_occ_pbc = np.concatenate(mo_occ_pbc)
            mo_spin_pbc = np.concatenate(mo_spin_pbc)

            # 4.2 Molecular orbitals (mo group)
            trexio.write_mo_type(trexio_file, mo_type)
            trexio.write_mo_num(trexio_file, mo_num_pbc)
            trexio.write_mo_k_point(trexio_file, mo_k_point_pbc)
            trexio.write_mo_energy(trexio_file, mo_energy_pbc)
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real_pbc)
            trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag_pbc)
            trexio.write_mo_occupation(trexio_file, mo_occ_pbc)
            trexio.write_mo_spin(trexio_file, mo_spin_pbc)

    # Open systems
    else:
        if _trexio_is_uhf_uks_mf(mf):
            mo_type = "UHF"
            mo_energy = np.ravel(mf.mo_energy)
            mo_num = mo_energy.size
            mo_up, mo_dn = mf.mo_coeff
            idx = _order_ao_index(mf.mol)
            mo_up = mo_up[idx].T
            mo_dn = mo_dn[idx].T
            mo = np.concatenate(
                [mo_up, mo_dn], axis=0
            )  # dim (num_mo, num_ao) but it is f-contiguous
            mo_coefficient = np.ascontiguousarray(
                mo
            )  # dim (num_mo, num_ao) and it is c-contiguous
            mo_occ = np.ravel(mf.mo_occ)
            mo_spin = np.zeros(mo_energy.size, dtype=int)
            mo_spin[mf.mo_energy[0].size :] = 1
        elif _trexio_is_rohf_roks_mf(mf) or _trexio_is_rhf_rks_mf(mf):
            mo_type = "ROHF" if _trexio_is_rohf_roks_mf(mf) else "RHF"
            mo_energy = mf.mo_energy
            mo_num = mo_energy.size
            mo = mf.mo_coeff
            idx = _order_ao_index(mf.mol)
            mo = mo[idx].T  # dim (num_mo, num_ao) but it is f-contiguous
            mo_coefficient = np.ascontiguousarray(
                mo
            )  # dim (num_mo, num_ao) and it is c-contiguous
            mo_occ = mf.mo_occ
            mo_spin = np.zeros(mo_energy.size, dtype=int)
        else:
            raise NotImplementedError(f"Conversion function for {mf.__class__}")

        # 4.2 Molecular orbitals (mo group)
        trexio.write_mo_type(trexio_file, mo_type)
        trexio.write_mo_num(trexio_file, mo_num)
        trexio.write_mo_coefficient(trexio_file, mo_coefficient)
        trexio.write_mo_energy(trexio_file, mo_energy)
        trexio.write_mo_occupation(trexio_file, mo_occ)
        trexio.write_mo_spin(trexio_file, mo_spin)


def _cc_to_trexio(cc_obj, trexio_file):
    raise NotImplementedError("CC conversion is not implemented yet.")


def _get_cas_rdm1s(cas_obj, ncas):
    neleca, nelecb = cas_obj.nelecas
    if hasattr(cas_obj.fcisolver, 'make_rdm1s'):
        rdm1a, rdm1b = cas_obj.fcisolver.make_rdm1s(cas_obj.ci, ncas, cas_obj.nelecas)
    else:
        rdm1 = cas_obj.fcisolver.make_rdm1(cas_obj.ci, ncas, cas_obj.nelecas)
        total = neleca + nelecb
        if total > 0:
            rdm1a = rdm1 * (neleca / total)
            rdm1b = rdm1 * (nelecb / total)
        else:
            rdm1a = rdm1 * 0.0
            rdm1b = rdm1 * 0.0
    return rdm1a, rdm1b


def _get_cas_rdm12(cas_obj, ncas):
    if hasattr(cas_obj.fcisolver, 'make_rdm12'):
        return cas_obj.fcisolver.make_rdm12(cas_obj.ci, ncas, cas_obj.nelecas)
    raise NotImplementedError(f'fcisolver {cas_obj.fcisolver.__class__} does not support make_rdm12')


def _get_cas_rdm12s(cas_obj, ncas):
    if hasattr(cas_obj.fcisolver, 'make_rdm12s'):
        out = cas_obj.fcisolver.make_rdm12s(cas_obj.ci, ncas, cas_obj.nelecas)
        if isinstance(out, (tuple, list)):
            if len(out) == 5:
                return out
            if len(out) == 2:
                dm1s, dm2s = out
                if isinstance(dm1s, (tuple, list)) and isinstance(dm2s, (tuple, list)):
                    if len(dm1s) == 2 and len(dm2s) == 3:
                        return dm1s[0], dm1s[1], dm2s[0], dm2s[1], dm2s[2]
        raise ValueError("Unexpected return from make_rdm12s")
    raise NotImplementedError(f'fcisolver {cas_obj.fcisolver.__class__} does not support make_rdm12s')


def _get_cas_h1eff(cas_obj):
    return cas_obj.get_h1eff()


def _get_cas_h2eff(cas_obj, ncas):
    h2eff = cas_obj.get_h2eff()
    if isinstance(h2eff, (tuple, list)):
        h2_blocks = []
        for block in h2eff:
            if block.ndim == 4:
                h2_blocks.append(block)
            else:
                h2_blocks.append(ao2mo.restore(1, block, ncas))
        return tuple(h2_blocks)
    if h2eff.ndim == 4:
        return h2eff
    return ao2mo.restore(1, h2eff, ncas)


def _write_sparse_4index(trexio_file, writer, idx, data):
    num = data.size
    writer(trexio_file, 0, num, idx.reshape(num, 4).astype(np.int32).ravel(), data.ravel())


def _ijkl_indices(n):
    grid = np.indices((n, n, n, n), dtype=np.int32)
    return grid.reshape(4, -1).T


def _gather_4index(data4, idx):
    return data4[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]


def _trexio_ncore_scalar(ncore):
    if isinstance(ncore, (tuple, list, np.ndarray)):
        if len(ncore) != 2:
            raise NotImplementedError(f"Unsupported ncore layout: {ncore}")
        if ncore[0] != ncore[1]:
            raise NotImplementedError(
                f"Different alpha/beta ncore is not supported: {ncore}"
            )
        return int(ncore[0])
    return int(ncore)


def _mcscf_to_trexio(cas_obj, trexio_file, ci_threshold=0., chunk_size=100000, write_eri=True, write_rdm=True):
    mol = cas_obj.mol
    scf_obj = cas_obj._scf
    _mol_to_trexio(mol, trexio_file)

    mo_energy_cas = getattr(cas_obj, 'mo_energy', None)
    mo_cas = cas_obj.mo_coeff
    idx = _order_ao_index(mol)

    ncore = _trexio_ncore_scalar(cas_obj.ncore)
    ncas = cas_obj.ncas

    is_uhf = _trexio_is_uhf_uks_mf(scf_obj)
    is_rhf = _trexio_is_rhf_rks_mf(scf_obj) or _trexio_is_rohf_roks_mf(scf_obj)
    if not (is_uhf or is_rhf):
        raise NotImplementedError(f"Conversion function for {scf_obj.__class__}")

    if is_uhf:
        mo_type_cas = 'CAS'
        mo_up, mo_dn = mo_cas
        mo_up = mo_up[idx].T
        mo_dn = mo_dn[idx].T
        num_mo_up = mo_up.shape[0]
        num_mo_dn = mo_dn.shape[0]
        num_mo = num_mo_up + num_mo_dn
        mo_coefficient = np.ascontiguousarray(np.concatenate([mo_up, mo_dn], axis=0))

        mo_energy = np.zeros(num_mo, dtype=float) #not well defined for UHF-CAS, so we set dummy values.

        mo_spin = np.zeros(num_mo, dtype=int)
        mo_spin[num_mo_up:] = 1

        mo_classes = np.array(["Virtual"] * num_mo, dtype=str)
        mo_classes[:ncore] = "Core"
        mo_classes[ncore:ncore + ncas] = "Active"
        beta_start = num_mo_up
        mo_classes[beta_start:beta_start + ncore] = "Core"
        mo_classes[beta_start + ncore:beta_start + ncore + ncas] = "Active"

        occ_alpha = np.zeros(num_mo_up)
        occ_beta = np.zeros(num_mo_dn)
        occ_alpha[:ncore] = 1.0
        occ_beta[:ncore] = 1.0
        rdm1a, rdm1b = _get_cas_rdm1s(cas_obj, ncas)
        nat_occ_a = np.linalg.eigvalsh(rdm1a)[::-1]
        nat_occ_b = np.linalg.eigvalsh(rdm1b)[::-1]
        occ_alpha[ncore:ncore + ncas] = nat_occ_a
        occ_beta[ncore:ncore + ncas] = nat_occ_b
        occupation = np.concatenate([occ_alpha, occ_beta])

    else:
        mo_type_cas = 'CAS'
        mo_energy = np.ravel(mo_energy_cas)
        num_mo = mo_energy.size
        mo = mo_cas[idx].T
        mo_coefficient = np.ascontiguousarray(mo)
        mo_spin = np.zeros(num_mo, dtype=int)

        mo_classes = np.array(["Virtual"] * num_mo, dtype=str)
        mo_classes[:ncore] = "Core"
        mo_classes[ncore:ncore + ncas] = "Active"

        occupation = np.zeros(num_mo)
        occupation[:ncore] = 2.0
        rdm1 = cas_obj.fcisolver.make_rdm1(cas_obj.ci, ncas, cas_obj.nelecas)
        natural_occ = np.linalg.eigvalsh(rdm1)[::-1]
        occupation[ncore:ncore + ncas] = natural_occ
        occupation[ncore + ncas:] = 0.0

    trexio.write_mo_type(trexio_file, mo_type_cas)
    trexio.write_mo_num(trexio_file, num_mo)
    trexio.write_mo_coefficient(trexio_file, mo_coefficient)
    trexio.write_mo_energy(trexio_file, mo_energy)
    trexio.write_mo_spin(trexio_file, mo_spin)
    trexio.write_mo_class(trexio_file, list(mo_classes))
    trexio.write_mo_occupation(trexio_file, occupation)

    nelec_cas = cas_obj.nelecas

    _det_to_trexio(cas_obj, ncas, nelec_cas, trexio_file, ci_threshold, chunk_size)

    if is_uhf:
        if write_rdm:
            dm1a, dm1b, dm2aa, dm2ab, dm2bb = _get_cas_rdm12s(cas_obj, ncas)

            rdm1_up = np.zeros((num_mo, num_mo))
            rdm1_dn = np.zeros((num_mo, num_mo))
            rdm1_up[ncore:ncore + ncas, ncore:ncore + ncas] = dm1a
            beta_start = num_mo_up
            rdm1_dn[beta_start + ncore:beta_start + ncore + ncas,
                    beta_start + ncore:beta_start + ncore + ncas] = dm1b

            if trexio.has_rdm_1e_up(trexio_file):
                trexio.delete_rdm_1e_up(trexio_file)
            if trexio.has_rdm_1e_dn(trexio_file):
                trexio.delete_rdm_1e_dn(trexio_file)
            trexio.write_rdm_1e_up(trexio_file, rdm1_up)
            trexio.write_rdm_1e_dn(trexio_file, rdm1_dn)

            idx_uu_base = _ijkl_indices(ncas)
            data_uu = _gather_4index(dm2aa, idx_uu_base)
            idx_uu = idx_uu_base[:, [2, 3, 0, 1]] + ncore
            _write_sparse_4index(
                trexio_file,
                trexio.write_rdm_2e_upup,
                idx_uu,
                data_uu,
            )

            idx_dd_base = _ijkl_indices(ncas)
            data_dd = _gather_4index(dm2bb, idx_dd_base)
            idx_dd = idx_dd_base[:, [2, 3, 0, 1]] + beta_start + ncore
            _write_sparse_4index(
                trexio_file,
                trexio.write_rdm_2e_dndn,
                idx_dd,
                data_dd,
            )

            idx_ud_base = _ijkl_indices(ncas)
            data_ud = _gather_4index(dm2ab, idx_ud_base)
            idx_ud = idx_ud_base[:, [2, 3, 0, 1]].copy()
            idx_ud[:, 0] += ncore
            idx_ud[:, 1] += beta_start + ncore
            idx_ud[:, 2] += ncore
            idx_ud[:, 3] += beta_start + ncore
            _write_sparse_4index(
                trexio_file,
                trexio.write_rdm_2e_updn,
                idx_ud,
                data_ud,
            )

    else:
        if write_rdm or write_eri:
            idx_base = _ijkl_indices(ncas)
            idx = idx_base + ncore
            idx_phys = idx[:, [2, 3, 0, 1]]

        if write_rdm:
            dm1_cas, dm2_cas = _get_cas_rdm12(cas_obj, ncas)

            dm1_full = np.zeros((num_mo, num_mo))
            dm1_full[ncore:ncore + ncas, ncore:ncore + ncas] = dm1_cas

            if trexio.has_rdm_1e(trexio_file):
                trexio.delete_rdm_1e(trexio_file)
            trexio.write_rdm_1e(trexio_file, dm1_full)

            data_rdm2 = _gather_4index(dm2_cas, idx_base)
            # Physicist notation: store as (k, l, i, j) to match rdm2 layout and UHF blocks.
            _write_sparse_4index(
                trexio_file,
                trexio.write_rdm_2e,
                idx_phys,
                data_rdm2,
            )

        if write_eri:
            h1eff, _ = _get_cas_h1eff(cas_obj)
            h2eff = _get_cas_h2eff(cas_obj, ncas)

            h1_full = np.zeros((num_mo, num_mo))
            h1_full[ncore:ncore + ncas, ncore:ncore + ncas] = h1eff

            if trexio.has_mo_1e_int_core_hamiltonian(trexio_file):
                trexio.delete_mo_1e_int_core_hamiltonian(trexio_file)
            trexio.write_mo_1e_int_core_hamiltonian(trexio_file, h1_full)

            data_h2 = _gather_4index(h2eff, idx_base)
            _write_sparse_4index(
                trexio_file,
                trexio.write_mo_2e_int_eri,
                idx_phys,
                data_h2,
            )


def _trexio_is_gamma_single_k(obj) -> bool:
    if not hasattr(obj, 'cell'):
        return False
    if hasattr(obj, 'kpt'):
        return np.allclose(np.asarray(obj.kpt), 0.0)
    if hasattr(obj, 'kpts'):
        kpts = np.asarray(obj.kpts)
        return (kpts.shape[0] == 1) and np.allclose(kpts[0], 0.0)
    return True


def _trexio_get_rhf_coeff_matrix(mf_obj, *, expect_gamma=False):
    coeff = mf_obj.mo_coeff
    if isinstance(coeff, np.ndarray):
        if coeff.ndim == 2:
            return coeff
        if expect_gamma and coeff.ndim == 3 and coeff.shape[0] == 1:
            return coeff[0]
    elif isinstance(coeff, (list, tuple)) and len(coeff) == 1:
        arr = np.asarray(coeff[0])
        if arr.ndim == 2:
            return arr
        if expect_gamma and arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]

    raise ValueError(
        f"Unsupported RHF/RKS mo_coeff layout for {mf_obj.__class__}: "
        f"shape={np.asarray(coeff, dtype=object).shape}"
    )


def _trexio_get_uks_coeff_matrices(mf_obj, *, expect_gamma=False):
    C = mf_obj.mo_coeff
    if isinstance(C, (list, tuple)) and len(C) == 2:
        Ca, Cb = C
    elif isinstance(C, np.ndarray) and C.ndim >= 3 and C.shape[0] == 2:
        if C.ndim == 3:
            Ca, Cb = C[0], C[1]
        elif C.ndim == 4:
            if expect_gamma and C.shape[1] == 1:
                Ca, Cb = C[0, 0], C[1, 0]
            else:
                raise NotImplementedError("Only single-k UKS/UHF is supported.")
        else:
            raise ValueError(f"Unexpected mo_coeff shape: {C.shape}")
    else:
        raise TypeError("Not a UKS/UHF object or unsupported mo_coeff layout.")

    if expect_gamma:
        if Ca.ndim == 3 and Ca.shape[0] == 1:
            Ca = Ca[0]
        if Cb.ndim == 3 and Cb.shape[0] == 1:
            Cb = Cb[0]
        if Ca.ndim != 2 or Cb.ndim != 2:
            raise NotImplementedError(
                "Only Gamma-point data are supported for combined MO coefficients."
            )

    if Ca.ndim != 2 or Cb.ndim != 2:
        raise ValueError(f"Unexpected UKS/UHF mo_coeff shapes: Ca {Ca.shape}, Cb {Cb.shape}")
    return Ca, Cb


def _trexio_concat_spin_coeff(Ca, Cb):
    if Ca.ndim != 2 or Cb.ndim != 2:
        raise ValueError(f"Unexpected UKS/UHF mo_coeff shapes: Ca {Ca.shape}, Cb {Cb.shape}")
    return np.concatenate([Ca, Cb], axis=1)


def _trexio_is_rohf_roks_mf(mf_obj) -> bool:
    return isinstance(
        mf_obj,
        (
            scf.rohf.ROHF,
            dft.roks.ROKS,
            pbcscf.rohf.ROHF,
            pbcscf.krohf.KROHF,
            pbcdft.roks.ROKS,
            pbcdft.kroks.KROKS,
        ),
    ) and not _trexio_is_uhf_uks_mf(mf_obj)


def _trexio_is_uhf_uks_mf(mf_obj) -> bool:
    return isinstance(
        mf_obj,
        (
            scf.uhf.UHF,
            dft.uks.UKS,
            pbcscf.uhf.UHF,
            pbcdft.uks.UKS,
            pbcscf.kuhf.KUHF,
            pbcdft.kuks.KUKS,
        ),
    )


def _trexio_is_rhf_rks_mf(mf_obj) -> bool:
    return (
        isinstance(
            mf_obj,
            (
                scf.hf.RHF,
                dft.rks.RKS,
                pbcscf.rhf.RHF,
                pbcdft.rks.RKS,
                pbcscf.krhf.KRHF,
                pbcdft.krks.KRKS,
            ),
        )
        and not _trexio_is_uhf_uks_mf(mf_obj)
        and not _trexio_is_rohf_roks_mf(mf_obj)
    )


def _write_2e_eri(
    mf, trexio_file, basis='mo', df_engine='MDF', sym='s1',
):
    """Write two-electron repulsion integrals to a TREXIO file.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. PBC data must be
        Gamma-only.
    trexio_file : trexio.File
        An open, writable TREXIO file object.
    basis : {'AO', 'MO'}, optional
        Basis in which integrals are written. AO is always spin-independent.
        MO concatenates alpha and beta MOs for UHF/UKS to include cross-spin
        blocks.
    df_engine : {'MDF', 'GDF'}, optional
        Density-fitting engine used for PBC ERIs (Gamma-only).
    sym : {'s1', 's4', 's8'}, optional
        ERI symmetry/packing. MO supports ``s1``/``s4``; AO supports
        ``s1``/``s4``/``s8``.

        Behavior
        --------
        - Two-electron integrals are treated as real-valued in this interface.
            If non-real values are detected, ``ValueError`` is raised.
    - PBC: only single-k Gamma is supported; non-Gamma raises
      ``NotImplementedError``.
    - MO + UHF/UKS: constructs the combined coefficient matrix ``[Ca | Cb]``
      and writes all spin blocks.
    - Arrays are made C-contiguous before writing; dtype is preserved.

    Raises
    ------
    ValueError
        For invalid ``basis``/``sym`` combinations or unexpected shapes.
    NotImplementedError
        For non-Gamma PBC data or unsupported symmetry in MO.
    """

    is_uhf_like = _trexio_is_uhf_uks_mf(mf)
    is_rhf_like = _trexio_is_rhf_rks_mf(mf) or _trexio_is_rohf_roks_mf(mf)
    if not (is_uhf_like or is_rhf_like):
        raise NotImplementedError(
            f"Conversion function for {mf.__class__} is not implemented in _write_2e_eri."
        )

    basis = basis.upper()
    sym = sym.lower()
    is_pbc = hasattr(mf, 'cell')

    if sym not in ('s1', 's4', 's8'):
        raise ValueError("sym must be 's1', 's4', or 's8'")
    if basis == 'MO' and sym == 's8':
        raise NotImplementedError("MO ERI does not support s8 symmetry; use s1 or s4")

    # ---------- helpers ----------
    def _df_obj():
        """Construct a DF engine for PBC Γ-point."""
        if df_engine.upper() == 'MDF':
            return pbcdf.MDF(mf.cell).build()
        if df_engine.upper() == 'GDF':
            return pbcdf.GDF(mf.cell).build()
        raise ValueError("df_engine must be 'MDF' or 'GDF'.")

    def _require_real_2e(eri):
        eri = np.asarray(eri)
        if np.iscomplexobj(eri):
            if np.any(np.imag(eri) != 0):
                raise ValueError("Two-electron integrals must be real-valued.")
            eri = np.real(eri)
        return eri

    # ---------------------
    # MO-basis ERI writing
    # ---------------------
    if basis == 'MO':
        # Molecular
        if not is_pbc:
            if sym == 's8':
                raise NotImplementedError("MO ERI does not support s8 symmetry")
            mo_compact = sym == 's4'
            if is_uhf_like:
                # UKS/UHF -> concatenate [alpha | beta], include cross-spin terms
                Ca, Cb = _trexio_get_uks_coeff_matrices(mf)
                C = _trexio_concat_spin_coeff(Ca, Cb)  # (nao, nalpha+nbeta)
                if getattr(mf, '_eri', None) is not None:
                    eri_mo = ao2mo.incore.full(mf._eri, C, compact=mo_compact)
                else:
                    eri_mo = ao2mo.kernel(mf.mol, C, compact=mo_compact)
                eri_mo = _require_real_2e(eri_mo)
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), trexio_file, 'MO', sym=sym)
            elif is_rhf_like:  # RHF/RKS
                C = _trexio_get_rhf_coeff_matrix(mf, expect_gamma=False)
                if getattr(mf, '_eri', None) is not None:
                    eri_mo = ao2mo.incore.full(mf._eri, C, compact=mo_compact)
                else:
                    eri_mo = ao2mo.kernel(mf.mol, C, compact=mo_compact)
                eri_mo = _require_real_2e(eri_mo)
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), trexio_file, 'MO', sym=sym)
            else:
                raise NotImplementedError(
                    f"Conversion function for {mf.__class__} is not implemented in _write_2e_eri(MO)."
                )

        # PBC (Gamma only)
        else:
            if not _trexio_is_gamma_single_k(mf):
                raise NotImplementedError("PBC MO-ERI: non-Gamma k-points are not supported (real-only backend).")
            if sym == 's8':
                raise NotImplementedError("PBC MO-ERI does not support s8 symmetry; use s1 or s4")
            dfobj = _df_obj()

            if is_uhf_like:
                # UKS/UHF @ Gamma: combined MO matrix [Ca | Cb]
                Ca, Cb = _trexio_get_uks_coeff_matrices(mf, expect_gamma=True)
                C = _trexio_concat_spin_coeff(Ca, Cb)
                eri_mo = dfobj.get_mo_eri((C, C, C, C))
                eri_mo = _require_real_2e(eri_mo)
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim == 2:
                        eri_mo = ao2mo.restore(1, ao2mo.restore(4, eri_mo, nmo), nmo)
                    elif eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                elif sym == 's4':
                    if eri_mo.ndim == 4:
                        eri_mo = ao2mo.restore(4, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), trexio_file, 'MO', sym=sym)
            elif is_rhf_like:  # RHF/RKS @ Gamma
                C = _trexio_get_rhf_coeff_matrix(mf, expect_gamma=True)
                eri_mo = dfobj.get_mo_eri((C, C, C, C))
                eri_mo = _require_real_2e(eri_mo)
                nmo = C.shape[1]
                if sym == 's1':
                    if eri_mo.ndim == 2:
                        eri_mo = ao2mo.restore(1, ao2mo.restore(4, eri_mo, nmo), nmo)
                    elif eri_mo.ndim < 4:
                        eri_mo = ao2mo.restore(1, eri_mo, nmo)
                elif sym == 's4':
                    if eri_mo.ndim == 4:
                        eri_mo = ao2mo.restore(4, eri_mo, nmo)
                _write_2e_int_eri(np.ascontiguousarray(eri_mo), trexio_file, 'MO', sym=sym)
            else:
                raise NotImplementedError(
                    f"Conversion function for {mf.__class__} is not implemented in _write_2e_eri(PBC-MO)."
                )

    # ---------------------
    # AO-basis ERI writing (spin-independent even for UKS/UHF)
    # ---------------------
    else:  # basis == 'AO'
        if is_pbc:
            # PBC AO: Gamma only via DF (real-only)
            if not _trexio_is_gamma_single_k(mf):
                raise NotImplementedError("PBC AO-ERI: non-Gamma k-points are not supported (real-only backend).")
            dfobj = _df_obj()
            eri2 = pbcdf.df_ao2mo.get_eri(dfobj, compact=False)  # (nao^2, nao^2)
            nao = mf.cell.nao_nr()
            if eri2.shape != (nao * nao, nao * nao):
                raise RuntimeError(f"Unexpected ERI shape {eri2.shape}; expected ({nao*nao}, {nao*nao}) at Gamma.")
            eri_ao = eri2.reshape(nao, nao, nao, nao)
            eri_ao = _require_real_2e(eri_ao)
            if sym == 's4':
                eri_ao = ao2mo.restore(4, eri_ao, nao)
            elif sym == 's8':
                eri_ao = ao2mo.restore(8, eri_ao, nao)
            _write_2e_int_eri(np.ascontiguousarray(eri_ao), trexio_file, 'AO', sym=sym)
        else:
            # Molecular AO
            eri_ao = None
            if sym == 's1':
                eri_ao = getattr(mf, '_eri', None)
                if eri_ao is None:
                    # 'int2e' follows mol.cart automatically (spherical vs Cartesian)
                    eri_ao = mf.mol.intor('int2e', aosym='s1')
                else:
                    # _eri may be stored in s4/s8; expand to full tensor for TREXIO write
                    if eri_ao.ndim < 4:
                        n_ao = mf.mol.nao
                        eri_ao = ao2mo.restore(1, eri_ao, n_ao)
            else:
                eri_ao = mf.mol.intor('int2e', aosym=sym)
            eri_ao = _require_real_2e(eri_ao)
            _write_2e_int_eri(np.ascontiguousarray(eri_ao), trexio_file, 'AO', sym=sym)


def _write_2e_int_eri(eri, trexio_file, basis='MO', sym='s1'):
    basis = basis.upper()
    sym = sym.lower()
    if basis not in ['MO', 'AO']:
        raise ValueError("basis must be 'MO' or 'AO'")
    if sym not in ('s1', 's4', 's8'):
        raise ValueError("sym must be 's1', 's4', or 's8'")

    def _pair_from_tril(n):
        i, j = np.tril_indices(n)
        return i.astype(np.int32), j.astype(np.int32)

    tf = trexio_file

    if sym == 's1':
        if eri.ndim != 4:
            raise ValueError(f'ERI array must be a full 4D tensor (p,q,r,s); got ndim={eri.ndim}')

        num_integrals = eri.size
        n = eri.shape[0]
        idx = lib.cartesian_prod([np.arange(n, dtype=np.int32)] * 4)

        # Physicist notation
        idx=idx.reshape((num_integrals,4))
        for i in range(num_integrals):
            idx[i,1],idx[i,2]=idx[i,2],idx[i,1]
        idx=idx.flatten()

        # write ERI
        if basis == 'AO':
            if not trexio.has_ao_num(tf):
                trexio.write_ao_num(tf, n)
        else:
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, n)
        if basis == 'MO':
            trexio.write_mo_2e_int_eri(tf, 0, num_integrals, idx, eri.ravel())
        else:
            trexio.write_ao_2e_int_eri(tf, 0, num_integrals, idx, eri.ravel())
        return

    if sym == 's4':
        if eri.ndim != 2 or eri.shape[0] != eri.shape[1]:
            raise ValueError("s4 ERI must be a square 2D array (npair, npair)")
        npair = eri.shape[0]
        n = (math.isqrt(8 * npair + 1) - 1) // 2
        if n * (n + 1) // 2 != npair:
            raise ValueError('Invalid s4 ERI size for pair indexing')

        pair_i, pair_j = _pair_from_tril(n)
        total = npair * npair
        chunk = 100000

        if basis == 'AO':
            if not trexio.has_ao_num(tf):
                trexio.write_ao_num(tf, n)
        else:
            if not trexio.has_mo_num(tf):
                trexio.write_mo_num(tf, n)

        offset = 0
        while offset < total:
            end = min(offset + chunk, total)
            t = np.arange(offset, end, dtype=np.int64)
            ij = t // npair
            kl = t % npair

            # Physicist notation
            i = pair_i[ij]
            j = pair_j[ij]
            k = pair_i[kl]
            l = pair_j[kl]
            idx = np.stack([i, k, j, l], axis=1).astype(np.int32).ravel()
            val = eri[ij, kl].ravel()

            if basis == 'MO':
                trexio.write_mo_2e_int_eri(tf, offset, end - offset, idx, val)
            else:
                trexio.write_ao_2e_int_eri(tf, offset, end - offset, idx, val)
            offset = end
        return

    # sym == 's8'
    if eri.ndim != 1:
        raise ValueError('s8 ERI must be a 1D packed array')
    total = eri.size
    npair = (math.isqrt(8 * total + 1) - 1) // 2
    if npair * (npair + 1) // 2 != total:
        raise ValueError('Invalid s8 ERI size for pair indexing')
    n = (math.isqrt(8 * npair + 1) - 1) // 2
    if n * (n + 1) // 2 != npair:
        raise ValueError('Invalid s8 ERI size for AO/MO indexing')

    pair_i, pair_j = _pair_from_tril(n)
    tri_i, tri_j = np.tril_indices(npair)
    chunk = 100000

    if basis == 'AO':
        if not trexio.has_ao_num(tf):
            trexio.write_ao_num(tf, n)
    else:
        if not trexio.has_mo_num(tf):
            trexio.write_mo_num(tf, n)

    offset = 0
    while offset < total:
        end = min(offset + chunk, total)
        ij = tri_i[offset:end]
        kl = tri_j[offset:end]

        # Physicist notation
        i = pair_i[ij]
        j = pair_j[ij]
        k = pair_i[kl]
        l = pair_j[kl]
        idx = np.stack([i, k, j, l], axis=1).astype(np.int32).ravel()
        val = eri[offset:end]

        if basis == 'MO':
            trexio.write_mo_2e_int_eri(tf, offset, end - offset, idx, val)
        else:
            trexio.write_ao_2e_int_eri(tf, offset, end - offset, idx, val)
        offset = end


def _write_1e_eri(
    mf, trexio_file, basis='AO', df_engine='MDF',
):
    """Write one-electron integrals to a TREXIO file.

    Stored quantities are overlap, kinetic, nuclear-electron potential, and
    the core Hamiltonian (plus ECP if present) in either AO or MO basis.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. PBC data must be
        Gamma-only.
    trexio_file : trexio.File
        An open, writable TREXIO file object.
    basis : {'AO', 'MO'}, optional
        Basis in which integrals are written. For MO + UHF/UKS, alpha and beta
        coefficients are concatenated column-wise to form a single block.
    df_engine : {'MDF', 'GDF'}, optional
        Density-fitting engine for PBC integrals (Gamma-only).

        Behavior
        --------
        - Complex one-electron integrals are not supported in this interface.
            If overlap/kinetic/potential/core are complex, ``NotImplementedError``
            is raised.
    - PBC: only single-k Gamma calculations are supported.
    - All matrices are hermitized before writing and made C-contiguous; dtype
      is preserved.

    Raises
    ------
    ValueError
        For invalid ``basis`` or unexpected shapes.
    NotImplementedError
        For complex data, non-Gamma PBC calculations, or unsupported MO layout.
    """

    is_uhf_like = _trexio_is_uhf_uks_mf(mf)
    is_rhf_like = _trexio_is_rhf_rks_mf(mf) or _trexio_is_rohf_roks_mf(mf)
    if not (is_uhf_like or is_rhf_like):
        raise NotImplementedError(
            f"Conversion function for {mf.__class__} is not implemented in _write_1e_eri."
        )

    basis = basis.upper()
    if basis not in ('AO', 'MO'):
        raise ValueError("basis must be either 'AO' or 'MO'")

    def _as_matrix(mat, label):
        if isinstance(mat, (tuple, list)):
            if len(mat) == 0:
                raise ValueError(f"Empty data for {label}")
            if len(mat) > 1:
                raise NotImplementedError(
                    f"{label}: multiple blocks are not supported in this helper"
                )
            mat = mat[0]
        mat = np.asarray(mat)
        if mat.ndim == 3:
            if mat.shape[0] != 1:
                raise NotImplementedError(
                    f"{label}: Gamma-only support; received shape {mat.shape}"
                )
            mat = mat[0]
        if mat.ndim != 2:
            raise ValueError(f"{label} must be a 2D matrix, got shape {mat.shape}")
        return mat

    def _hermitize(mat):
        return 0.5 * (mat + mat.T.conj())

    def _reject_complex_1e(mat, label):
        mat = np.asarray(mat)
        if np.iscomplexobj(mat):
            raise NotImplementedError(
                f"Complex {label} is not supported in trexio.py _write_1e_eri."
            )
        return mat

    is_pbc = hasattr(mf, 'cell')

    ecp_mat = None

    if is_pbc:
        if not _trexio_is_gamma_single_k(mf):
            raise NotImplementedError(
                "PBC one-electron integrals are implemented for Gamma-point only."
            )

        cell = mf.cell
        overlap = _as_matrix(mf.get_ovlp(), 'AO overlap')
        kinetic = _as_matrix(cell.pbc_intor('int1e_kin', 1, 1), 'AO kinetic')

        df_builder = getattr(mf, 'with_df', None)
        if df_builder is None:
            if df_engine.upper() == 'MDF':
                df_builder = pbcdf.MDF(cell).build()
            elif df_engine.upper() == 'GDF':
                df_builder = pbcdf.GDF(cell).build()
            else:
                raise ValueError("df_engine must be 'MDF' or 'GDF'")
        else:
            df_builder = df_builder.build()

        if cell.pseudo:
            potential = _as_matrix(df_builder.get_pp(), 'AO potential')
        else:
            potential = _as_matrix(df_builder.get_nuc(), 'AO potential')

        if len(getattr(cell, '_ecpbas', [])) > 0:
            from pyscf.pbc.gto import ecp
            ecp_mat = _as_matrix(ecp.ecp_int(cell), 'AO ECP potential')
            potential += ecp_mat

        core = kinetic + potential
    else:
        mol = mf.mol
        overlap = _as_matrix(mf.get_ovlp(), 'AO overlap')
        kinetic = _as_matrix(mol.intor('int1e_kin'), 'AO kinetic')
        potential = _as_matrix(mol.intor('int1e_nuc'), 'AO potential')
        if mol._ecp:
            ecp_mat = _as_matrix(mol.intor('ECPscalar'), 'AO ECP potential')
            potential += ecp_mat
        core = kinetic + potential

    overlap = np.ascontiguousarray(_reject_complex_1e(_hermitize(overlap), 'overlap'))
    kinetic = np.ascontiguousarray(_reject_complex_1e(_hermitize(kinetic), 'kinetic'))
    potential = np.ascontiguousarray(_reject_complex_1e(_hermitize(potential), 'potential'))
    core = np.ascontiguousarray(_reject_complex_1e(_hermitize(core), 'core'))
    if ecp_mat is not None:
        ecp_mat = np.ascontiguousarray(_reject_complex_1e(_hermitize(ecp_mat), 'ecp'))

    if basis == 'AO':
        _write_1e_int_eri(overlap, kinetic, potential, core, trexio_file, 'AO')
        if ecp_mat is not None:
            trexio.write_ao_1e_int_ecp(trexio_file, ecp_mat)
        return

    if is_uhf_like:
        Ca, Cb = _trexio_get_uks_coeff_matrices(mf, expect_gamma=is_pbc)
        C = _trexio_concat_spin_coeff(Ca, Cb)
    elif is_rhf_like:
        C = _trexio_get_rhf_coeff_matrix(mf, expect_gamma=is_pbc)
    else:
        raise NotImplementedError(
            f"Conversion function for {mf.__class__} is not implemented in _write_1e_eri(MO)."
        )

    if C.ndim != 2:
        raise ValueError(f"MO coefficient matrix must be 2D, got shape {C.shape}")

    def _ao_to_mo(mat, coeff):
        return _hermitize(coeff.conj().T @ mat @ coeff)

    mo_overlap = np.ascontiguousarray(_reject_complex_1e(_ao_to_mo(overlap, C), 'mo_overlap'))
    mo_kinetic = np.ascontiguousarray(_reject_complex_1e(_ao_to_mo(kinetic, C), 'mo_kinetic'))
    mo_potential = np.ascontiguousarray(_reject_complex_1e(_ao_to_mo(potential, C), 'mo_potential'))
    mo_core = np.ascontiguousarray(_reject_complex_1e(_ao_to_mo(core, C), 'mo_core'))
    mo_ecp = None
    if ecp_mat is not None:
        mo_ecp = np.ascontiguousarray(_reject_complex_1e(_ao_to_mo(ecp_mat, C), 'mo_ecp'))

    _write_1e_int_eri(
        mo_overlap, mo_kinetic, mo_potential, mo_core, trexio_file, 'MO'
    )
    if mo_ecp is not None:
        trexio.write_mo_1e_int_ecp(trexio_file, mo_ecp)


def _write_1e_int_eri(overlap, kinetic, potential, core, trexio_file, basis='AO'):
    basis = basis.upper()
    if basis not in ('AO', 'MO'):
        raise ValueError("basis must be either 'AO' or 'MO'")

    overlap = np.ascontiguousarray(overlap)
    kinetic = np.ascontiguousarray(kinetic)
    potential = np.ascontiguousarray(potential)
    core = np.ascontiguousarray(core)
    tf = trexio_file

    if basis == 'AO':
        ao_dim = overlap.shape[0]
        if not trexio.has_ao_num(tf):
            trexio.write_ao_num(tf, ao_dim)
        trexio.write_ao_1e_int_overlap(tf, overlap)
        trexio.write_ao_1e_int_kinetic(tf, kinetic)
        trexio.write_ao_1e_int_potential_n_e(tf, potential)
        trexio.write_ao_1e_int_core_hamiltonian(tf, core)
    else:
        mo_dim = overlap.shape[0]
        if not trexio.has_mo_num(tf):
            trexio.write_mo_num(tf, mo_dim)
        trexio.write_mo_1e_int_overlap(tf, overlap)
        trexio.write_mo_1e_int_kinetic(tf, kinetic)
        trexio.write_mo_1e_int_potential_n_e(tf, potential)
        trexio.write_mo_1e_int_core_hamiltonian(tf, core)


def _write_1b_rdm(mf, trexio_file):
    """Write a one-body reduced density matrix in MO basis to TREXIO.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. Uses ``mo_occ`` to
        build diagonal MO densities. PBC data must be Gamma-only.
    trexio_file : trexio.File
        An open, writable TREXIO file object.

    Behavior
    --------
    - MO basis is enforced (TREXIO convention).
    - RHF/RKS: writes a spin-summed diagonal density ``diag(mo_occ)``.
    - UHF/UKS: writes spin-blocked ``rdm_1e_up``/``rdm_1e_dn`` and a
      block-diagonal spin-summed matrix. Alpha/beta MO counts must match.
    - PBC: only single-k Gamma is supported; k-resolved occupations are
      squeezed to 1D first.
    - Real-only backend: imaginary parts larger than 1e-12 raise
      ``NotImplementedError``.

    Raises
    ------
    ValueError
        When alpha/beta dimensions differ or input shapes are inconsistent.
    NotImplementedError
        For complex densities or non-Gamma PBC calculations.
    """

    is_pbc = hasattr(mf, 'cell')
    if is_pbc and not _trexio_is_gamma_single_k(mf):
        raise NotImplementedError("RDM write supports Gamma-point only for PBC.")

    is_uhf_like = _trexio_is_uhf_uks_mf(mf)
    is_rhf_like = _trexio_is_rhf_rks_mf(mf) or _trexio_is_rohf_roks_mf(mf)
    if not (is_uhf_like or is_rhf_like):
        raise NotImplementedError(
            f"Conversion function for {mf.__class__} is not implemented in _write_1b_rdm."
        )

    # MO-basis density is diagonal in canonical orbitals
    if is_uhf_like:
        if isinstance(mf.mo_occ, (tuple, list)) and len(mf.mo_occ) == 2:
            occ_a, occ_b = mf.mo_occ
            occ_a = np.asarray(occ_a)
            occ_b = np.asarray(occ_b)
        else:
            occ = np.asarray(mf.mo_occ)
            if occ.ndim >= 2 and occ.shape[0] == 2:
                occ_a, occ_b = occ[0], occ[1]
            else:
                raise ValueError(
                    f"Unsupported UHF/UKS mo_occ layout for {mf.__class__}: shape {occ.shape}"
                )
    else:
        occ = np.asarray(mf.mo_occ)
        occ_a = occ
        occ_b = None

    if is_pbc and isinstance(occ_a, np.ndarray) and occ_a.ndim == 2:
        if occ_a.shape[0] != 1:
            raise NotImplementedError(
                "PBC RDM write: only single-k Gamma supported for MO basis.")
        occ_a = occ_a[0]
        if occ_b is not None and occ_b.ndim == 2:
            occ_b = occ_b[0]

    if occ_a.ndim != 1:
        raise ValueError(f"Expected 1D mo_occ for alpha/spin-summed branch, got shape {occ_a.shape}")
    if occ_b is not None and occ_b.ndim != 1:
        raise ValueError(f"Expected 1D mo_occ for beta branch, got shape {occ_b.shape}")

    if occ_b is not None:
        dm_a = np.diag(occ_a)
        dm_b = np.diag(occ_b)
        dm_a = np.asarray(dm_a)
        if np.iscomplexobj(dm_a):
            dm_a = np.real(dm_a)
        dm_b = np.asarray(dm_b)
        if np.iscomplexobj(dm_b):
            dm_b = np.real(dm_b)
        dm_a = np.ascontiguousarray(dm_a)
        dm_b = np.ascontiguousarray(dm_b)
        tf = trexio_file
        nmo_up = dm_a.shape[0]
        nmo_dn = dm_b.shape[0]
        if nmo_dn != nmo_up:
            raise ValueError("Alpha/beta MO sizes do not match for RDM write.")
        nmo = nmo_up + nmo_dn
        if not trexio.has_mo_num(tf):
            trexio.write_mo_num(tf, nmo)
        trexio.write_rdm_1e_up(tf, dm_a)
        trexio.write_rdm_1e_dn(tf, dm_b)
        dm_tot = np.zeros((nmo, nmo), dtype=dm_a.dtype)
        dm_tot[:nmo_up, :nmo_up] = dm_a
        dm_tot[nmo_up:, nmo_up:] = dm_b
        trexio.write_rdm_1e(tf, dm_tot)
    else:
        dm = np.diag(occ_a)
        dm = np.asarray(dm)
        if np.iscomplexobj(dm):
            dm = np.real(dm)
        dm = np.ascontiguousarray(dm)
        tf = trexio_file
        nmo = dm.shape[0]
        if not trexio.has_mo_num(tf):
            trexio.write_mo_num(tf, nmo)
        trexio.write_rdm_1e(tf, dm)


def _write_2b_rdm(mf, trexio_file, chunk_size=100000):
    """Write a two-body reduced density matrix in MO basis to TREXIO.

    Parameters
    ----------
    mf : SCF/KSCF object
        Converged molecular or PBC mean-field object. Uses ``mo_occ`` to
        build diagonal MO densities. PBC data must be Gamma-only.
    trexio_file : trexio.File
        An open, writable TREXIO file object.
    chunk_size : int, optional
        Number of elements streamed per block when writing flattened ERIs.

    Behavior
    --------
    - MO basis is enforced (TREXIO convention).
    - RHF/RKS: writes spin-summed 2-RDM using ``dm = diag(mo_occ)`` and
      ``G[pqrs] = dm[p,r]*dm[q,s] - 0.5*dm[p,s]*dm[q,r]``.
    - UHF/UKS: writes spin-resolved blocks ``G_uu``, ``G_dd``, and ``G_ud``
      constructed from ``diag(occ_a)`` and ``diag(occ_b)``; alpha/beta sizes
      must match.
    - PBC: only single-k Gamma is supported; k-resolved occupations are
      squeezed to 1D first.
    - Data are streamed in chunks to avoid holding the entire flattened array
      in memory at once; memory still scales as O(n^4).

    Raises
    ------
    ValueError
        When alpha/beta dimensions differ or input shapes are inconsistent.
    NotImplementedError
        For non-Gamma PBC calculations.
    """

    is_pbc = hasattr(mf, 'cell')
    if is_pbc and not _trexio_is_gamma_single_k(mf):
        raise NotImplementedError("RDM write supports Gamma-point only for PBC.")

    is_uhf_like = _trexio_is_uhf_uks_mf(mf)
    is_rhf_like = _trexio_is_rhf_rks_mf(mf) or _trexio_is_rohf_roks_mf(mf)
    if not (is_uhf_like or is_rhf_like):
        raise NotImplementedError(
            f"Conversion function for {mf.__class__} is not implemented in _write_2b_rdm."
        )

    # Spin-summed occupations or spin-separated for UHF/UKS
    if is_uhf_like:
        if isinstance(mf.mo_occ, (tuple, list)) and len(mf.mo_occ) == 2:
            occ_a, occ_b = mf.mo_occ
            occ_a = np.asarray(occ_a)
            occ_b = np.asarray(occ_b)
        else:
            occ = np.asarray(mf.mo_occ)
            if occ.ndim >= 2 and occ.shape[0] == 2:
                occ_a, occ_b = occ[0], occ[1]
            else:
                raise ValueError(
                    f"Unsupported UHF/UKS mo_occ layout for {mf.__class__}: shape {occ.shape}"
                )
    else:
        occ = np.asarray(mf.mo_occ)
        occ_a = occ
        occ_b = None

    if is_pbc and isinstance(occ_a, np.ndarray) and occ_a.ndim == 2:
        if occ_a.shape[0] != 1:
            raise NotImplementedError(
                "PBC RDM write: only single-k Gamma supported for MO basis.")
        occ_a = occ_a[0]
        if occ_b is not None and occ_b.ndim == 2:
            occ_b = occ_b[0]

    if occ_a.ndim != 1:
        raise ValueError(f"Expected 1D mo_occ for alpha/spin-summed branch, got shape {occ_a.shape}")
    if occ_b is not None and occ_b.ndim != 1:
        raise ValueError(f"Expected 1D mo_occ for beta branch, got shape {occ_b.shape}")

    if occ_b is not None:
        dm_a = np.diag(occ_a)
        dm_b = np.diag(occ_b)
        dm_a = np.asarray(dm_a)
        if np.iscomplexobj(dm_a):
            dm_a = np.real(dm_a)
        dm_b = np.asarray(dm_b)
        if np.iscomplexobj(dm_b):
            dm_b = np.real(dm_b)
        dm_a = np.ascontiguousarray(dm_a)
        dm_b = np.ascontiguousarray(dm_b)

        g2_uu = (
            np.einsum('pr,qs->pqrs', dm_a, dm_a)
            - np.einsum('ps,qr->pqrs', dm_a, dm_a)
        )
        g2_dd = (
            np.einsum('pr,qs->pqrs', dm_b, dm_b)
            - np.einsum('ps,qr->pqrs', dm_b, dm_b)
        )
        g2_ud = np.einsum('pr,qs->pqrs', dm_a, dm_b)

        g2_uu = np.ascontiguousarray(g2_uu)
        g2_dd = np.ascontiguousarray(g2_dd)
        g2_ud = np.ascontiguousarray(g2_ud)

        nmo = dm_a.shape[0]
        if dm_b.shape[0] != nmo:
            raise ValueError("Alpha/beta MO sizes do not match for RDM write.")
        idx = lib.cartesian_prod([np.arange(nmo, dtype=np.int32)] * 4)
        flat_uu = g2_uu.reshape(-1)
        flat_dd = g2_dd.reshape(-1)
        flat_ud = g2_ud.reshape(-1)

        tf = trexio_file
        if not trexio.has_mo_num(tf):
            trexio.write_mo_num(tf, nmo)

        total = idx.shape[0]
        offset = 0
        while offset < total:
            end = min(offset + chunk_size, total)
            trexio.write_rdm_2e_upup(
                tf,
                offset,
                end - offset,
                idx[offset:end],
                flat_uu[offset:end],
            )
            trexio.write_rdm_2e_dndn(
                tf,
                offset,
                end - offset,
                idx[offset:end],
                flat_dd[offset:end],
            )
            trexio.write_rdm_2e_updn(
                tf,
                offset,
                end - offset,
                idx[offset:end],
                flat_ud[offset:end],
            )
            offset = end
    else:
        dm = np.diag(occ_a)
        dm = np.asarray(dm)
        if np.iscomplexobj(dm):
            dm = np.real(dm)
        dm = np.ascontiguousarray(dm)

        # Build dense spin-summed 2-RDM in MO basis.
        # For ROHF, separate alpha (occ >=1) and beta (occ == 2) contributions
        # so that the exchange part is treated correctly:
        #   g2[pqrs] = D_tot[pr]*D_tot[qs] - D_a[ps]*D_a[qr] - D_b[ps]*D_b[qr]
        # For RHF (D_a = D_b = D/2) this reduces to D*D - 0.5*D*D.
        if _trexio_is_rohf_roks_mf(mf):
            occ_a_spin = np.minimum(occ_a, 1.0)
            occ_b_spin = occ_a - occ_a_spin
            dm_a = np.ascontiguousarray(np.diag(occ_a_spin))
            dm_b = np.ascontiguousarray(np.diag(occ_b_spin))
            g2 = (
                np.einsum('pr,qs->pqrs', dm, dm)
                - np.einsum('ps,qr->pqrs', dm_a, dm_a)
                - np.einsum('ps,qr->pqrs', dm_b, dm_b)
            )
        else:
            g2 = (
                np.einsum('pr,qs->pqrs', dm, dm)
                - 0.5 * np.einsum('ps,qr->pqrs', dm, dm)
            )
        g2 = np.ascontiguousarray(g2)

        nmo = dm.shape[0]
        idx = lib.cartesian_prod([np.arange(nmo, dtype=np.int32)] * 4)
        flat_g2 = g2.reshape(-1)

        tf = trexio_file
        if not trexio.has_mo_num(tf):
            trexio.write_mo_num(tf, nmo)

        total = idx.shape[0]
        offset = 0
        while offset < total:
            end = min(offset + chunk_size, total)
            trexio.write_rdm_2e(
                tf,
                offset,
                end - offset,
                idx[offset:end],
                flat_g2[offset:end],
            )
            offset = end


def _order_ao_index(mol):
    if mol.cart:
        return np.arange(mol.nao)

    # reorder spherical functions to
    # Pz, Px, Py
    # D 0, D+1, D-1, D+2, D-2
    # F 0, F+1, F-1, F+2, F-2, F+3, F-3
    # G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
    # ...
    lmax = mol._bas[:, gto.ANG_OF].max()
    cache_by_l = [np.array([0]), np.array([2, 0, 1])]
    for l in range(2, lmax + 1):
        idx = np.empty(l * 2 + 1, dtype=int)
        idx[::2] = l - np.arange(0, l + 1)
        idx[1::2] = np.arange(l + 1, l + l + 1)
        cache_by_l.append(idx)

    idx = []
    off = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for n in range(mol.bas_nctr(ib)):
            idx.append(cache_by_l[l] + off)
            off += l * 2 + 1
    return np.hstack(idx)


def _group_by(a, keys):
    """keys must be sorted"""
    assert all(a <= b for a, b in zip(keys, keys[1:]))
    idx = np.unique(keys, return_index=True)[1]
    return np.split(a, idx[1:])


def _get_occsa_and_occsb(mcscf_obj, norb, nelec, ci_threshold=0.0):
    ci_coeff = mcscf_obj.ci
    if isinstance(nelec, (tuple, list, np.ndarray)) and len(nelec) == 2:
        neleca, nelecb = nelec
    else:
        neleca = nelecb = nelec // 2

    occslst_a = fci.cistring.gen_occslst(range(norb), neleca)
    occslst_b = fci.cistring.gen_occslst(range(norb), nelecb)

    occsa = []
    occsb = []
    ci_values = []

    for i in range(min(len(occslst_a), mcscf_obj.ci.shape[0])):
        for j in range(min(len(occslst_b), mcscf_obj.ci.shape[1])):
            ci_coeff = mcscf_obj.ci[i, j]

            if np.abs(ci_coeff) > ci_threshold:  # Check if CI coefficient is significant compared to user defined value
                occsa.append(occslst_a[i])
                occsb.append(occslst_b[j])
                ci_values.append(ci_coeff)

    # Sort by the absolute value of the CI coefficients in descending order
    sorted_indices = np.argsort(-np.abs(ci_values))
    occsa_sorted = [occsa[idx] for idx in sorted_indices]
    occsb_sorted = [occsb[idx] for idx in sorted_indices]
    ci_values_sorted = [ci_values[idx] for idx in sorted_indices]
    num_determinants = len(ci_values_sorted)

    return occsa_sorted, occsb_sorted, ci_values_sorted, num_determinants


def _det_to_trexio(
    mcscf_obj, norb, nelec, trexio_file, ci_threshold=0.0, chunk_size=100000
):
    ncore = _trexio_ncore_scalar(mcscf_obj.ncore)
    int64_num = trexio.get_int64_num(trexio_file)
    orb_num = ncore + norb
    calc_int64_num = (orb_num + 63) // 64
    try:
        int64_num = int(int64_num)
    except (TypeError, ValueError):
        int64_num = 0
    if int64_num <= 0:
        int64_num = calc_int64_num
    else:
        int64_num = max(int64_num, calc_int64_num)

    occsa, occsb, ci_values, num_determinants = _get_occsa_and_occsb(
        mcscf_obj, norb, nelec, ci_threshold
    )

    det_list = []
    for a, b, coeff in zip(occsa, occsb, ci_values):
        occsa_upshifted = [orb for orb in range(ncore)] + [orb + ncore for orb in a]
        occsb_upshifted = [orb for orb in range(ncore)] + [orb + ncore for orb in b]
        occsa_upshifted = [int(orb) for orb in occsa_upshifted]
        occsb_upshifted = [int(orb) for orb in occsb_upshifted]
        det_a = np.asarray(trexio.to_bitfield_list(int64_num, occsa_upshifted), dtype=np.int64)
        det_b = np.asarray(trexio.to_bitfield_list(int64_num, occsb_upshifted), dtype=np.int64)
        det_tmp = np.hstack([det_a, det_b]).astype(np.int64, copy=False)
        det_list.append(det_tmp)

    if num_determinants > chunk_size:
        n_chunks = math.ceil(num_determinants / chunk_size)
    else:
        n_chunks = 1

    if trexio.has_determinant(trexio_file):
        trexio.delete_determinant(trexio_file)

    offset_file = 0
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_determinants)
        current_chunk_size = end - start

        if current_chunk_size > 0:
            trexio.write_determinant_list(
                trexio_file, offset_file, current_chunk_size, det_list[start:end]
            )
            trexio.write_determinant_coefficient(
                trexio_file, offset_file, current_chunk_size, ci_values[start:end]
            )
            offset_file += current_chunk_size


def _to_segment_contraction(mol):
    """transform generally contracted basis to segment contracted basis"""
    _bas = []
    for shell in mol._bas:
        nctr = shell[gto.NCTR_OF]
        if nctr == 1:
            _bas.append(shell)
            continue

        nprim = shell[gto.NPRIM_OF]
        pcoeff = shell[gto.PTR_COEFF]
        bs = np.repeat(shell[np.newaxis], nctr, axis=0)
        bs[:, gto.NCTR_OF] = 1
        bs[:, gto.PTR_COEFF] = np.arange(pcoeff, pcoeff + nprim * nctr, nprim)
        _bas.append(bs)

    pmol = mol.copy()
    pmol._bas = np.asarray(np.vstack(_bas), dtype=np.int32)
    return pmol
