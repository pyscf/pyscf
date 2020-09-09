import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
#import pyscf.gto
import pyscf.pbc
import pyscf.pbc.df
import pyscf.pbc.tools
from pyscf import molstructures
import pyscf.molstructures.lattices
from pyscf import embcc

from pyscf.molstructures import mod_for_counterpoise

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
# CELL
parser.add_argument("--lattice-const", type=float, nargs=2, default=[2.51, 30.0],
        help="Lattice constants a (plane) and c (non-periodic dimension) in Angstrom")
parser.add_argument("--supercell", type=int, nargs=3,
        default=[2, 2, 1],
        #default=[3, 3, 1],
        #default=[4, 4, 1],
        help="Supercell size in each direction")
parser.add_argument("--distances", type=float, nargs="*",
        #default=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        #default=[2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        default=[2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        help="Set of substrate-surface distances to calculate")
parser.add_argument("--ke-cutoff", type=float, help="Planewave cutoff")
parser.add_argument("--precision", type=float, default=1e-6,
        help="Precision for density fitting, determines cell.mesh")

parser.add_argument("--exp-to-discard", type=float,
        help="Threshold for discarding diffuse Gaussians, helps convergence.")

parser.add_argument("--basis", nargs="*", help="Basis sets: 1) for H2O-N, 2) rest of impurity, 3) Rest of surface",
        default=["gth-aug-tzvp", "gth-tzvp", "gth-dzv"])

parser.add_argument("--dimension", type=int, default=2)

parser.add_argument("--pseudopot", default="gth-pade", help="Pseudo potential.")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")

parser.add_argument("--impurity-atoms", nargs="*",
        #default=["H*2", "O*1", "H*0", "N#0"],
        #default=["H*2", "O*1", "H*0", "N#0", "B#1"],
        default=["H*2", "O*1", "H*0", "N#0", "B#1", "N#2"],
        help="Atoms to include in the impurity space. N1 for closest nitrogen atom, B2 for next-nearest boron atoms.")

parser.add_argument("--solver", choices=["MP2", "CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmarks", nargs="*", choices=["MP2", "CISD", "CCSD", "FCI"])
parser.add_argument("--max-memory", type=int, default=1e5)
parser.add_argument("-o", "--output", default="energies.txt")

parser.add_argument("--bath-type")
parser.add_argument("--local-type", choices=["IAO", "LAO", "AO"], default="IAO")
parser.add_argument("--bath-tol", type=float, nargs=2)
parser.add_argument("--bath-size", type=int, nargs=2)
parser.add_argument("--bath-relative-size", type=float, nargs=2)
parser.add_argument("--mp2-correction", action="store_true")
# Load and restore DF integrals
parser.add_argument("--cderi-name", default="cderi-%.2f")
parser.add_argument("--cderi-load", action="store_true")
parser.add_argument("--cderi-save", action="store_true")

parser.add_argument("--df", choices=["gaussian", "mixed"], default="gaussian")
parser.add_argument("--mf-only", action="store_true")
parser.add_argument("--no-embcc", action="store_true")
parser.add_argument("--preconverge-mf", action="store_true")
parser.add_argument("--xc")

parser.add_argument("--exxdiv", default="ewald")

# Counterpoise
parser.add_argument("--fragment", choices=["all", "water", "surface"], default="all")

args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

def make_basis(basis):

    if len(basis) == 1:
        basis = [basis[0], basis[0], basis[0]]
    elif len(basis) == 2:
        basis = [basis[0], basis[0], basis[1]]
    elif len(basis) != 3:
        raise ValueError()

    basis_dict = {"default" : basis[2]}
    for atom in args.impurity_atoms:
        #if atom in ("H*0", "N#0"):
        if atom in ("H*0", "H*2", "O*1", "N#0"):
            basis_dict[atom] = basis[0]
        else:
            basis_dict[atom] = basis[1]
    return basis_dict

args.basis = make_basis(args.basis)

if args.bath_size is None:
    args.bath_size = args.bath_relative_size
    del args.bath_relative_size

if args.minao == "full":
    args.minao = args.basis

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%20s: %r", name, value)

def setup_cell(distance, args, **kwargs):
    """Setup PySCF cell object."""

    a_matrix, atoms = pyscf.molstructures.lattices.hexagonal.make(
            a=args.lattice_const[0], c=args.lattice_const[1], atoms=["B", "N"],
            supercell=args.supercell, distance=distance)

    cell = pyscf.pbc.gto.Cell()
    cell.a = a_matrix

    cell.atom = atoms
    cell.basis = args.basis

    if args.pseudopot:
        cell.pseudo = args.pseudopot
    cell.verbose = 4

    cell.dimension = args.dimension

    cell.precision = args.precision
    if args.ke_cutoff is not None:
        cell.ke_cutoff = args.ke_cutoff
    if args.exp_to_discard is not None:
        cell.exp_to_discard = args.exp_to_discard

    if args.max_memory is not None:
        cell.max_memory = args.max_memory

    for key, val in kwargs.items():
        log.debug("Setting cell.%s to %r", key, val)
        setattr(cell, key, val)

    cell.build()

    return cell

dm0 = None
ref_orbitals = None
refdata = None
for icalc, distance in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.3f", distance)

    cell = setup_cell(distance, args)

    # Counterpoise
    if args.fragment != "all":
        water, surface = cell.make_counterpoise_fragments([["H*0", "O*1", "H*2"]])
        if args.fragment == "water":
            cell = water
        elif args.fragment == "surface":
            cell = surface

    if args.xc in (None, "HF"):
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        import pyscf.pbc.dft
        mf = pyscf.pbc.dft.RKS(cell)
        mf.xc = args.xc

    # Exxdiv
    if args.exxdiv != "ewald":
        if args.exxdiv == "none":
            args.exxdiv = None
        mf.exxdiv = args.exxdiv

    # Density fitting
    if args.df == "gaussian":
        mf = mf.density_fit()
    elif args.df == "mixed":
        mf = mf.mix_density_fit()
    # Even tempered Gaussian as auxiliary basis [should be default anyway?]
    if False:
        mf.with_df.auxbasis = pyscf.pbc.df.aug_etb(cell)

    cderi_name = args.cderi_name % distance
    if args.cderi_load:
        log.debug("Loading DF integrals from file %s", cderi_name)
        mf.with_df._cderi = cderi_name
    elif args.cderi_save:
        log.debug("Saving DF integrals in file %s", cderi_name)
        mf.with_df._cderi_to_save = cderi_name

    # Start from HF in reduced basis
    if dm0 is None and args.preconverge_mf:
        cell0 = cell.copy()
        cell0.exp_to_discard = 0.2
        cell0._built = False
        cell0.build(True, False)
        mask = np.isin(cell.ao_labels(), cell0.ao_labels())
        if not np.all(mask):
            mf0 = pyscf.pbc.scf.RHF(cell0)
            # Exxdiv
            if args.exxdiv != "ewald":
                mf0.exxdiv = args.exxdiv
            mf0 = mf0.density_fit()
            t0 = MPI.Wtime()
            mf0.kernel()
            log.info("Time for mean-field in reduced basis: %.2g", MPI.Wtime()-t0)
            assert mf0.converged

            dm0 = np.zeros((cell.nao_nr(), cell.nao_nr()))
            dm0[np.ix_(mask, mask)] = mf0.make_rdm1()
            # Fix the number of electrons of density guess
            ne = np.sum(dm0 * mf.get_ovlp())
            dm0 *= cell.nelectron / ne
            ne = np.sum(dm0 * mf.get_ovlp())
            assert np.isclose(ne, cell.nelectron)

    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)
    dm0 = mf.make_rdm1()

    with open(args.output + ".mf", "a") as f:
        f.write("%.3f  %.8e\n" % (distance, mf.e_tot))

    if args.mf_only:
        continue

    assert mf.converged

    if args.benchmarks:
        energies = []
        import pyscf.pbc
        for bm in args.benchmarks:
            t0 = MPI.Wtime()
            if bm == "MP2":
                import pyscf.pbc.mp
                mp2 = pyscf.pbc.mp.MP2(mf)
                mp2.kernel()
                energies.append(mf.e_tot + mp2.e_corr)
            elif bm == "CISD":
                import pyscf.pbc.ci
                ci = pyscf.pbc.ci.CISD(mf)
                ci.kernel()
                assert ci.converged
                energies.append(mf.e_tot + ci.e_corr)
            elif bm == "CCSD":
                import pyscf.pbc.cc
                cc = pyscf.pbc.cc.CCSD(mf)
                cc.kernel()
                assert cc.converged
                energies.append(mf.e_tot + cc.e_corr)
            elif bm == "FCI":
                import pyscf.pbc.fci
                fci = pyscf.pbc.fci.FCI(mol, mf.mo_coeff)
                fci.kernel()
                assert fci.converged
                energies.append(mf.e_tot + fci.e_corr)
            log.info("Time for %s: %.2g", bm, MPI.Wtime()-t0)

        if icalc == 0:
            with open(args.output, "w") as f:
                f.write("#distance  HF  " + "  ".join(args.benchmarks) + "\n")
        with open(args.output, "a") as f:
            f.write(("%.3f  %.8e" + (len(args.benchmarks)*"  %.8e") + "\n") % (distance, mf.e_tot, *energies))

    elif not args.no_embcc:
        cc = embcc.EmbCC(mf, local_type=args.local_type, minao=args.minao, solver=args.solver,
                bath_type=args.bath_type, bath_size=args.bath_size, bath_tol=args.bath_tol,
                mp2_correction=args.mp2_correction)

        cc.make_atom_cluster(args.impurity_atoms)

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        cc.set_refdata(refdata)
        cc.print_clusters()

        if icalc == 0 and MPI_rank == 0:
            cc.print_clusters_orbitals()

        cc.run()

        refdata = cc.get_refdata()

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if icalc == 0:
                with open(args.output, "a") as f:
                    f.write("#distance  HF  EmbCC  dMP2  EmbCCSD+dMP2  FullCC\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
