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
        #default=[2, 2, 1],
        #default=[3, 3, 1],
        default=[4, 4, 1],
        help="Supercell size in each direction")
parser.add_argument("--distances", type=float, nargs="*",
        #default=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        #default=[2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        default=[2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        help="Set of substrate-surface distances to calculate")
parser.add_argument("--ke-cutoff", type=float, help="Planewave cutoff")
parser.add_argument("--precision", type=float, default=1e-6,
        help="Precision for density fitting, determines cell.mesh")
#parser.add_argument("--exp-to-discard", type=float, default=0.1,
parser.add_argument("--exp-to-discard", type=float,
        help="Threshold for discarding diffuse Gaussians, helps convergence.")

parser.add_argument("--default-basis", help="Basis set.")
parser.add_argument("--basis", nargs="*", help="Basis set.")
parser.add_argument("--basis-file", nargs=2, help="Basis set file.")

parser.add_argument("--dimension", type=int, default=2)

parser.add_argument("--pseudopot", default="gth-pade", help="Pseudo potential.")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")

parser.add_argument("--impurity-atoms", nargs="*",
        #default=["H*2", "O*1", "H*0", "N#0"],
        #default=["H*2", "O*1", "H*0", "N#0", "B#1"],
        default=["H*2", "O*1", "H*0", "N#0", "B#1", "N#2"],
        help="Atoms to include in the impurity space. N1 for closest nitrogen atom, B2 for next-nearest boron atoms.")

#parser.add_argument("-p", "--max-power", type=int, default=0)
#parser.add_argument("--recalc-bath", action="store_true")
parser.add_argument("--solver", choices=["MP2", "CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmarks", nargs="*", choices=["MP2", "CISD", "CCSD", "FCI"])
parser.add_argument("--max-memory", type=int, default=1e5)
parser.add_argument("-o", "--output", default="energies.txt")

parser.add_argument("--bath-type")
parser.add_argument("--local-type", choices=["IAO", "LAO", "AO"], default="IAO")
#parser.add_argument("--local-type", choices=["IAO", "AO"], default="AO")
parser.add_argument("--bath-tol", type=float, nargs=2)
parser.add_argument("--bath-size", type=int, nargs=2)
parser.add_argument("--bath-relative-size", type=float, nargs=2)
parser.add_argument("--mp2-correction", action="store_true")
#parser.add_argument("--use-)
# Load and restore DF integrals
parser.add_argument("--cderi-name", default="cderi-%.2f")
parser.add_argument("--cderi-load", action="store_true")

parser.add_argument("--df", choices=["gaussian", "mixed"], default="gaussian")
parser.add_argument("--print-ao", action="store_true")
parser.add_argument("--mf-only", action="store_true")
parser.add_argument("--no-embcc", action="store_true")
parser.add_argument("--preconverge_mf", action="store_true")
parser.add_argument("--xc")

# Counterpoise
parser.add_argument("--counterpoise", choices=["none", "water", "water-full", "surface", "surface-full"])

args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

def parse_basis(args):
    if args.default_basis is not None:
        args.basis = {"default" : args.default_basis}
        return

    # Default basis
    basis = {"default" : "gth-dzv"}
    #basis = {"default" : "gth-tzvp"}
    for atom in args.impurity_atoms:
        #if atom in ("H*0", "N#0", "H*2", "O*1"):
        #if atom in ("H*0", "N#0", "H*2", "O*1"):
        if atom in ("H*0", "N#0"):
            #basis[atom] = "gth-aug-dzvp"
            #basis[atom] = "gth-dzvp"
            basis[atom] = "gth-aug-tzvp"
        else:
            #basis[atom] = "gth-dzvp"
            basis[atom] = "gth-tzvp"

    log.debug("Default basis=%r", basis)

    if args.basis:
        basis_updates = {}
        for elem in args.basis:
            atom, bas = elem.split("=")
            basis_updates[atom] = bas
        basis.update(basis_updates)

        log.debug("Final basis=%r", basis)

    return basis


if args.bath_size is None:
    args.bath_size = args.bath_relative_size
    del args.bath_relative_size

if args.counterpoise == "none":
    args.counterpoise = None

if args.basis_file:
    #args.basis = args.basis_file
    args.basis = {
            "default" : args.basis_file[0],
            "H*0" : args.basis_file[1],
            "O*1" : args.basis_file[1],
            "H*2" : args.basis_file[1],
            "N#0" : args.basis_file[1],
            }
else:
    args.basis = parse_basis(args)

if args.minao == "full":
    args.minao = args.basis

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%20s: %r", name, value)

def counterpoise_mod(atoms, basis):
    water = ["H*2", "O*1", "H*0"]
    surface = [a[0] for a in atoms if (a[0] not in water)]
    if args.counterpoise == "water":
        atoms_new, basis_new = mod_for_counterpoise(atoms, basis, fragment=water, remove_basis=False)
    elif args.counterpoise == "surface":
        atoms_new, basis_new = mod_for_counterpoise(atoms, basis, fragment=surface, remove_basis=False)
    return atoms_new, basis_new

    #log.debug("Atoms before assigning ghosts:\n%r", atoms)
    #log.debug("Basis before assigning ghosts:\n%r", basis)
    #atoms_new = []
    #basis_new = {}
    ## Remove surface
    #if args.counterpoise == "water":
    #    atoms_new = [atom for atom in atoms if atom[0][0] in ("H", "O")]
    #    #basis_new = args.basis
    #    basis_new = {key : basis[key] for key in basis if key[0] in ("H", "O")}
    #    basis_new["default"] = basis["default"]
    ## Remove water
    #elif args.counterpoise == "surface":
    #    atoms_new = [atom for atom in atoms if atom[0][0] in ("B", "N")]
    #    basis_new = {key : basis[key] for key in basis if key[0] in ("B", "N")}
    #    basis_new["default"] = basis["default"]
    #    #basis_new = args.basis
    ## Remove surface but keep surface basis
    #elif args.counterpoise == "water-full":
    #    for atom in atoms:
    #        if atom[0][0] in ("B", "N"):
    #            atomlabel = "GHOST-" + atom[0]
    #        else:
    #            atomlabel = atom[0]
    #        atoms_new.append((atomlabel, atom[1]))
    #    for key, val in args.basis.items():
    #        if key[0] in ("B", "N"):
    #            key = "GHOST-" + key
    #        basis_new[key] = val
    ## Remove water but keep water basis
    #elif args.counterpoise == "surface-full":
    #    for atom in atoms:
    #        if atom[0][0] in ("H", "O"):
    #            atomlabel = "GHOST-" + atom[0]
    #        else:
    #            atomlabel = atom[0]
    #        atoms_new.append((atomlabel, atom[1]))
    #    for key, val in basis.items():
    #        if key[0] in ("H", "O"):
    #            key = "GHOST-" + key
    #        basis_new[key] = val

    #log.debug("Atoms after assigning ghosts:\n%r", atoms_new)
    #log.debug("Basis after assigning ghosts:\n%r", basis_new)
    #return atoms_new, basis_new


def setup_cell(distance, args, **kwargs):
    """Setup PySCF cell object."""

    a_matrix, atoms = pyscf.molstructures.lattices.hexagonal.make(
            a=args.lattice_const[0], c=args.lattice_const[1], atoms=["B", "N"],
            supercell=args.supercell, distance=distance)

    cell = pyscf.pbc.gto.Cell()
    cell.a = a_matrix

    if args.counterpoise:
        #atoms, basis = counterpoise_mod(atoms, args.basis)
        atoms, _ = counterpoise_mod(atoms, None)
        basis = args.basis
    else:
        basis = args.basis

    cell.atom = atoms
    cell.basis = basis

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

ref_orbitals = None
refdata = None
for icalc, distance in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.3f", distance)

    cell = setup_cell(distance, args)

    if args.print_ao:
        for ao in cell.ao_labels():
            log.info("%s", ao)
        break

    if args.xc in (None, "HF"):
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        import pyscf.pbc.dft
        mf = pyscf.pbc.dft.RKS(cell)
        mf.xc = args.xc

    # Density fitting
    if args.df == "gaussian":
        mf = mf.density_fit()
    elif args.df == "mixed":
        mf = mf.mix_density_fit()
    # Even tempered Gaussian as auxiliary basis
    if False:
        mf.with_df.auxbasis = pyscf.pbc.df.aug_etb(cell)

    cderi_name = args.cderi_name % distance
    if args.cderi_load:
        log.debug("Loading DF from file %s", cderi_name)
        mf.with_df._cderi = cderi_name
    else:
        log.debug("Saving DF in file %s", cderi_name)
        mf.with_df._cderi_to_save = cderi_name

    # Start from HF in reduced basis
    if args.preconverge_mf:
        cell0 = setup_cell(distance, args, exp_to_discard=0.1)
        mask = np.isin(cell.ao_labels(), cell0.ao_labels())
        if not np.all(mask):
            mf0 = pyscf.pbc.scf.RHF(cell0)
            mf0 = mf0.density_fit()
            t0 = MPI.Wtime()
            mf0.kernel()
            log.info("Time for mean-field in reduced basis: %.2g", MPI.Wtime()-t0)
            assert mf0.converged

            dm0 = np.zeros((cell.nao_nr(), cell.nao_nr()))
            dm0[np.ix_(mask, mask)] = mf0.make_rdm1()

            # Fix the number of electrons
            ne = np.sum(dm0 * mf.get_ovlp())
            log.debug("Number of electrons in initial guess DM0=%.8g", ne)
            dm0 *= cell.nelectron / ne
            ne = np.sum(dm0 * mf.get_ovlp())
            log.debug("Number of electrons in initial guess DM0 (fixed)=%.8g", ne)
            assert np.isclose(ne, cell.nelectron)

        else:
            dm0 = None
    else:
        dm0 = None

    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)

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
