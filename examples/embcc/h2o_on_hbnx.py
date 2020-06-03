import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
#import pyscf.gto
import pyscf.pbc
import pyscf.pbc.tools
from pyscf import molstructures
import pyscf.molstructures.lattices
from pyscf import embcc

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
        default=[3, 3, 1],
        #default=[4, 4, 1],
        help="Supercell size in each direction")
parser.add_argument("--distances", type=float, nargs="*",
        default=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        help="Set of substrate-surface distances to calculate")
parser.add_argument("--ke-cutoff", type=float, help="Planewave cutoff")
parser.add_argument("--precision", type=float, default=1e-6,
        help="Precision for density fitting, determines cell.mesh")
parser.add_argument("--exp-to-discard", type=float, default=0.1,
        help="Threshold for discarding diffuse Gaussians, helps convergence.")
parser.add_argument("--basis", default="gth-dzv", help="Basis set")
parser.add_argument("--large-basis-atoms", default=["H*2", "O*1", "H*0", "N#0", "B#1"],
        help="High accuracy basis set for chemical active atoms")
parser.add_argument("--large-basis", default="gth-dzvp",
        help="High accuracy basis set for chemical active atoms")
parser.add_argument("--pseudopot", default="gth-pade", help="Pseudo potential.")

parser.add_argument("--impurity-atoms", nargs="*",
        default=["H*2", "O*1", "H*0", "N#0"],
        #default=["H*2", "O*1", "H*0", "N#0", "B#1"],
        help="Atoms to include in the impurity space. N1 for closest nitrogen atom, B2 for next-nearest boron atoms.")

parser.add_argument("-p", "--max-power", type=int, default=0)
parser.add_argument("--recalc-bath", action="store_true")
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmark", choices=["CISD", "CCSD", "FCI"])
parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--name", default="output")
parser.add_argument("--max-memory", type=int, default=1e5)
parser.add_argument("--output")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.output is None:
    args.output = args.name + ".out"

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

def setup_cell(distance, args):
    """Setup PySCF cell object."""

    a_matrix, atoms = pyscf.molstructures.lattices.hexagonal.make(
            a=args.lattice_const[0], c=args.lattice_const[1], atoms=["B", "N"],
            supercell=args.supercell, distance=distance)

    cell = pyscf.pbc.gto.Cell()
    cell.a = a_matrix
    cell.atom = atoms

    if args.large_basis == args.basis or not args.large_basis_atoms:
        cell.basis = args.basis
    else:
        cell.basis = {atom : args.large_basis for atom in args.large_basis_atoms}
        cell.basis["default"] = args.basis

    if args.pseudopot:
        cell.pseudo = args.pseudopot
    cell.verbose = 4
    cell.dimension = 2
    cell.precision = args.precision
    if args.ke_cutoff is not None:
        cell.ke_cutoff = args.ke_cutoff
    if args.exp_to_discard is not None:
        cell.exp_to_discard = args.exp_to_discard

    if args.max_memory is not None:
        cell.max_memory = args.max_memory

    cell.build()

    return cell

for icalc, distance in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.3f", distance)

    cell = setup_cell(distance, args)

    mf = pyscf.pbc.scf.RHF(cell)
    mf = mf.density_fit()

    madelung = pyscf.pbc.tools.madelung(cell, [(0,0,0)])
    e_madelung = -0.5 * madelung * cell.nelectron

    mf.kernel()

    if args.benchmark:
        import pyscf.pbc
        if args.benchmark == "CISD":
            import pyscf.pbc.ci
            bm = pyscf.pbc.ci.CISD(mf)
        elif args.benchmark == "CCSD":
            import pyscf.pbc.cc
            bm = pyscf.pbc.cc.CCSD(mf)
        elif args.benchmark == "FCI":
            import pyscf.pbc.fci
            bm = pyscf.pbc.fci.FCI(mol, mf.mo_coeff)
        bm.kernel()
        assert bm.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (distance, mf.e_tot, bm.e_tot))

    else:
        if icalc == 0:
            cc = embcc.EmbCC(mf, solver=args.solver, tol_bath=args.tol_bath, use_ref_orbitals_bath=not args.recalc_bath)
            cc.make_custom_atom_cluster(args.impurity_atoms)
            #cc.make_custom_atom_cluster(args.impurity_atoms)
            cc.make_rest_cluster(solver=None)
        else:
            cc.reset(mf=mf)

        if icalc == 0 and MPI_rank == 0:
            cc.print_clusters()

        conv = cc.run(max_power=args.max_power)
        if MPI_rank == 0:
            assert conv

        if MPI_rank == 0:
            if icalc == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(alt)  EmbCCSD(full)\n")
            with open(args.output, "a") as f:
                f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_tot_alt, mf.e_tot+cc.clusters[0].e_corr_full))
