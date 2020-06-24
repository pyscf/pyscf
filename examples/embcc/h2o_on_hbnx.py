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
        #default=[2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
        default=[2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.8, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0, 8.0],
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
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")

parser.add_argument("--impurity-atoms", nargs="*",
        #default=["H*2", "O*1", "H*0", "N#0"],
        default=["H*2", "O*1", "H*0", "N#0", "B#1"],
        help="Atoms to include in the impurity space. N1 for closest nitrogen atom, B2 for next-nearest boron atoms.")

#parser.add_argument("-p", "--max-power", type=int, default=0)
#parser.add_argument("--recalc-bath", action="store_true")
parser.add_argument("--solver", choices=["MP2", "CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmark", choices=["MP2", "CISD", "CCSD", "FCI"])
parser.add_argument("--max-memory", type=int, default=1e5)
parser.add_argument("-o", "--output", default="energies.txt")

parser.add_argument("--bath-type")
parser.add_argument("--local-type", choices=["IAO", "AO"], default="IAO")
#parser.add_argument("--local-type", choices=["IAO", "AO"], default="AO")
parser.add_argument("--bath-tol", type=float, nargs=2)
parser.add_argument("--bath-size", type=int, nargs=2)
parser.add_argument("--bath-relative-size", type=float, nargs=2, default=[None, None])
parser.add_argument("--mp2-correction", action="store_true")
# Load and restore DF integrals
parser.add_argument("--cderi-name", default="cderi-%.2f")
parser.add_argument("--cderi-save", action="store_true")
parser.add_argument("--cderi-load", action="store_true")

args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.cderi_load is None:
    args.cderi_load = args.cderi_save

if args.bath_size is None:
    args.bath_size = args.bath_relative_size
    del args.bath_relative_size

if args.large_basis == args.basis or not args.large_basis_atoms:
    basis = args.basis
else:
    basis = {atom : args.large_basis for atom in args.large_basis_atoms}
    basis["default"] = args.basis
    args.basis = basis
del args.large_basis
del args.large_basis_atoms

if args.minao == "full":
    args.minao = args.basis

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%20s: %r", name, value)

def setup_cell(distance, args):
    """Setup PySCF cell object."""

    a_matrix, atoms = pyscf.molstructures.lattices.hexagonal.make(
            a=args.lattice_const[0], c=args.lattice_const[1], atoms=["B", "N"],
            supercell=args.supercell, distance=distance)

    cell = pyscf.pbc.gto.Cell()
    cell.a = a_matrix
    cell.atom = atoms

    #if args.large_basis == args.basis or not args.large_basis_atoms:
    #    cell.basis = args.basis
    #else:
    #    cell.basis = {atom : args.large_basis for atom in args.large_basis_atoms}
    #    cell.basis["default"] = args.basis
    cell.basis = args.basis

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

ref_orbitals = None
for icalc, distance in enumerate(args.distances):
    if MPI_rank == 0:
        log.info("distance=%.3f", distance)

    cell = setup_cell(distance, args)

    mf = pyscf.pbc.scf.RHF(cell)

    # Density fitting
    mf = mf.density_fit()
    cderi_name = args.cderi_name % distance
    if args.cderi_save:
        log.debug("Saving DF in file %s", cderi_name)
        mf.with_df._cderi_to_save = cderi_name
    elif args.cderi_load:
        log.debug("Loading DF from file %s", cderi_name)
        mf.with_df._cderi = cderi_name

    t0 = MPI.Wtime()
    mf.kernel()
    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)

    if args.benchmark:
        import pyscf.pbc
        if args.benchmark == "MP2":
            import pyscf.pbc.mp
            bm = pyscf.pbc.mp.MP2(mf)
        elif args.benchmark == "CISD":
            import pyscf.pbc.ci
            bm = pyscf.pbc.ci.CISD(mf)
        elif args.benchmark == "CCSD":
            import pyscf.pbc.cc
            bm = pyscf.pbc.cc.CCSD(mf)
        elif args.benchmark == "FCI":
            import pyscf.pbc.fci
            bm = pyscf.pbc.fci.FCI(mol, mf.mo_coeff)
        bm.kernel()
        if hasattr(bm, "converged"):
            assert bm.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, mf.e_tot+bm.e_corr, bm.e_tot))

    else:
        #if args.local_type == "AO":
        #    if icalc == 0:
        #        #cc = embcc.EmbCC(mf, solver=args.solver, tol_bath=args.tol_bath, use_ref_orbitals_bath=not args.recalc_bath)
        #        cc = embcc.EmbCC(mf, solver=args.solver, tol_bath=args.tol_bath, use_ref_orbitals_bath=not args.recalc_bath,
        #                tol_vno=args.tol_vno, vno_ratio=args.vno_ratio)
        #        cc.make_custom_atom_cluster(args.impurity_atoms)
        #        #cc.make_custom_atom_cluster(args.impurity_atoms)
        #        cc.make_rest_cluster(solver=None)
        #    else:
        #        cc.reset(mf=mf)

        #    if icalc == 0 and MPI_rank == 0:
        #        cc.print_clusters()
        #else:
        #cc = embcc.EmbCC(mf, solver=args.solver, bath_type=args.bath_type, bath_target_size=args.bath_target_size)
        cc = embcc.EmbCC(mf, local_orbital_type=args.local_type, minao=args.minao, solver=args.solver, bath_type=args.bath_type, bath_size=args.bath_size, bath_tol=args.bath_tol)
        #cc.make_custom_iao_atom_cluster(args.impurity_atoms, minao=args.minao)
        cc.make_atom_cluster(args.impurity_atoms)

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        if icalc == 0 and MPI_rank == 0:
            cc.print_clusters()

        cc.run()

        if args.local_type == "IAO":
            ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if icalc == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCCSD  EmbCCSD(alt)  EmbCCSD(full)\n")
                    #f.write("#IRC  HF  EmbCCSD  EmbCCSD(v)  EmbCCSD(dMP2)  EmbCCSD(v,dMP2)\n")
                    f.write("#IRC  HF  EmbCCSD  EmbCCSD(dMP2)\n")
            with open(args.output, "a") as f:
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_tot_alt, mf.e_tot+cc.clusters[0].e_corr_full))
                #f.write("%3f  %.8e  %.8e  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_tot_v, cc.e_tot_dmp2, cc.e_tot_v_dmp2))
                f.write("%3f  %.8e  %.8e  %.8e\n" % (distance, mf.e_tot, cc.e_tot, cc.e_tot_dmp2))
