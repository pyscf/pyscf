# Standard
import sys
import argparse
import os.path
# External
import numpy as np
from mpi4py import MPI
# Internal
import pyscf
import pyscf.pbc
import pyscf.pbc.tools

import pyscf.embcc
import pyscf.embcc.k2gamma

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = pyscf.embcc.log

def get_arguments():
    """Get arguments from command line."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # System
    parser.add_argument("--system", choices=["diamond", "hBN", "perovskite"], default="diamond")
    parser.add_argument("--atoms", nargs="*")
    parser.add_argument("--basis", default="gth-dzv")
    parser.add_argument("--pseudopot")
    parser.add_argument("--ecp")
    parser.add_argument("--supercell", type=int, nargs=3)
    parser.add_argument("--k-points", type=int, nargs=3)
    parser.add_argument("--lattice-consts", type=float, nargs="*")
    parser.add_argument("--ndim", type=int)
    parser.add_argument("--vacuum-size", type=float)                    # For 2D
    parser.add_argument("--precision", type=float, default=1e-5)
    parser.add_argument("--pyscf-verbose", type=int, default=4)
    parser.add_argument("--exp-to-discard", type=float, help="If set, discard diffuse basis functions.")
    parser.add_argument("--energy-per", choices=["atom", "cell"], help="Express total energy per unit cell or per atom.")

    # Mean-field
    parser.add_argument("--save-scf", default="scf-%.2f.chk")           # If containg "%", it will be formatted as args.save_scf % a with a being the lattice constant
    parser.add_argument("--load-scf")
    parser.add_argument("--hf-stability-check", type=int, choices=[0, 1], default=0)

    # Density-fitting
    parser.add_argument("--df", choices=["FFTDF", "GDF"], default="FFTDF", help="Density-fitting method")
    parser.add_argument("--auxbasis", help="Auxiliary basis. Only works for those known to PySCF.")
    parser.add_argument("--auxbasis-file", help="Load auxiliary basis from file (NWChem format)")
    #parser.add_argument("--save-gdf", default="gdf-%.2f.npy")
    parser.add_argument("--save-gdf", default="gdf-%.2f.h5")
    parser.add_argument("--load-gdf")

    # Embedded correlated calculation
    parser.add_argument("--solver", default="CCSD")
    parser.add_argument("--minao", default="gth-szv", help="Minimial basis set for IAOs.")
    # Bath specific
    parser.add_argument("--bath-type", default="mp2-natorb", help="Type of additional bath orbitals.")
    parser.add_argument("--dmet-bath-tol", type=float, default=0.05, help="Tolerance for DMET bath orbitals. Default=0.05.")
    #parser.add_argument("--bath-sizes", type=int, nargs="*")
    parser.add_argument("--bath-tol", type=float, nargs="*",
            default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            help="""Tolerance for additional bath orbitals. If positive, interpreted as an occupation number threshold.
            If negative, the bath is extended until it contains 100*(1-|tol|)% of all electrons/holes of the environment
            on the MP2 level.""")
    parser.add_argument("--bath-multiplier-per-fragment", type=float, nargs="*", help="Adjust the bath tolerance per fragment.")
    parser.add_argument("--mp2-correction", type=int, choices=[0, 1], default=1, help="Calculate MP2 correction to energy.")
    # Other type of bath orbitals (pre MP2-natorb)
    parser.add_argument("--power1-occ-bath-tol", type=float, default=False)
    parser.add_argument("--power1-vir-bath-tol", type=float, default=False)
    parser.add_argument("--local-occ-bath-tol", type=float, default=False)
    parser.add_argument("--local-vir-bath-tol", type=float, default=False)
    args, restargs = parser.parse_known_args()
    sys.argv[1:] = restargs

    # System specific default arguments
    if args.system == "diamond":
        defaults = {
                "atoms" : ["C"],
                "ndim" : 3,
                "pseudopot" : "gth-pade",
                "lattice_consts" : np.arange(3.55, 3.62+1e-12, 0.01),
                # For 2x2x2:
                #"lattice_consts" : np.arange(3.61, 3.68+1e-12, 0.01),
                }
    elif args.system == "hBN":
        defaults = {
                "atoms" : ["B", "N"],
                "ndim" : 2,
                "pseudopot" : "gth-pade",
                "lattice_consts" : np.arange(2.44, 2.56+1e-12, 0.02),
                "vacuum_size" : 20.0
                }
    elif args.system == "perovskite":
        defaults = {
                "atoms" : ["Sr", "Ti", "O"],
                "ndim" : 3,
                "lattice_consts" : np.arange(3.6, 4.2+1e-12, 0.1),
                }
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    if args.energy_per is None:
        if args.system in ("diamond",):
            args.energy_per = "atom"
        else:
            args.energy_per = "cell"

    if MPI_rank == 0:
        log.info("PARAMETERS")
        log.info("**********")
        for name, value in sorted(vars(args).items()):
            log.info("  * %-32s: %r", name, value)

    return args

def make_diamond(a, atoms):
    amat = a * np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    coords = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    atom = [(atoms[0], coords[0]), (atoms[0], coords[1])]
    return amat, atom

def make_hBN(a, c, atoms):
    amat = np.asarray([
            [a, 0, 0],
            [a/2, a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 3.0]])/6
    coords = np.dot(coords_internal, amat)
    atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]
    return amat, atom

def make_perovskite(a, atoms):
    amat = a * np.eye(3)
    atom = [
        ("%s %f %f %f" % (atoms[0], 0,      0,      0)),
        ("%s %f %f %f" % (atoms[1], a/2,    a/2,    a/2)),
        ("%s %f %f %f" % (atoms[2], 0,      a/2,    a/2)),
        ("%s %f %f %f" % (atoms[2], a/2,    0,      a/2)),
        ("%s %f %f %f" % (atoms[2], a/2,    a/2,    0)),
        ]
    return amat, atom

def make_cell(a, args):
    cell = pyscf.pbc.gto.Cell()
    if args.system == "diamond":
        cell.a, cell.atom = make_diamond(a, atoms=args.atoms)
    if args.system == "hBN":
        cell.a, cell.atom = make_hBN(a, c=args.vacuum_size, atoms=args.atoms)
    elif args.system == "perovskite":
        cell.a, cell.atom = make_perovskite(a, atoms=args.atoms)
    cell.dimension = args.ndim
    cell.precision = args.precision
    cell.verbose = args.pyscf_verbose
    cell.basis = args.basis
    if args.pseudopot:
        cell.pseudo = args.pseudopot
    if args.ecp:
        cell.ecp = args.ecp
    if args.exp_to_discard:
        cell.exp_to_discard = args.exp_to_discard
    cell.build()
    if args.supercell and not np.all(args.supercell == 1):
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)
    return cell

def run_mf(a, cell, args):
    if args.k_points is None or np.product(args.k_points) == 1:
        kpts = cell.make_kpts([1, 1, 1])
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        kpts = cell.make_kpts(args.k_points)
        mf = pyscf.pbc.scf.KRHF(cell, kpts)
    # Load SCF from checkpoint file
    load_scf_ok = False
    if args.load_scf:
        fname = (args.load_scf % a) if ("%" in args.load_scf) else args.load_scf
        log.info("Loading SCF from file %s...", fname)
        try:
            mf.__dict__.update(pyscf.pbc.scf.chkfile.load(fname, "scf"))
            load_scf_ok = True
            log.info("SCF loaded successfully.")
        except IOError:
            log.error("IO ERROR loading SCF from file %s", fname)
            log.error("Calculating SCF instead.")
        except Exception as e:
            log.error("Exception: %s", e)
            log.error("ERROR loading SCF from file %s", fname)
            log.error("Calculating SCF instead.")

    # Density-fitting
    if args.df == "GDF":
        mf = mf.density_fit()
        if args.auxbasis is not None:
            log.info("Loading auxbasis %s.", args.auxbasis)
            mf.with_df.auxbasis = args.auxbasis
        elif args.auxbasis_file is not None:
            log.info("Loading auxbasis from file %s.", args.auxbasis_file)
            mf.with_df.auxbasis = {atom : pyscf.gto.load(args.auxbasis, atom) for atom in args.atoms}
        load_gdf_ok = False
        # Load GDF
        if args.load_gdf is not None:
            fname = (args.load_gdf % a) if ("%" in args.load_gdf) else args.load_gdf
            if os.path.isfile(fname):
                log.info("Loading GDF from file %s...", fname)
                mf.with_df._cderi = fname
                load_gdf_ok = True
            else:
                log.error("Could not load GDF from file %s. File not found." % fname)
        # Calculate GDF
        if not load_gdf_ok:
            if args.save_gdf is not None:
                fname = (args.save_gdf % a) if ("%" in args.save_gdf) else args.save_gdf
                mf.with_df._cderi_to_save = fname
                log.info("Saving GDF to file %s...", fname)
            log.info("Building GDF...")
            t0 = MPI.Wtime()
            mf.with_df.build()
            log.info("Time for GDF build: %.3f s", (MPI.Wtime()-t0))

    # Calculate SCF
    if not load_scf_ok:
        if args.save_scf:
            fname = (args.save_scf % a) if ("%" in args.save_scf) else args.save_scf
            mf.chkfile = fname
        t0 = MPI.Wtime()
        mf.kernel()
        log.info("Time for HF: %.3f s", (MPI.Wtime()-t0))
        if args.hf_stability_check:
            t0 = MPI.Wtime()
            mo_stab = mf.stability()[0]
            stable = np.allclose(mo_stab, mf.mo_coeff)
            log.info("Time for HF stability check: %.3f s", (MPI.Wtime()-t0))
            assert stable
    log.info("HF converged: %r", mf.converged)
    log.info("HF energy: %.8e", mf.e_tot)

    return mf

args = get_arguments()

# Loop over geometries
for i, a in enumerate(args.lattice_consts):

    if MPI_rank == 0:
        log.info("LATTICE CONSTANT %.2f", a)
        log.info("*********************")
        log.changeIndentLevel(1)

    # Setup cell
    cell = make_cell(a, args)

    # Mean-field
    mf = run_mf(a, cell, args)

    # k-point to supercell gamma point
    if args.k_points is not None and np.product(args.k_points) > 1:
        t0 = MPI.Wtime()
        mf = pyscf.embcc.k2gamma.k2gamma(mf, args.k_points)
        log.info("Time for k2gamma: %.3f s", (MPI.Wtime()-t0))
        ncells = np.product(args.k_points)
    else:
        mf._eri = None
        ncells = np.product(args.supercell) if args.supercell else 1

    natom = mf.mol.natm
    eref = natom if (args.energy_per == "atom") else ncells

    energies = { "hf" : [mf.e_tot / eref], "ccsd" : [], "ccsd-dmp2" : [] }
    if args.solver == "CCSD(T)":
        energies["ccsdt"] = []
        energies["ccsdt-dmp2"] = []

    # Embedding calculations
    # ----------------------
    # Loop over bath tolerances
    for j, btol in enumerate(args.bath_tol):

        ccx = pyscf.embcc.EmbCC(mf, solver=args.solver, minao=args.minao, dmet_bath_tol=args.dmet_bath_tol,
            bath_type=args.bath_type, bath_tol=btol,
            mp2_correction=args.mp2_correction,
            power1_occ_bath_tol=args.power1_occ_bath_tol, power1_vir_bath_tol=args.power1_vir_bath_tol,
            )

        # Define atomic fragments, first argument is atom label or index
        if args.system == "diamond":
            ccx.make_atom_cluster(0, symmetry_factor=natom)
        elif args.system == "hBN":
            ccx.make_atom_cluster(0, symmetry_factor=ncells, **kwargs)
            ccx.make_atom_cluster(1, symmetry_factor=ncells, **kwargs)
        elif args.system == "perovskite":
            # Overwrites ccx.bath_tol per cluster
            if args.bath_multiplier_per_fragment is not None:
                ccx.make_atom_cluster(0, symmetry_factor=ncells, bath_tol=args.bath_multiplier_per_fragment[0]*btol)
                ccx.make_atom_cluster(1, symmetry_factor=ncells, bath_tol=args.bath_multiplier_per_fragment[1]*btol)
                ccx.make_atom_cluster(2, symmetry_factor=3*ncells, bath_tol=args.bath_multiplier_per_fragment[2]*btol)
            else:
                ccx.make_atom_cluster(0, symmetry_factor=ncells)
                ccx.make_atom_cluster(1, symmetry_factor=ncells)
                ccx.make_atom_cluster(2, symmetry_factor=3*ncells)

        ccx.kernel()

        # Save energies
        energies["ccsd"].append(ccx.e_tot / eref)
        energies["ccsd-dmp2"].append((ccx.e_tot + ccx.e_delta_mp2) / eref)
        if args.solver == "CCSD(T)":
            energies["ccsdt"].append((ccx.e_tot + ccx.e_pert_t) / eref)
            energies["ccsdt-dmp2"].append((ccx.e_tot + ccx.e_delta_mp2 + ccx.e_pert_t) / eref)

    # Write energies to files
    if MPI_rank == 0:
        for key, val in energies.items():
            with open("%s.txt" % key, "a") as f:
                f.write(("%6.3f" + len(val)*"  %+16.12e" + "\n") % (a, *val))

    log.changeIndentLevel(-1)
