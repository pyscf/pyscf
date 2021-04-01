# Standard
import sys
import argparse
import os.path
# External
import numpy as np
# Internal
import pyscf
import pyscf.pbc
import pyscf.pbc.dft
import pyscf.pbc.tools
import pyscf.pbc.mp

import pyscf.embcc
import pyscf.embcc.k2gamma_gdf

try:
    from mpi4py import MPI
    MPI_comm = MPI.COMM_WORLD
    MPI_rank = MPI_comm.Get_rank()
    MPI_size = MPI_comm.Get_size()
    timer = MPI.Wtime
except (ImportError, ModuleNotFoundError) as e:
    MPI = False
    MPI_rank = 0
    MPI_size = 1
    from timeit import default_timer as timer

log = pyscf.embcc.log

def get_arguments():
    """Get arguments from command line."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # System
    parser.add_argument("--system", choices=["diamond", "graphene", "perovskite"], default="diamond")
    parser.add_argument("--atoms", nargs="*")
    parser.add_argument("--basis", default="gth-dzv")
    parser.add_argument("--pseudopot", default="gth-pade")
    parser.add_argument("--ecp")
    parser.add_argument("--supercell", type=int, nargs=3)
    parser.add_argument("--supercell-in", type=int, nargs=3)
    parser.add_argument("--k-points", type=int, nargs=3)
    parser.add_argument("--lattice-consts", type=float, nargs="*")
    parser.add_argument("--ndim", type=int)
    parser.add_argument("--vacuum-size", type=float)                    # For 2D
    parser.add_argument("--precision", type=float, default=1e-5)
    parser.add_argument("--rcut", type=float)
    parser.add_argument("--pyscf-verbose", type=int, default=10)
    parser.add_argument("--exp-to-discard", type=float, help="If set, discard diffuse basis functions.")
    parser.add_argument("--ke-cutoff", type=float)
    parser.add_argument("--energy-per", choices=["atom", "cell"], help="Express total energy per unit cell or per atom.")

    # Mean-field
    parser.add_argument("--save-scf", help="Save primitive cell SCF.", default="scf-%.2f.chk")           # If containg "%", it will be formatted as args.save_scf % a with a being the lattice constant
    parser.add_argument("--load-scf", help="Load primitive cell SCF.")
    parser.add_argument("--load-scf-init")
    parser.add_argument("--hf-stability-check", type=int, choices=[0, 1], default=0)
    parser.add_argument("--exxdiv-none", action="store_true")

    # MP
    parser.add_argument("--canonical-mp2", action="store_true", help="Perform canonical MP2 calculation.")

    # Density-fitting
    parser.add_argument("--df", choices=["FFTDF", "GDF"], default="GDF", help="Density-fitting method")
    parser.add_argument("--gdf-mesh", type=int, nargs=3)
    parser.add_argument("--gdf-lindep-threshold", type=float)
    parser.add_argument("--gdf-mesh-factor", type=float)
    parser.add_argument("--gdf-exp-to-discard", type=float)
    parser.add_argument("--auxbasis", help="Auxiliary basis. Only works for those known to PySCF.")
    parser.add_argument("--auxbasis-file", help="Load auxiliary basis from file (NWChem format)")
    parser.add_argument("--save-gdf", help="Save primitive cell GDF") #, default="gdf-%.2f.h5")
    parser.add_argument("--load-gdf", help="Load primitive cell GDF")
    parser.add_argument("--load-gdf-unfolded", action="store_true")
    parser.add_argument("--gdf-force-eig", action="store_true")

    # Embedded correlated calculation
    parser.add_argument("--solver", default="CCSD")
    parser.add_argument("--minao", default="gth-szv", help="Minimial basis set for IAOs.")
    parser.add_argument("--opts", nargs="*", default=[])
    # Bath specific
    parser.add_argument("--bath-type", default="mp2-natorb", help="Type of additional bath orbitals.")
    parser.add_argument("--dmet-bath-tol", type=float, default=1e-4, help="Tolerance for DMET bath orbitals. Default=0.05.")
    #parser.add_argument("--bath-sizes", type=int, nargs="*")
    parser.add_argument("--bath-tol", type=float, nargs="*",
            default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
            help="""Tolerance for additional bath orbitals. If positive, interpreted as an occupation number threshold.
            If negative, the bath is extended until it contains 100*(1-|tol|)% of all electrons/holes of the environment
            on the MP2 level.""")
    parser.add_argument("--bath-tol-fragment-weight", type=float, nargs="*", help="Adjust the bath tolerance per fragment.")
    parser.add_argument("--mp2-correction", type=int, choices=[0, 1], default=1, help="Calculate MP2 correction to energy.")
    # Other type of bath orbitals (pre MP2-natorb)
    parser.add_argument("--power1-occ-bath-tol", type=float, default=False)
    parser.add_argument("--power1-vir-bath-tol", type=float, default=False)
    parser.add_argument("--local-occ-bath-tol", type=float, default=False)
    parser.add_argument("--local-vir-bath-tol", type=float, default=False)
    parser.add_argument("--prim-mp2-bath-tol-occ", type=float, default=False)
    parser.add_argument("--prim-mp2-bath-tol-vir", type=float, default=False)
    parser.add_argument("--dft")
    args, restargs = parser.parse_known_args()
    sys.argv[1:] = restargs

    # System specific default arguments
    if args.system == "diamond":
        defaults = {
                "atoms" : ["C", "C"],
                "ndim" : 3,
                #"lattice_consts" : np.arange(3.55, 3.62+1e-12, 0.01),
                "lattice_consts" : np.arange(3.4, 3.8+1e-12, 0.1),
                # For 2x2x2:
                #"lattice_consts" : np.arange(3.61, 3.68+1e-12, 0.01),
                }
    elif args.system == "graphene":
        defaults = {
                "atoms" : ["C", "C"],
                "ndim" : 2,
                "lattice_consts" : np.arange(2.35, 2.6+1e-12, 0.05),
                #"lattice_consts" : np.arange(2.44, 2.56+1e-12, 0.02),
                "vacuum_size" : 20.0
                }
    elif args.system == "perovskite":
        defaults = {
                "atoms" : ["Sr", "Ti", "O"],
                "ndim" : 3,
                #
                #"lattice_consts" : np.arange(3.6, 4.2+1e-12, 0.1),
                "lattice_consts" : np.arange(3.7, 4.2+1e-12, 0.1),
                }

    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    args.lattice_consts = np.asarray(args.lattice_consts)

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
    atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]
    return amat, atom

def make_graphene(a, c, atoms):
    amat = np.asarray([
            [a, 0, 0],
            [a/2, a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    coords_internal = np.asarray([
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 3.0]])/6
    coords = np.dot(coords_internal, amat)

    if args.supercell_in is None:
        atom = [(atoms[0], coords[0]), (atoms[1], coords[1])]
    else:
        atom = []
        ncopy = args.supercell_in
        nr = 0
        for x in range(ncopy[0]):
            for y in range(ncopy[1]):
                for z in range(ncopy[2]):
                    shift = x*amat[0] + y*amat[1] + z*amat[2]
                    atom.append((atoms[0]+str(nr), coords[0]+shift))
                    atom.append((atoms[1]+str(nr), coords[1]+shift))
                    nr += 1

        amat = np.einsum("i,ij->ij", ncopy, amat)

    return amat, atom

def make_perovskite(a, atoms):
    amat = a * np.eye(3)
    coords = np.asarray([
                [0,     0,      0],
                [a/2,   a/2,    a/2],
                [0,     a/2,    a/2],
                [a/2,   0,      a/2],
                [a/2,   a/2,    0]
                ])
    if args.supercell_in is not None:
        atom = []
        ncopy = args.supercell_in
        nr = 0
        for x in range(ncopy[0]):
            for y in range(ncopy[1]):
                for z in range(ncopy[2]):
                    shift = x*amat[0] + y*amat[1] + z*amat[2]
                    atom.append((atoms[0]+str(nr),      coords[0]+shift))
                    atom.append((atoms[1]+str(nr),      coords[1]+shift))
                    atom.append((atoms[2]+str(nr)+"@1", coords[2]+shift))
                    atom.append((atoms[2]+str(nr)+"@2", coords[3]+shift))
                    atom.append((atoms[2]+str(nr)+"@3", coords[4]+shift))
                    nr += 1

        amat = np.einsum("i,ij->ij", ncopy, amat)

    else:
        atom = [
            (atoms[0], coords[0]),
            (atoms[1], coords[1]),
            (atoms[2]+"@1", coords[2]),
            (atoms[2]+"@2", coords[3]),
            (atoms[2]+"@3", coords[4]),
            ]

    return amat, atom

def make_cell(a, args):

    cell = pyscf.pbc.gto.Cell()
    if args.system == "diamond":
        cell.a, cell.atom = make_diamond(a, atoms=args.atoms)
        cell._natom_prim = 2
    if args.system == "graphene":
        cell.a, cell.atom = make_graphene(a, c=args.vacuum_size, atoms=args.atoms)
        cell._natom_prim = 2
    elif args.system == "perovskite":
        cell.a, cell.atom = make_perovskite(a, atoms=args.atoms)
        cell._natom_prim = 5
    cell.dimension = args.ndim
    cell.precision = args.precision

    if args.rcut is not None:
        cell.rcut = args.rcut
    if args.ke_cutoff is not None:
        cell.ke_cutoff = args.ke_cutoff

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
        #cell = pyscf.pbc.tools.super_cell(cell, args.supercell, update_mesh=False)
        #cell = pyscf.pbc.tools.super_cell(cell, args.supercell, update_mesh=True, update_ewald=True)

    return cell

def run_mf(a, cell, args):
    if args.k_points is None or np.product(args.k_points) == 1:
        kpts = cell.make_kpts([1, 1, 1])
        if args.dft:
            mf = pyscf.pbc.dft.RKS(cell)
        else:
            mf = pyscf.pbc.scf.RHF(cell)
    else:
        kpts = cell.make_kpts(args.k_points)
        if args.dft:
            mf = pyscf.pbc.dft.KRKS(cell, kpts)
        else:
            mf = pyscf.pbc.scf.KRHF(cell, kpts)
    if args.exxdiv_none:
        mf.exxdiv = None
    # Load SCF from checkpoint file
    load_scf_ok = False
    if args.load_scf:
        fname = (args.load_scf % a) if ("%" in args.load_scf) else args.load_scf
        log.info("Loading SCF from file %s...", fname)
        try:
            chkfile_dict = pyscf.pbc.scf.chkfile.load(fname, "scf")
            log.info("Loaded attributes: %r", list(chkfile_dict.keys()))
            mf.__dict__.update(chkfile_dict)
            mf.converged = True
        except IOError:
            log.error("Could not load SCF from file %s. File not found." % fname)
            log.error("Calculating SCF instead.")
        except Exception as e:
            log.error("ERROR loading SCF from file %s", fname)
            log.error("Exception: %s", e)
            log.error("Calculating SCF instead.")
        else:
            load_scf_ok = True
            log.info("SCF loaded successfully.")

    # Density-fitting
    if args.df == "GDF":
        mf = mf.density_fit()
        df = mf.with_df
        # TEST

        if args.gdf_exp_to_discard is not None:
            df.exp_to_discard = args.gdf_exp_to_discard
        if args.gdf_lindep_threshold is not None:
            df.linear_dep_threshold = args.gdf_lindep_threshold
        if args.gdf_force_eig:
            df.force_eig = True
        if args.auxbasis is not None:
            log.info("Loading auxbasis %s.", args.auxbasis)
            df.auxbasis = args.auxbasis
        elif args.auxbasis_file is not None:
            log.info("Loading auxbasis from file %s.", args.auxbasis_file)
            df.auxbasis = {atom : pyscf.gto.load(args.auxbasis, atom) for atom in args.atoms}
        load_gdf_ok = False
        # Load GDF
        if args.load_gdf is not None:
            if not args.load_gdf_unfolded:
                fname = (args.load_gdf % a) if ("%" in args.load_gdf) else args.load_gdf
                log.info("Loading GDF from file %s...", fname)
                if os.path.isfile(fname):
                    df._cderi = fname
                    load_gdf_ok = True
                else:
                    log.error("Could not load GDF from file %s. File not found." % fname)
            else:
                log.info("Loading of unfolded GDF deferred.")
                load_gdf_ok = True
        # Calculate GDF
        if not load_gdf_ok:
            if args.gdf_mesh is not None:
                df.mesh = args.gdf_mesh
            elif args.gdf_mesh_factor is not None:
                df.mesh = [int(args.gdf_mesh_factor*n) for n in df.mesh]
            # Force odd mesh
            df.mesh = [2*(n//2)+1 for n in df.mesh]
            if args.save_gdf is not None:
                fname = (args.save_gdf % a) if ("%" in args.save_gdf) else args.save_gdf
                df._cderi_to_save = fname
                log.info("Saving GDF to file %s...", fname)
            log.info("Building GDF...")
            t0 = timer()
            df.build()
            log.info("Time for GDF build: %.3f s", (timer()-t0))

    # Calculate SCF
    if not load_scf_ok:
        if args.save_scf:
            fname = (args.save_scf % a) if ("%" in args.save_scf) else args.save_scf
            mf.chkfile = fname

        if args.load_scf_init:
            fname = (args.load_scf_init % a) if ("%" in args.load_scf_init) else args.load_scf_init
            log.info("Loading initial guess for SCF from file %s...", fname)
            chkfile_dict = pyscf.pbc.scf.chkfile.load(fname, "scf")
            log.info("Loaded attributes: %r", list(chkfile_dict.keys()))
            occ0, c0 = chkfile_dict["mo_occ"], chkfile_dict["mo_coeff"]
            c0 = c0[:,occ0>0]
            dm0 = 2*np.dot(c0 * occ0[occ0>0], c0.T.conj())
        else:
            dm0 = None

        t0 = timer()
        mf.kernel(dm0=dm0)
        log.info("Time for HF: %.3f s", (timer()-t0))
        if args.hf_stability_check:
            t0 = timer()
            mo_stab = mf.stability()[0]
            stable = np.allclose(mo_stab, mf.mo_coeff)
            log.info("Time for HF stability check: %.3f s", (timer()-t0))
            assert stable
    log.info("HF converged: %r", mf.converged)
    log.info("HF energy: %.8e", mf.e_tot)
    assert(mf.converged)

    # Check orthogonality
    if hasattr(mf, "kpts"):
        sk = mf.get_ovlp()
        for ik, k in enumerate(mf.kpts):
            c = mf.mo_coeff[ik]
            csc = np.linalg.multi_dot((c.T.conj(), sk[ik], c))
            err = abs(csc-np.eye(c.shape[-1])).max()
            if err > 1e-9:
                log.error("MOs not orthogonal at k-point %d. Error= %.2e", ik, err)
    else:
        s = mf.get_ovlp()
        c = mf.mo_coeff
        csc = np.linalg.multi_dot((c.T.conj(), s, c))
        err = abs(csc-np.eye(c.shape[-1])).max()
        if err > 1e-9:
            log.error("MOs not orthogonal. Error= %.2e", err)

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
    log.info("Nuclear energy= %.8g", mf.energy_nuc())

    # k-point to supercell gamma point
    if args.k_points is not None and np.product(args.k_points) > 1:
        log.info("k2gamma...")
        t0 = timer()
        if args.load_gdf and args.load_gdf_unfolded:
            mf = pyscf.embcc.k2gamma_gdf.k2gamma_gdf(mf, args.k_points, unfold_j3c=False)
            fname = (args.load_gdf % a) if ("%" in args.load_gdf) else args.load_gdf
            log.info("Loading unfolded GDF from file %s...", fname)
            mf.with_df._cderi = fname
        else:
            mf = pyscf.embcc.k2gamma_gdf.k2gamma_gdf(mf, args.k_points)
        log.info("Time for k2gamma: %.3f s", (timer()-t0))
    else:
        mf._eri = None

    np.savetxt("mo-energies-%.2f.txt" % a, mf.mo_energy)

    natom = mf.mol.natm
    ncells = natom / cell._natom_prim
    eref = natom if (args.energy_per == "atom") else ncells

    energies = { "hf" : [mf.e_tot / eref], "ccsd" : [], "ccsd-dmp2" : [] }

    # Canonical full system MP2
    if args.canonical_mp2:
        try:
            t0 = timer()
            mp2 = pyscf.pbc.mp.MP2(mf)
            mp2.kernel()
            log.info("Ecorr(MP2)= %.8g", mp2.e_corr)
            log.info("Time for canonical MP2: %.3f s", (timer()-t0))
            energies["mp2"] = [mp2.e_tot / eref]
        except Exception as e:
            log.error("Error in canonical MP2 calculation: %s", e)

    if args.solver == "CCSD(T)":
        energies["ccsdt"] = []
        energies["ccsdt-dmp2"] = []

    # Embedding calculations
    # ----------------------
    # Loop over bath tolerances
    for j, btol in enumerate(args.bath_tol):

        kwargs = {opt : True for opt in args.opts}
        if args.prim_mp2_bath_tol_occ:
            kwargs["prim_mp2_bath_tol_occ"] = args.prim_mp2_bath_tol_occ
        if args.prim_mp2_bath_tol_vir:
            kwargs["prim_mp2_bath_tol_vir"] = args.prim_mp2_bath_tol_vir

        ccx = pyscf.embcc.EmbCC(mf, solver=args.solver, minao=args.minao, dmet_bath_tol=args.dmet_bath_tol,
            bath_type=args.bath_type, bath_tol=btol,
            mp2_correction=args.mp2_correction,
            power1_occ_bath_tol=args.power1_occ_bath_tol, power1_vir_bath_tol=args.power1_vir_bath_tol,
            **kwargs
            )

        # Define atomic fragments, first argument is atom index
        if args.system == "diamond":
            ccx.make_atom_cluster(0, symmetry_factor=natom)
        elif args.system == "graphene":
            for ix in range(2):
                ccx.make_atom_cluster(ix, symmetry_factor=ncells, **kwargs)
        elif args.system == "perovskite":
            weights = args.bath_tol_fragment_weight
            # Overwrites ccx.bath_tol per cluster
            if weights is not None:
                for ix in range(5):
                    ccx.make_atom_cluster(ix, symmetry_factor=ncells, bath_tol=weights[ix]*btol)
            else:
                for ix in range(5):
                    ccx.make_atom_cluster(ix, symmetry_factor=ncells)

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
            if val:
                fname = "%s.txt" % key
                log.info("Writing to file %s", fname)
                with open(fname, "a") as f:
                    f.write(("%6.3f" + len(val)*"  %+16.12e" + "\n") % (a, *val))

    log.changeIndentLevel(-1)
