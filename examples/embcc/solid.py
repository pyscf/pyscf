import sys
import logging
import argparse
import itertools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.pbc
import pyscf.pbc.tools
from pyscf import embcc

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

#log = logging.getLogger(__name__)
log = embcc.log

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--system", choices=["diamond", "hBN", "perovskite"], default="diamond")
parser.add_argument("--energy-per", choices=["atom", "cell"])
# For Perovskite only:
parser.add_argument("--atoms", nargs=2, default=["Sr", "Ti"])
parser.add_argument("--ndim", type=int)

parser.add_argument("--basis", default="gth-dzv")
parser.add_argument("--solver", default="CCSD")
parser.add_argument("--benchmarks", nargs="*", default=[])
#parser.add_argument("--c-list", nargs="*", default=list(range(1, 21)))
#parser.add_argument("--maxbath", type=int, default=100)
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
#parser.add_argument("--pseudopot", default="gth-pade")
parser.add_argument("--pseudopot")
parser.add_argument("--auxbasis")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")
parser.add_argument("--ecp")
parser.add_argument("--dmet-bath-tol", type=float, default=0.05)
parser.add_argument("--bath-sizes", type=int, nargs="*")
parser.add_argument("--bath-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
#parser.add_argument("--bath-energy-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
parser.add_argument("--bath-energy-tols", type=float, nargs="*")
parser.add_argument("--supercell", type=int, nargs=3)
parser.add_argument("--k-points", type=int, nargs=3)
parser.add_argument("--pyscf-verbose", type=int, default=4)
parser.add_argument("--scf-max-cycle", type=int)
#parser.add_argument("--lattice-consts", type=float, nargs="*", default=[3.45, 3.50, 3.55, 3.60, 3.65, 3.70])
parser.add_argument("--lattice-consts", type=float, nargs="*")
parser.add_argument("--vacuum-size", type=float)
parser.add_argument("--df", choices=["FFTDF", "GDF"], default="FFTDF")
parser.add_argument("--mp2-correction", type=int, choices=[0, 1], default=1)
parser.add_argument("--hf-stability-check", type=int, choices=[0, 1], default=0)
parser.add_argument("--power1-occ-bath-tol", type=float, default=False)
parser.add_argument("--power1-vir-bath-tol", type=float, default=False)
parser.add_argument("--local-occ-bath-tol", type=float, default=False)
parser.add_argument("--local-vir-bath-tol", type=float, default=False)
parser.add_argument("--save-scf", default="scf-%.2f.chk")
parser.add_argument("--load-scf")
#parser.add_argument("--hf-stability-check", type=int)
#parser.add_argument("--use-pbc")
#parser.add_argument("--bath-energy-tol", type=float, default=-1)

parser.add_argument("--precision", type=float,
        default=1e-5,
        #default=1e-6,
        #default=1e-8,
        help="Precision for density fitting, determines cell.mesh")
parser.add_argument("--exp-to-discard", type=float)

parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

# System specific
if args.system == "diamond":
    defaults = {
            "ndim" : 3,
            "pseudopot" : "gth-pade",
            "lattice_consts" : [3.55, 3.56, 3.57, 3.58, 3.59, 3.60, 3.61, 3.62],
            # For 2x2x2:
            #"lattice_consts" : [3.61, 3.62, 3.63, 3.64, 3.65, 3.66, 3.67, 3.68],
            }
elif args.system == "hBN":
    defaults = {
            "ndim" : 2,
            "pseudopot" : "gth-pade",
            #"lattice_consts" : [2.46, 2.48, 2.50, 2.52, 2.54, 2.56, 2.58, 2.60],
            "lattice_consts" : 2.5 + 0.02*np.arange(-3, 3+1),
            "vacuum_size" : 20.0
            }
elif args.system == "perovskite":
    defaults = {
            "ndim" : 3,
            "lattice_consts" : [3.7, 3.8, 3.9, 4.0, 4.1]
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
        log.info("  * %-24s: %r", name, value)

def make_diamond(a):
    amat = a * np.asarray([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]])
    c_pos = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    atom = [("C %f %f %f" % (c[0], c[1], c[2])) for c in c_pos]
    return amat, atom

def make_hBN(a, c=args.vacuum_size):
    amat = np.asarray([
            [a, 0, 0],
            [a/2, a*np.sqrt(3.0)/2, 0],
            [0, 0, c]])
    internal = np.asarray([
        [2.0, 2.0, 3.0],
        [4.0, 4.0, 3.0]])/6
    external = np.dot(internal, amat)
    atom = [("B", external[0]), ("N", external[1])]
    return amat, atom

def make_perovskite(metal_a, metal_b, a):
    amat = a * np.eye(3)
    atom = [
        ("%s %f %f %f" % (metal_a, 0, 0, 0)),
        ("%s %f %f %f" % (metal_b, a/2, a/2, a/2)),
        ("%s %f %f %f" % ("O", 0, a/2, a/2)),
        ("%s %f %f %f" % ("O", a/2, 0, a/2)),
        ("%s %f %f %f" % ("O", a/2, a/2, 0)),
        ]
    return amat, atom

for i, a in enumerate(args.lattice_consts):

    # Build system
    if MPI_rank == 0:
        log.info("LATTICE CONSTANT %.2f", a)
        log.info("*********************")
    if args.system == "diamond":
        amat, atom = make_diamond(a)
    if args.system == "hBN":
        amat, atom = make_hBN(a)
    elif args.system == "perovskite":
        amat, atom = make_perovskite(*args.atoms, a)

    #cell = pyscf.pbc.gto.M(atom=atom, a=amat, basis=args.basis, pseudo=args.pseudopot,
    #        dimension=args.ndim,
    #        precision=args.precision, verbose=args.pyscf_verbose)
    cell = pyscf.pbc.gto.Cell()
    cell.atom = atom
    cell.a = amat
    cell.dimension = args.ndim
    cell.precision = args.precision
    cell.verbose=args.pyscf_verbose

    if args.system == "perovskite":
        cell.basis = args.basis
        minao = args.minao
        if args.ecp:
            cell.ecp = args.ecp
        if args.pseudopot:
            cell.pseudo = args.pseudopot

    else:
        cell.basis = args.basis
        cell.pseudo = args.pseudopot
        minao = args.minao

    if args.exp_to_discard:
        cell.exp_to_discard = args.exp_to_discard
    cell.build()
    if args.supercell and not np.all(args.supercell == 1):
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)

    # Mean-field
    if args.k_points is None:
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        kpts = cell.make_kpts(args.k_points)
        mf = pyscf.pbc.scf.KRHF(cell, kpts)

    # Load SCF
    load_scf_success = False
    if args.load_scf:
        fname = args.load_scf % a
        log.info("Loading SCF from file %s", fname)
        try:
            mf.__dict__.update(pyscf.pbc.scf.chkfile.load(fname, "scf"))
            log.info("SCF successfully loaded.")
            load_scf_success = True
        except IOError as err:
            log.error("IO ERROR loading SCF from file %s", fname)
            log.error("Calculating SCF instead.")
        except Exception as e:
            log.error("Exception: %s", e)
            log.error("ERROR loading SCF from file %s", fname)
            log.error("Calculating SCF instead.")

    # Calculate SCF
    if not load_scf_success:
        # Density-fitting
        if args.df != "FFTDF":
            if args.df == "GDF":
                mf = mf.density_fit()
                if args.auxbasis is not None:
                    log.info("Loading auxbasis %s", args.auxbasis)
                    #mf.with_df.auxbasis = args.auxbasis
                    mf.with_df.auxbasis = {
                            "Sr" : pyscf.gto.load(args.auxbasis, "Sr"),
                            "Ti" : pyscf.gto.load(args.auxbasis, "Ti"),
                            "O" : pyscf.gto.load(args.auxbasis, "O"),
                            }
                t0 = MPI.Wtime()
                mf.with_df.build()
                log.info("Time for GDF [s]: %.3f", (MPI.Wtime()-t0))
            else:
                raise NotImplementedError()
        if args.scf_max_cycle is not None:
            mf.max_cycle = args.scf_max_cycle
        if args.save_scf:
            mf.chkfile = args.save_scf % a
        # Mean-field calculation
        t0 = MPI.Wtime()
        mf.kernel()
        log.info("Time for HF [s]: %.3f", (MPI.Wtime()-t0))
        if args.hf_stability_check:
            t0 = MPI.Wtime()
            mo_stab = mf.stability()[0]
            stable = np.allclose(mo_stab, mf.mo_coeff)
            log.info("Time for HF stability check [s]: %.3f", (MPI.Wtime()-t0))
            assert stable
    log.info("HF energy: %.8e", mf.e_tot)
    log.info("HF converged: %r", mf.converged)

    # K-point to supercell gamma point
    if args.k_points is not None:
        t0 = MPI.Wtime()
        #if True:
        if False:
            from pyscf.pbc.tools import k2gamma
            mf_sc = k2gamma.k2gamma(mf, args.k_points, tol_orth=1e-4)
            ncells = np.product(args.k_points)
            assert ncells == (mf_sc.mol.natm // mf.mol.natm)
            # Scale total energy to supercell size
            mf_sc.e_tot = mf.e_tot * ncells
            mf_sc.converged = mf.converged
            # Reset GDF
            if args.df == "GDF":
                 mf_sc = mf_sc.density_fit()
                 mf_sc.with_df._cderi_to_save = "cderi-%.2f.hdf5" % a
        else:
            #from pyscf.pbc.tools import k2gamma as pyscf_k2gamma
            from pyscf.embcc import k2gamma
            #test = pyscf_k2gamma.get_phase(mf.cell, args.k_points)
            mf_sc = k2gamma.k2gamma(mf, args.k_points)
            ncells = np.product(args.k_points)
        log.info("Time for k2gamma [s]: %.3f", (MPI.Wtime()-t0))
    else:
        ncells = np.product(args.supercell) if args.supercell else 1
        mf_sc = mf

    # Impurity labels
    natom = mf_sc.mol.natm
    log.info("Ncells = %d, Natom = %d", ncells, natom)
    eunit = natom if (args.energy_per == "atom") else ncells
    if args.system == "diamond":
        assert natom % 2 == 0
        assert ncells == natom // 2
        impurities = [0]
        symfactor = [natom]
    elif args.system == "hBN":
        assert natom % 2 == 0
        assert ncells == natom // 2
        impurities = [0, 1]
        symfactor = [ncells, ncells]
    elif args.system == "perovskite":
        assert natom % 5 == 0
        assert ncells == natom // 5
        impurities = [0, 1, 2]
        symfactor = [ncells, ncells, 3*ncells]

    mf_sc._eri = None

    enrgs_ccsd = []
    enrgs_ccsdt = []
    enrgs_ccsd_dmp2 = []
    enrgs_ccsdt_dmp2 = []

    bath_params = (args.bath_sizes or (args.bath_tols or args.bath_energy_tols))
    for j, bath in enumerate(bath_params):

        if args.bath_sizes is not None:
            kwargs = {"bath_size" : bath}
        elif args.bath_tols is not None:
            kwargs = {"bath_tol" : bath}
        elif args.bath_energy_tols is not None:
            kwargs = {"bath_energy_tol" : bath}

        cc = embcc.EmbCC(mf_sc,
                #local_type=args.local_type,
                minao=minao,
                dmet_bath_tol=args.dmet_bath_tol,
                bath_type=args.bath_type,
                solver=args.solver,
                bath_tol_per_electron=False,
                mp2_correction=args.mp2_correction,
                power1_occ_bath_tol=args.power1_occ_bath_tol, power1_vir_bath_tol=args.power1_vir_bath_tol,
                **kwargs
                )

        solver_opts = {}
        #for k, implabel in enumerate(implabels):
        #    cc.make_atom_cluster(implabel, symmetry_factor=symfactor[k], solver_options=solver_opts)
        #    #, bath_tol_per_electron=(len(implabels) > 1))

        for k, imp in enumerate(impurities):
            cc.make_atom_cluster(imp, symmetry_factor=symfactor[k], solver_options=solver_opts)
            #, bath_tol_per_electron=(len(implabels) > 1))
        cc.run()

        enrgs_ccsd.append(cc.e_tot / eunit)
        enrgs_ccsd_dmp2.append((cc.e_tot + cc.e_delta_mp2) / eunit)
        enrgs_ccsdt.append((cc.e_tot + cc.e_pert_t) / eunit)
        enrgs_ccsdt_dmp2.append((cc.e_tot + cc.e_delta_mp2 + cc.e_pert_t) / eunit)

        # No need to calculate smaller tolerances - all orbitals were active before
        if cc.clusters[0].nfrozen == 0:
            enrgs_ccsd += (len(bath_params)-j-1)*[enrgs_ccsd[-1]]
            enrgs_ccsd_dmp2 += (len(bath_params)-j-1)*[enrgs_ccsd_dmp2[-1]]
            enrgs_ccsdt += (len(bath_params)-j-1)*[enrgs_ccsdt[-1]]
            enrgs_ccsdt_dmp2 += (len(bath_params)-j-1)*[enrgs_ccsdt_dmp2[-1]]
            break

    if MPI_rank == 0:
        files = ["ccsd.txt", "ccsdt.txt", "ccsd-dmp2.txt", "ccsdt-dmp2.txt"]
        data = [enrgs_ccsd, enrgs_ccsdt, enrgs_ccsd_dmp2, enrgs_ccsdt_dmp2]

        if (i == 0):
            title = "#A  HF  " + "  ".join([("bath-%e" % b) for b in bath_params])
            for file in files:
                with open(file, "a") as f:
                    f.write(title + "\n")

        fmtstr = ((len(enrgs_ccsd)+2) * "  %+16.12e") + "\n"
        for idx, file in enumerate(files):
            with open(file, "a") as f:
                f.write(fmtstr % (a, mf_sc.e_tot/eunit, *(data[idx])))
