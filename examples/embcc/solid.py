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

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--system", choices=["diamond", "perovskite"], default="diamond")
parser.add_argument("--energy-per", choices=["atom", "cell"])
# For Perovskite only:
parser.add_argument("--ab-atoms", nargs=2, default=["Mg", "Si"])
parser.add_argument("--basis", default="gth-dzv")
parser.add_argument("--solver", default="CCSD(T)")
parser.add_argument("--benchmarks", nargs="*", default=[])
#parser.add_argument("--c-list", nargs="*", default=list(range(1, 21)))
#parser.add_argument("--maxbath", type=int, default=100)
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--pseudopot", default="gth-pade")
parser.add_argument("--minao", default="gth-szv", help="Basis set for IAOs.")
parser.add_argument("--dmet-bath-tol", type=float, default=0.05)
parser.add_argument("--bath-sizes", type=int, nargs="*")
parser.add_argument("--bath-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
#parser.add_argument("--bath-energy-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
parser.add_argument("--bath-energy-tols", type=float, nargs="*")
parser.add_argument("--unit-cell", choices=["primitive", "conventional"], default="primitive")
parser.add_argument("--supercell", type=int, nargs=3)
parser.add_argument("--k-points", type=int, nargs=3)
parser.add_argument("--pyscf-verbose", type=int, default=4)
parser.add_argument("--lattice-consts", type=float, nargs="*", default=[3.45, 3.50, 3.55, 3.60, 3.65, 3.70])
parser.add_argument("--df", choices=["FFTDF", "GDF"], default="FFTDF")
parser.add_argument("--mp2-correction", type=int, choices=[0, 1], default=1)
parser.add_argument("--hf-stability-check", type=int, choices=[0, 1], default=0)
parser.add_argument("--power1-occ-bath-tol", type=float, default=False)
parser.add_argument("--power1-vir-bath-tol", type=float, default=False)
parser.add_argument("--local-occ-bath-tol", type=float, default=False)
parser.add_argument("--local-vir-bath-tol", type=float, default=False)
#parser.add_argument("--hf-stability-check", type=int)
#parser.add_argument("--use-pbc")
#parser.add_argument("--bath-energy-tol", type=float, default=-1)

parser.add_argument("--precision", type=float,
        default=1e-5,
        #default=1e-6,
        #default=1e-8,
        help="Precision for density fitting, determines cell.mesh")

parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.energy_per is None:
    if args.system in ("diamond",):
        args.energy_per = "atom"
    else:
        args.eenrgy_per = "cell"

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%24s: %r", name, value)

def make_diamond(a, supercell=None):
    # Primitive cell
    if args.unit_cell == "primitive":
        amat = a * np.asarray([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5]])
        c_pos = a * np.asarray([[0, 0, 0], [1, 1, 1]])/4
    # Conventional cell
    elif args.unit_cell == "conventional":
        amat = a * np.eye(3)
        c_pos = a * np.asarray([
                (0, 0, 0),
                (1, 1, 1),
                (2, 2, 0),
                (3, 3, 1),
                (2, 0, 2),
                (3, 1, 3),
                (0, 2, 2),
                (1, 3, 3),
                ]) / 4.0
    atom = [("C%d %f %f %f" % (n, c[0], c[1], c[2])) for n, c in enumerate(c_pos)]
    return amat, atom

def make_perovskite(metal_a, metal_b, a):
    amat = a * np.eye(3)
    atom = [
        ("%s %f %f %f" % (metal_a, a/2, a/2, a/2)),
        ("%s %f %f %f" % (metal_b, 0, 0, 0)),
        ("%s %f %f %f" % ("O1", a/2, 0, 0)),
        ("%s %f %f %f" % ("O2", 0, a/2, 0)),
        ("%s %f %f %f" % ("O2", 0, 0, a/2)),
        ]
    return amat, atom

for i, a in enumerate(args.lattice_consts):

    # Build system
    if MPI_rank == 0:
        log.info("Lattice constant= %.3f", a)
        log.info("=======================")
    if args.system == "diamond":
        amat, atom = make_diamond(a)
    elif args.system == "perovskite":
        amat, atom = make_perovskite(*args.ab_atoms, a)
    cell = pyscf.pbc.gto.M(atom=atom, a=amat, basis=args.basis, pseudo=args.pseudopot,
            precision=args.precision, verbose=args.pyscf_verbose)
    if args.supercell:
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)

    # Mean-field
    if args.k_points is None:
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        kpts = cell.make_kpts(args.k_points)
        mf = pyscf.pbc.scf.KRHF(cell, kpts)
    # Density-fitting
    if args.df != "FFTDF":
        if args.df == "GDF":
            mf = mf.density_fit()
        else:
            raise NotImplementedError()
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
    assert mf.converged
    # K-point to supercell gamma point
    if args.k_points is not None:
        t0 = MPI.Wtime()
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
        log.info("Time for k2gamma [s]: %.3f", (MPI.Wtime()-t0))
    else:
        ncells = 1
        mf_sc = mf

    # Impurity labels
    natom = mf_sc.mol.natm
    ncells = np.product(args.k_points)
    eunit = natom if (args.energy_per == "atom") else ncells
    if args.system == "diamond":
        assert natom % 2 == 0
        assert ncells == natom // 2
        implabels = ["C000"]
        symfactor = [natom]
    elif args.system == "perovskite":
        assert natom % 5 == 0
        assert ncells == natom // 5
        implabels = [args.ab_atoms[0] + "000", args.ab_atoms[1] + "000", "O000"]
        symfactor = [ncells, ncells, 3*ncells]
    if isinstance(mf_sc.mol.atom, list):
        #for i in range(len(implabels)):
        for j, implabel in enumerate(implabels):
            if isinstance(mf_sc.mol.atom[j], str):
                _, pos = mf_sc.mol.atom[j].split(" ", 1)
                mf_sc.mol.atom[j] = " ".join([implabel, pos])
            elif isinstance(mf_sc.mol.atom[j], tuple):
                pos = mf_sc.mol.atom[j][1]
                mf_sc.mol.atom[j] = (implabel, pos)
            else:
                raise NotImplementedError()
    else:
        print(mf_sc.mol.atom)
        print(mf_sc.mol.atom[0])
        raise NotImplementedError()
    mf_sc.mol.build(False, False)
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
                minao=args.minao,
                dmet_bath_tol=args.dmet_bath_tol,
                bath_type=args.bath_type,
                solver=args.solver,
                #bath_tol_per_electron=False,
                mp2_correction=args.mp2_correction,
                power1_occ_bath_tol=args.power1_occ_bath_tol, power1_vir_bath_tol=args.power1_vir_bath_tol,
                **kwargs
                )

        solver_opts = {}
        for k, implabel in enumerate(implabels):
            cc.make_atom_cluster(implabel, symmetry_factor=symfactor[k], solver_options=solver_opts,
                    bath_tol_per_electron=(len(implabels) > 1))
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
