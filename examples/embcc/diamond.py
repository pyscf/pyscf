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
parser.add_argument("--bath-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
#parser.add_argument("--bath-energy-tols", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
parser.add_argument("--bath-energy-tols", type=float, nargs="*")
parser.add_argument("--unit-cell", choices=["primitive", "conventional"], default="primitive")
parser.add_argument("--supercell0", type=int, nargs=3)
parser.add_argument("--supercell", type=int, nargs=3)
parser.add_argument("--k-points", type=int, nargs=3)
parser.add_argument("--pyscf-verbose", type=int, default=4)
parser.add_argument("--steps", type=int, nargs=2, default=[2, 4])
parser.add_argument("--df", choices=["FFTDF", "GDF"], default="FFTDF")
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

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

def make_diamond(supercell=None):
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

    if supercell is not None:
        c_pos_sc = []
        for i in range(supercell[0]):
            for j in range(supercell[1]):
                for k in range(supercell[2]):
                    for c in c_pos:
                        c_sc = c + i*amat[0] + j*amat[1] + k*amat[2]
                        c_pos_sc.append(c_sc)
        c_pos = np.asarray(c_pos_sc)

        for d in range(3):
            amat[d] *= supercell[d]

    atom = [("C%d %f %f %f" % (n, c[0], c[1], c[2])) for n, c in enumerate(c_pos)]
    #print(atom)
    #1/0
    return amat, atom


a_eq = 3.5668
a_step = 2.0*a_eq/100.0

n_left = 2
n_right = 4
a_list = a_eq + a_step*np.arange(-args.steps[0], args.steps[1]+1)

for i, a in enumerate(a_list):

    if MPI_rank == 0:
        log.info("Lattice constant= %.3f", a)
        log.info("=======================")

    amat, atom = make_diamond(args.supercell0)
    cell = pyscf.pbc.gto.M(atom=atom, a=amat, basis=args.basis, pseudo=args.pseudopot,
            precision=args.precision, verbose=args.pyscf_verbose)
    if args.supercell:
        cell = pyscf.pbc.tools.super_cell(cell, args.supercell)
        #log.debug("Atoms:")
        #for atom in cell.atom:
        #    log.debug("%s %r" % atom)

    if args.k_points is None:
        mf = pyscf.pbc.scf.RHF(cell)
    else:
        kpts = cell.make_kpts(args.k_points)
        mf = pyscf.pbc.scf.KRHF(cell, kpts)

    if args.df == "FFTDF":
        # FFTDF is default for pbc
        pass
    elif args.df == "GDF":
        mf = mf.density_fit()
    else:
        raise NotImplementedError()

    t0 = MPI.Wtime()
    mf.kernel()
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("Time for HF [s]: %.3f", (MPI.Wtime()-t0))
    assert stable
    assert mf.converged

    if False:
        # Mean field energy per primitive cell
        if args.supercell:
            ncells = np.product(args.supercell)
        else:
            ncells = 1
        e_mf = mf.e_tot / ncells

        e_mp2 = 0.0
        e_ccsd = 0.0
        if "MP2" in args.benchmarks:
            from pyscf.pbc import mp
            if args.k_points:
                mp2 = mp.KMP2(mf)
            else:
                mp2 = mp.MP2(mf)
            t0 = MPI.Wtime()
            mp2.kernel()
            log.info("Time for %s [s]: %.3f", "MP2", (MPI.Wtime()-t0))
            e_mp2 = mp2.e_corr / ncells
        if "CCSD" in args.benchmarks:
            from pyscf.pbc import cc
            if args.k_points:
                ccsd = cc.KCCSD(mf)
            else:
                ccsd = cc.CCSD(mf)
            t0 = MPI.Wtime()
            ccsd.kernel()
            log.info("Time for %s [s]: %.3f", "CCSD", (MPI.Wtime()-t0))
            e_ccsd = ccsd.e_corr / ncells

        # Transform k-point calculation to Gamma point of super cell
        e_mf_sc = 0.0
        e_mp2_sc = 0.0
        e_ccsd_sc = 0.0
        if args.k_points is not None:
            from pyscf.pbc.tools import k2gamma
            mf_sc = k2gamma.k2gamma(mf, args.k_points, tol_orth=1e-4)
            nkpoints = np.product(args.k_points)
            #mf_gamma.e_tot = mf.e_tot * nkpoints
            e_mf_sc = mf.e_tot
            mf_sc.converged = mf.converged
            # Reset the DF
            mf_sc = mf_sc.density_fit()

            if "MP2" in args.benchmarks:
                from pyscf.pbc import mp
                mp2 = mp.MP2(mf_sc)
                t0 = MPI.Wtime()
                mp2.kernel()
                log.info("Time for %s [s]: %.3f", "MP2", (MPI.Wtime()-t0))
                e_mp2_sc = mp2.e_corr / nkpoints
            if "CCSD" in args.benchmarks:
                from pyscf.pbc import cc
                ccsd = cc.CCSD(mf_sc)
                t0 = MPI.Wtime()
                ccsd.kernel()
                log.info("Time for %s [s]: %.3f", "CCSD", (MPI.Wtime()-t0))
                e_ccsd_sc = ccsd.e_corr / nkpoints

        fmtstr = (7*"  %16.12e") + "\n"
        with open(args.output, "a") as f:
            f.write(fmtstr % (a, mf.e_tot, e_mp2, e_ccsd, e_mf_sc, e_mp2_sc, e_ccsd_sc))
        continue

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

    #print(mf_sc.with_df.kpts)
    #print(mf_sc.with_df.__dict__)
    #1/0

    # TEST CCSD(T)
    if 0:
        from pyscf.pbc import cc
        ccsd = cc.CCSD(mf_sc)
        ccsd.kernel()
        et = ccsd.ccsd_t()
        e_ccsd = ccsd.e_corr
        e_ccsdt = ccsd.e_corr + et
        log.info("Exact CCSD = %16.8g , (T) = %16.8g , CCSD(T) = %16.8g", e_ccsd, et, e_ccsdt)

    #from pyscf.cc import ccsd_t_slow
    #et2 = ccsd_t_slow.kernel(ccsd, ccsd.ao2mo())
    #log.info("Exact CCSD = %16.8g , (T) = %16.8g , CCSD(T) = %16.8g", e_ccsd, et2, e_ccsdt)
    #1/0

    implabel = "C000"
    if isinstance(mf_sc.mol.atom, list):
        if isinstance(mf_sc.mol.atom[0], str):
            _, pos = mf_sc.mol.atom[0].split(" ", 1)
            mf_sc.mol.atom[0] = " ".join([implabel, pos])
        elif isinstance(mf_sc.mol.atom[0], tuple):
            pos = mf_sc.mol.atom[0][1]
            mf_sc.mol.atom[0] = (implabel, pos)
        else:
            raise NotImplementedError()
    else:
        print(mf_sc.mol.atom)
        print(mf_sc.mol.atom[0])
        raise NotImplementedError()
    #mf_sc.mol.atom[0] = (implabel, mf_sc.mol.atom[0][1])
    mf_sc.mol.build(False, False)

    #print("ADAD")
    #print(mf.kpt)

    mf_sc._eri = None

    #t0 = MPI.Wtime()
    #eris = mf_sc.with_df.ao2mo(mf_sc.mo_coeff)
    #log.info("Time for full AO->MO [s]: %.3f", (MPI.Wtime()-t0))

    #mp2 = pyscf.pbc.mp.MP2(mf_sc)
    #t0 = MPI.Wtime()
    #mp2.kernel()
    #log.info("Time for full MP2 [s]: %.3f", (MPI.Wtime()-t0))
    #1/0


    enrgs_ccsd = []
    enrgs_ccsdt = []
    enrgs_ccsd_dmp2 = []
    enrgs_ccsdt_dmp2 = []

    natom = mf_sc.mol.natm

    log.info("MF mesh: %r" % mf_sc.mol.mesh)

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
                #bath_type=args.bath_type,
                solver=args.solver,
                **kwargs
                )

        solver_opts = {}
        symfac = natom
        cc.make_atom_cluster(implabel, symmetry_factor=symfac, solver_options=solver_opts)
        cc.run()

        enrgs_ccsd.append(cc.e_tot / natom)
        enrgs_ccsd_dmp2.append((cc.e_tot + cc.e_delta_mp2) / natom)
        enrgs_ccsdt.append((cc.e_tot + cc.e_pert_t) / natom)
        enrgs_ccsdt_dmp2.append((cc.e_tot + cc.e_delta_mp2 + cc.e_pert_t) / natom)
        #enrgs_mp2.append((cc.e_tot + cc.e_delta_mp2)/natom)

    if MPI_rank == 0:
        files = ["ccsd.txt", "ccsdt.txt", "ccsd-dmp2.txt", "ccsdt-dmp2.txt"]
        data = [enrgs_ccsd, enrgs_ccsdt, enrgs_ccsd_dmp2, enrgs_ccsdt_dmp2]
        #out_ccsd = "ccsd.txt"
        #out_ccsdt = "ccsd_t.txt"
        #output2 = args.output.rsplit(".")[0] + "-dmp2." + args.output.rsplit(".")[1]

        if (i == 0):
            title = "#A  HF  " + "  ".join([("bath-%e" % b) for b in bath_params])
            for file in files:
                with open(file, "a") as f:
                    f.write(title + "\n")

        fmtstr = ((len(enrgs_ccsd)+2) * "  %+16.12e") + "\n"
        for idx, file in enumerate(files):
            with open(file, "a") as f:
                f.write(fmtstr % (a, mf_sc.e_tot/natom, *(data[idx])))

    #if MPI_rank == 0:
    #    if (i == 0):
    #        with open(args.output, "a") as f:
    #            f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2\n")
    #    with open(args.output, "a") as f:
    #        f.write(("%.5f  " + 4*"  %16.12e" + "\n") % (a, mf.e_tot, cc.e_corr, cc.e_delta_mp2, cc.e_corr+cc.e_delta_mp2))
