import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
from pyscf import molstructures
from pyscf import embcc

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="cc-pVDZ")
#parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--solver", default="CCSD")
parser.add_argument("--benchmarks", nargs="*")
#parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--distances", type=float, nargs="*")
parser.add_argument("--distances-range", type=float, nargs=3, default=[0.4, 5.0, 0.1])
parser.add_argument("--local-type", choices=["IAO", "AO", "LAO", "NonOrth-IAO"], default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-size", type=float, nargs=2)
parser.add_argument("--impurity", nargs="*", default=["H1"])
parser.add_argument("--atoms", nargs=2, default=["H", "H"])
#parser.add_argument("--mp2-correction", action="store_true")
#parser.add_argument("--use-ref-orbitals-bath", type=int, default=0)
parser.add_argument("--minao", default="minao")
parser.add_argument("--no-embcc", action="store_true")
parser.add_argument("--electron-target", type=float)

#parser.add_argument("--counterpoise", choices=["none", "water", "water-full", "borazine", "borazine-full"])
parser.add_argument("--fragment", choices=["all", "water", "boronene"], default="all")
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

#args.use_ref_orbitals_bath = bool(args.use_ref_orbitals_bath)

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])
del args.distances_range

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

structure_builder = functools.partial(molstructures.build_dimer, atoms=args.atoms, add_labels=True)
#structure_builder = functools.partial(molstructures.build_dimer, atoms=args.atoms, add_labels=False)
basis = args.basis

dm0 = None
refdata = None

for idist, dist in enumerate(args.distances):

    if MPI_rank == 0:
        log.info("Distance=%.2f", dist)
        log.info("=============")

    mol = structure_builder(dist, basis=basis, verbose=4)
    #mol = structure_builder(dist, basis=basis, verbose=4, symmetry=True)

    #if args.fragment != "all":
    #    water, boronene = mol.make_counterpoise_fragments([["O*", "H*"]])
    #    if args.fragment == "water":
    #        mol = water
    #    else:
    #        mol = boronene

    mf = pyscf.scf.RHF(mol)
    #mf = mf.density_fit()
    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    mo_stab = mf.stability()[0]
    stable = np.allclose(mo_stab, mf.mo_coeff)
    log.info("HF stable? %r", stable)

    #if True:
    if False:
        while not stable:
            dm0 = mf.make_rdm1(mo_stab, mf.mo_occ)
            mf.kernel(dm0)
            mo_stab = mf.stability()[0]
            stable = np.allclose(mo_stab, mf.mo_coeff)
            log.info("HF stable? %r", stable)

    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)
    assert mf.converged
    dm0 = mf.make_rdm1()

    if args.benchmarks:

        #cas_size = (6, 6)
        cas_size = (8, 8)
        #cas_size = (10, 10)
        #cas_space = {'A1g' : 1 , 'A1u' : 1, 'E1gx' : 1, "E1gy" : 1, 'E1ux' : 1, "E1uy" : 1}
        #core_space = {'A1g' : 2 , 'A1u' : 2}

        run_benchmarks(mf, args.benchmarks, dist, "benchmarks.txt", print_header=(idist==0),
                #cas_size=cas_size, cas_space=cas_space, core_space=core_space)
                cas_size=cas_size)

    #if not args.no_embcc:
    else:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                minao=args.minao,
                bath_type=args.bath_type, bath_size=args.bath_size,
                solver=args.solver,
                #mp2_correction=args.mp2_correction,
                #use_ref_orbitals_bath=args.use_ref_orbitals_bath,
                )
        #cc.make_atom_cluster(args.impurity, symmetry_factor=2)

        solver_opts = {}
        #solver_opts = {"fix_spin" : 0}

        if True:
            #impurity = ["N1 2p"]
            #cc.make_custom_cluster(impurity, symmetry_factor=2.0, solver_options=solver_opts)
            #impurity = ["N2 2p"]
            #cc.make_custom_cluster(impurity, symmetry_factor=1.0, solver_options=solver_opts)

            impurity = ["Li1 2s"]
            cc.make_custom_cluster(impurity, solver_options=solver_opts)
            impurity = ["F2 2s", "F2 2p"]
            cc.make_custom_cluster(impurity, solver_options=solver_opts)

        else:
            if args.electron_target is not None:
                cc.make_all_atom_clusters(nelectron_target=args.electron_target, solver_options=solver_opts)
            else:
                cc.make_all_atom_clusters(solver_options=solver_opts)


        if idist == 0 and MPI_rank == 0:
            cc.print_clusters()

        #cc.set_refdata(refdata)
        cc.run()
        #refdata = cc.get_refdata()

        if MPI_rank == 0:
            if idist == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCC  EmbCC(vir)  EmbCC(dem)  EmbCC(dMP2)  EmbCC(vir,dMP2)  Embcc(dem,dMP2)\n")
                    f.write("#IRC  HF  DMET  EmbCC  dMP2  EmbCC+dMP2  EmbCC(full)\n")
            with open(args.output, "a") as f:
                f.write(("%3f" + 6*"  %12.8e" + "\n") % (dist, mf.e_tot, cc.e_dmet, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))

            with open("nelectron.txt", "a") as f:
                f.write("%3f  %.8e\n" % (dist, cc.get_nelectron_total()))

