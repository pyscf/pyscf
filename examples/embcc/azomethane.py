import sys
import logging
import argparse
import functools

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
from pyscf import molstructures
from pyscf import embcc

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="6-31g**")
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="FCI")
parser.add_argument("--benchmarks", nargs="*", choices=["MP2", "CISD", "CCSD", "FCI"])
parser.add_argument("--distances", type=float, nargs="*")
        #default=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.6, 3.0, 4.0, 5.0])
parser.add_argument("--distances-range", type=float, nargs=3, default=[1.0, 4.0, 0.1])
parser.add_argument("--local-type", choices=["IAO", "AO", "LAO"], default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--bath-size", type=float, nargs=2, default=[0.0, 0.0])
parser.add_argument("--impurity", nargs="*",
        default=["N1", "N2"]
        )
parser.add_argument("--minao", default="minao")
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])
del args.distances_range

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

structure_builder = molstructures.build_azomethane

dm0 = None
refdata = None

for idist, dist in enumerate(args.distances):

    if MPI_rank == 0:
        log.info("Distance=%.2f", dist)
        log.info("=============")

    mol = structure_builder(dist, basis=args.basis, verbose=4)

    mf = pyscf.scf.RHF(mol)
    t0 = MPI.Wtime()
    mf.kernel(dm0=dm0)
    log.info("Time for mean-field: %.2g", MPI.Wtime()-t0)
    assert mf.converged
    dm0 = mf.make_rdm1()

    if args.benchmarks:

        energies = []

        casscf = pyscf.mcscf.CASCI(mf, 8, 8)
        casscf.kernel()
        energies.append(casscf.e_tot)

        casscf = pyscf.mcscf.CASSCF(mf, 8, 8)
        casscf.kernel()
        energies.append(casscf.e_tot)

        #energies = []
        #for bm in args.benchmarks:
        #    t0 = MPI.Wtime()
        #    if bm == "MP2":
        #        import pyscf.mp
        #        mp2 = pyscf.mp.MP2(mf)
        #        mp2.kernel()
        #        energies.append(mf.e_tot + mp2.e_corr)
        #    elif bm == "CISD":
        #        import pyscf.ci
        #        ci = pyscf.ci.CISD(mf)
        #        ci.kernel()
        #        assert ci.converged
        #        energies.append(mf.e_tot + ci.e_corr)
        #    elif bm == "CCSD":
        #        import pyscf.cc
        #        cc = pyscf.cc.CCSD(mf)
        #        cc.kernel()
        #        assert cc.converged
        #        energies.append(mf.e_tot + cc.e_corr)
        #    elif bm == "FCI":
        #        import pyscf.fci
        #        fci = pyscf.fci.FCI(mol, mf.mo_coeff)
        #        fci.kernel()
        #        assert fci.converged
        #        energies.append(mf.e_tot + fci.e_corr)
        #    log.info("Time for %s: %.2g", bm, MPI.Wtime()-t0)

        #if idist == 0:
        #    with open(args.output, "w") as f:
        #        f.write("#distance  HF  " + "  ".join(args.benchmarks) + "\n")
        with open(args.output, "a") as f:
            f.write(("%.3f  %.8e" + (len(energies)*"  %.8e") + "\n") % (dist, mf.e_tot, *energies))

    else:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                minao=args.minao,
                bath_type=args.bath_type, bath_size=args.bath_size,
                )
        #cc.make_atom_cluster(args.impurity, solver=args.solver)
        #cc.make_atom_cluster(args.impurity, solver=args.solver)
        #cc.make_custom_cluster(["N1 2p", "N2 2p"], solver=args.solver)
        #cc.make_custom_cluster(["N1 2p"], solver=args.solver)
        #cc.make_custom_cluster(["N2 2p"], solver=args.solver)
        cc.make_custom_cluster(["N1 2s", "N1 2p"], solver=args.solver, symmetry_factor=2)
        cc.make_custom_cluster(["N1 1s", "C3 1s", "C3 2s", "C3 2p"], solver="CCSD", symmetry_factor=2)

        #cc.make_custom_cluster(["C3 2p x", "C3 2p y", "N1 2s", "N1 2p"], solver=args.solver, symmetry_factor=2)
        #cc.make_custom_cluster(["N1 2s", "N2 2s", "N1 2p y", "N2 2p y"], solver=args.solver, symmetry_factor=1)
        #cc.make_custom_cluster(["N1 2p y", "N2 2p y", "N1 2p z", "N2 2p z"], solver=args.solver, symmetry_factor=1)

        #cc.make_custom_cluster(["C3 2p", "N1 2p"], solver=args.solver, symmetry_factor=2)
        #cc.make_custom_cluster(["N1 1s", "N1 2s", "N1 2p"], solver=args.solver, symmetry_factor=2)
        #cc.make_custom_cluster(["C3 2p", "N1 2s", "N1 2p"], solver=args.solver, symmetry_factor=2)
        #cc.make_custom_cluster(["N1 2s", "N1 2p", "N2 2s", "N2 2p"], solver=args.solver)


        if idist == 0 and MPI_rank == 0:
            cc.print_clusters()

        cc.set_refdata(refdata)
        cc.run()
        refdata = cc.get_refdata()

        if MPI_rank == 0:
            if idist == 0:
                with open(args.output, "a") as f:
                    #f.write("#IRC  HF  EmbCC  EmbCC(vir)  EmbCC(dem)  EmbCC(dMP2)  EmbCC(vir,dMP2)  Embcc(dem,dMP2)\n")
                    f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2  EmbCC(full)\n")
            with open(args.output, "a") as f:
                f.write(("%3f" + 5*"  %12.8e" + "\n") % (dist, mf.e_tot, cc.e_tot, cc.e_delta_mp2, cc.e_tot+cc.e_delta_mp2, mf.e_tot+cc.e_corr_full))
