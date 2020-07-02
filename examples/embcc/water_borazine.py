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

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-b", "--basis", default="cc-pVDZ")
parser.add_argument("--solver", choices=["CISD", "CCSD", "FCI"], default="CCSD")
parser.add_argument("--benchmark", choices=["CISD", "CCSD", "FCI"])
#parser.add_argument("--tol-bath", type=float, default=1e-3)
parser.add_argument("--distances", type=float, nargs="*")
parser.add_argument("--distances-range", type=float, nargs=3, default=[2.8, 8.0, 0.2])
parser.add_argument("--local-type", choices=["IAO", "AO", "LAO"], default="IAO")
parser.add_argument("--bath-type")
parser.add_argument("--bath-size", type=float, nargs=2)
parser.add_argument("--impurity", nargs="*", default=["O1", "H1", "H2", "N1"])
#parser.add_argument("--impurity", nargs="*", default=["O1", "H1", "H2", "N1", "B1", "B3"])
parser.add_argument("--mp2-correction", action="store_true")
parser.add_argument("--use-ref-orbitals-bath", type=int, default=0)
parser.add_argument("--minao", default="minao")
parser.add_argument("-o", "--output", default="energies.txt")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

args.use_ref_orbitals_bath = bool(args.use_ref_orbitals_bath)

if args.distances is None:
    args.distances = np.arange(args.distances_range[0], args.distances_range[1]+1e-14, args.distances_range[2])
del args.distances_range

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

structure_builder = molstructures.build_water_borazine

ref_orbitals = None
refdata = None

for idist, dist in enumerate(args.distances):

    if MPI_rank == 0:
        log.info("Distance=%.2f", dist)
        log.info("=============")

    mol = structure_builder(dist, basis=args.basis, verbose=4)

    mf = pyscf.scf.RHF(mol)
    #mf = mf.density_fit()
    mf.kernel()
    assert mf.converged

    if args.benchmark:
        if args.benchmark == "CISD":
            import pyscf.ci
            cc = pyscf.ci.CISD(mf)
        elif args.benchmark == "CCSD":
            import pyscf.cc
            cc = pyscf.cc.CCSD(mf)
        elif args.benchmark == "FCI":
            import pyscf.fci
            cc = pyscf.fci.FCI(mol, mf.mo_coeff)
        cc.kernel()
        assert cc.converged

        with open(args.output, "a") as f:
            f.write("%3f  %.8e  %.8e\n" % (dist, mf.e_tot, cc.e_tot))

    else:
        cc = embcc.EmbCC(mf,
                local_type=args.local_type,
                minao=args.minao,
                bath_type=args.bath_type, bath_size=args.bath_size,
                mp2_correction=args.mp2_correction,
                use_ref_orbitals_bath=args.use_ref_orbitals_bath,
                )
        cc.make_atom_cluster(args.impurity)
        if idist == 0 and MPI_rank == 0:
            cc.print_clusters()

        if ref_orbitals is not None:
            cc.set_reference_orbitals(ref_orbitals)

        cc.set_refdata(refdata)
        cc.run()
        refdata = cc.get_refdata()

        ref_orbitals = cc.get_orbitals()

        if MPI_rank == 0:
            if idist == 0:
                with open(args.output, "a") as f:
                    f.write("#IRC  HF  EmbCC  EmbCC(vir)  EmbCC(dem)  EmbCC(dMP2)  EmbCC(vir,dMP2)  Embcc(dem,dMP2)\n")
            with open(args.output, "a") as f:
                f.write(("%3f" + 7*"  %12.8e" + "\n") %
                        (dist, mf.e_tot,
                        cc.e_tot, mf.e_tot+cc.e_corr_v, mf.e_tot+cc.e_corr_d,
                        cc.e_tot + cc.e_delta_mp2, mf.e_tot+cc.e_corr_v+cc.e_delta_mp2, mf.e_tot+cc.e_corr_d + cc.e_delta_mp2))
