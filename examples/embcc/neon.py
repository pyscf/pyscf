import sys
import logging
import argparse

import numpy as np
from mpi4py import MPI

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mp
import pyscf.cc
from pyscf import embcc

from util import run_benchmarks

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--basis", default="cc-pVQZ")
parser.add_argument("--solver", default="CCSD")
parser.add_argument("--atom", default="Ne")
parser.add_argument("--dimer-distance", type=float)
parser.add_argument("--spin", type=int, default=0)
parser.add_argument("--benchmarks", nargs="*")
parser.add_argument("--local-type", default="IAO")
parser.add_argument("--bath-type", default="mp2-natorb")
parser.add_argument("--prefix", default="")
args, restargs = parser.parse_known_args()
sys.argv[1:] = restargs

if MPI_rank == 0:
    log.info("Parameters")
    log.info("----------")
    for name, value in sorted(vars(args).items()):
        log.info("%10s: %r", name, value)

if args.dimer_distance is not None:
    mol = pyscf.gto.M(atom="%s1 0 0 0; %s2 0 0 %f" % (args.atom, args.atom, args.dimer_distance), basis=args.basis, spin=args.spin, verbose=4)
else:
    mol = pyscf.gto.M(atom="%s 0 0 0" % args.atom, basis=args.basis, spin=args.spin, verbose=4)

mf = pyscf.scf.RHF(mol)
t0 = MPI.Wtime()
mf.kernel()
mo_stab = mf.stability()[0]
stable = np.allclose(mo_stab, mf.mo_coeff)
log.info("Time for HF (s): %.3f", (MPI.Wtime()-t0))
assert stable
assert mf.converged

mp2 = pyscf.mp.MP2(mf)
mp2.kernel()

#print(mp2.e_tot)
#
#cc = pyscf.cc.CCSD(mf)
#cc.kernel()
#
#print(cc.e_tot)
#
#1/0

#FNOCCSD
#fnotols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-6]
#fnotols = [1e-1, 1e-2, 1e-3, 1e-4]
tols = np.logspace(0, -29, num=30, base=2)

#if True:
if False:
    for tol in tols:
        fnocc = pyscf.cc.FNOCCSD(mf, thresh=tol)
        fnocc.kernel()
        with open("%s-fnoccsd-%s-%s.txt" % (args.prefix, args.atom, args.basis), "a") as f:
            f.write("%3d  %16.12e  %16.12e\n" % (fnocc.nmo, fnocc.e_tot, fnocc.e_tot + fnocc.delta_emp2))


#etols = fnotols
for i, tol in enumerate(tols):

    cc = embcc.EmbCC(mf,
            local_type=args.local_type,
            bath_type=args.bath_type,
            minao="sto-3g",
            #bath_energy_tol=tol,
            #bath_energy_tol=tol,
            bath_tol=tol,
            solver=args.solver,
            )

    if args.dimer_distance is not None:
        symfac = 2.0
    else:
        symfac = 1.0
    solver_opts = {}

    #cc.make_all_atom_clusters(solver_options=solver_opts)
    cc.make_atom_cluster("%s1" % args.atom, symmetry_factor=symfac, solver_options=solver_opts)
    cc.run()

    nact = cc.clusters[0].nactive
    print("Active orbitals = %d" % nact)

    output = "%s-embcc-%s-%s.txt" % (args.prefix, args.atom, args.basis)
    if MPI_rank == 0:
        if (i == 0):
            with open(output, "a") as f:
                f.write("#IRC  HF  EmbCC  dMP2  EmbCC+dMP2\n")
        with open(output, "a") as f:
            f.write(("%3d  " + 3*"  %16.12e" + "\n") % (nact, mf.e_tot, mf.e_tot+cc.e_corr, mf.e_tot+cc.e_corr+cc.e_delta_mp2))
