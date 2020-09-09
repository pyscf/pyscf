import logging
from mpi4py import MPI


import numpy as np
import pyscf

MPI_comm = MPI.COMM_WORLD
MPI_rank = MPI_comm.Get_rank()
MPI_size = MPI_comm.Get_size()

__all__ = [
        "run_benchmarks",
        ]

log = logging.getLogger(__name__)

def run_benchmarks(mf, benchmarks, irc, filename, print_header=True,
        # For MCSCF:
        cas_size=None, cas_space=None, core_space=None,
        # For AVAS:
        avas_ao_labels=None):
    assert mf.converged
    energies = []
    for bm in benchmarks:
        t0 = MPI.Wtime()
        if bm == "MP2":
            import pyscf.mp
            mp2 = pyscf.mp.MP2(mf)
            mp2.kernel()
            energies.append(mf.e_tot + mp2.e_corr)
        elif bm == "CISD":
            import pyscf.ci
            ci = pyscf.ci.CISD(mf)
            ci.kernel()
            #assert ci.converged
            if ci.converged:
                energies.append(mf.e_tot + ci.e_corr)
            else:
                energies.append(np.nan)
        elif bm == "CCSD":
            import pyscf.cc
            cc = pyscf.cc.CCSD(mf)
            cc.kernel()
            #assert cc.converged
            if cc.converged:
                energies.append(mf.e_tot + cc.e_corr)
            else:
                energies.append(np.nan)
        elif bm == "FCI":
            import pyscf.fci
            fci = pyscf.fci.FCI(mf.mol, mf.mo_coeff)
            fci.kernel()
            #assert fci.converged
            if fci.converged:
                energies.append(fci.e_tot)
            else:
                energies.append(np.nan)
        elif bm.endswith("CASCI") or bm.endswith("CASSCF"):
            import pyscf.mcscf

            mo_coeff = None
            if "AVAS" in bm:
                from pyscf.mcscf import avas
                norb, ne_act, mo_coeff = avas.avas(mf, avas_ao_labels, canonicalize=False)
                #norb, ne_act, mo_coeff = avas.avas(mf, avas_ao_labels, canonicalize=False, openshell_option=3)
                cas_size = (norb, ne_act)

            if bm.endswith("CASCI"):
                cas = pyscf.mcscf.CASCI(mf, cas_size[1], cas_size[0])
            elif bm.endswith("CASSCF"):
                cas = pyscf.mcscf.CASSCF(mf, cas_size[1], cas_size[0])
            if cas_space is not None and mo_coeff is None:
                mo_coeff = pyscf.mcscf.sort_mo_by_irrep(cas, mf.mo_coeff, cas_space, cas_irrep_ncore=core_space)

            cas.kernel(mo_coeff)
            #assert cas.converged
            if cas.converged:
                energies.append(cas.e_tot)
            else:
                energies.append(np.nan)

            if "NEVPT2" in bm:
                import pyscf.mrpt
                nevpt2 = pyscf.mrpt.NEVPT(cas)
                nevpt2.kernel()
                energies.append(cas.e_tot + nevpt2.e_corr)

        log.info("Time for %s: %.2g", bm, MPI.Wtime()-t0)

    if print_header:
        titles = [t.split("@")[::-1] for t in benchmarks]
        titles = [t for sub in titles for t in sub]
        titles = ["HF"] + titles
        with open(filename, "w") as f:
            f.write("#IRC  " + "  ".join(titles) + "\n")
    with open(filename, "a") as f:
        f.write(("%.3f" + ((len(energies)+1)*"  %.8e") + "\n") % (irc, mf.e_tot, *energies))
