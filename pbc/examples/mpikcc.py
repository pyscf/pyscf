import numpy as np
from pyscf.pbc import mpicc as pbccc
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run_kccsd(mf):
    cc = pbccc.KCCSD(mf)
    cc.verbose = 7
    cc.ccsd()
    return cc

def run_krccsd(mf):
    cc = pbccc.KRCCSD(mf)
    cc.verbose = 7
    cc.ccsd()
    return cc

def run_ip_krccsd(cc, kptlist=[0], nroots=3):
    e,c = cc.ipccsd(nroots, kptlist)
    return e,c

def run_lip_krccsd(cc, kptlist=[0], nroots=3):
    e,c = cc.lipccsd(nroots, kptlist)
    return e,c

def run_ea_krccsd(cc, kptlist=[0], nroots=3):
    e,c = cc.eaccsd(nroots, kptlist)
    return e,c

def run_lea_krccsd(cc, kptlist=[0], nroots=3):
    e,c = cc.leaccsd(nroots, kptlist)
    return e,c

def run_eom_krccsd_bands(cell, nmp, kpts_red):
    from scf import run_khf
    e_kn = []
    qp_kn = []
    vbmax = -99
    cbmin = 99

    for kpt in kpts_red:
        mf = run_khf(cell, nmp=nmp, kshift=kpt, gamma=True, exxdiv=None)
        # Setting up everything for MPI after mean field
        comm.Barrier()
        mo_coeff  = comm.bcast(mf.mo_coeff,root=0)
        mo_energy = comm.bcast(mf.mo_energy,root=0)
        mo_occ    = comm.bcast(mf.mo_occ,root=0)
        kpts      = comm.bcast(mf.kpts,root=0)
        mf.mo_coeff = mo_coeff
        mf.mo_energy = mo_energy
        mf.mo_occ = mo_occ
        mf.kpts   = kpts
        comm.Barrier()
        # Done with setting up

        # Running ccsd
        cc = run_krccsd(mf)
        nocc = cc.nocc()
        nvir = cc.nmo() - nocc

        eip,cip = run_ip_krccsd(cc, kptlist=[0])
        eip, cip = eip[0], cip[0]
        qpip = np.linalg.norm(cip[:nocc],axis=0)**2

        eea,cea = run_ea_krccsd(cc, kptlist=[0])
        eea, cea = eea[0], cea[0]
        qpea = np.linalg.norm(cea[:nvir],axis=0)**2

        e_kn.append( np.append(-eip[::-1], eea) )
        qp_kn.append( np.append(qpip[::-1], qpea) )
        if rank == 0:
            filename = "kpt_%.4f_%.4f_%.4f-band.dat"%(kpt[0], kpt[1], kpt[2])
            f = open(filename,'w')
            f.write("# IP\n")
            for ekn, qpkn in zip(eip,qpip):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("# EA \n")
            for ekn, qpkn in zip(eea,qpea):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("\n")
            f.close()
        if np.max(-eip) > vbmax:
            vbmax = np.max(-eip)
        if np.min(eea) < cbmin:
            cbmin = np.min(eea)

        eip,lcip = run_lip_krccsd(cc, kptlist=[0])
        eip, lcip = eip[0], lcip[0]

        eea,lcea = run_lea_krccsd(cc, kptlist=[0])
        eea, lcea = eea[0], lcea[0]

        eip_star = cc.ipccsd_star(eip,cip,lcip)
        eip_star = eip_star.real

        eea_star = cc.eaccsd_star(eea,cea,lcea)
        eea_star = eea_star.real

        if rank == 0:
            filename = "kpt_%.4f_%.4f_%.4f-STAR-band.dat"%(kpt[0], kpt[1], kpt[2])
            f = open(filename,'w')
            f.write("# IP\n")
            for ekn, qpkn in zip(eip_star,qpip):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("# EA \n")
            for ekn, qpkn in zip(eea_star,qpea):
                f.write("%0.6f %0.6f \n"%(ekn, qpkn))
            f.write("\n")
            f.close()

        cc = None

        if rank == 0:
            if os.path.isfile("eris1.hdf5") is True:
                os.remove("eris1.hdf5")
                #shutil.rmtree('./tmp')
        comm.Barrier()

    for k, ek in enumerate(e_kn):
        e_kn[k] = ek-vbmax
    bandgap = cbmin - vbmax
    return e_kn, qp_kn, bandgap

if __name__ == '__main__':
    import sys
    from helpers import get_ase_diamond_primitive, build_cell
    from scf import run_khf
    from helpers import get_ase_atom, build_cell, get_bandpath_fcc

    args = sys.argv[1:]
    if len(args) != 5 and len(args) != 7:
        print 'usage: formula basis nkx nky nkz [start_band end_band]'
        sys.exit(1)
    formula = args[0]
    bas = args[1]
    nmp = np.array([int(nk) for nk in args[2:5]])
    start_band = 0
    end_band = 30
    if len(args) == 7:
        start_band = int(args[5])
        end_band =   int(args[6])

    ase_atom = get_ase_atom(formula)
    cell = build_cell(ase_atom, ke=40.0, basis=bas, incore_anyway=True, verbose=6)

    kpts_red, kpts_cart, kpath, sp_points = get_bandpath_fcc(ase_atom,npoints=30)

    e_kn, qp_kn, bandgap = run_eom_krccsd_bands(cell, nmp, kpts_red[start_band:end_band,:])

    if rank == 0:
        filename = "%s_%s_%d%d%d-bands.dat"%(formula.lower(), bas[4:],
                                             nmp[0], nmp[1], nmp[2])
        f = open(filename,'w')
        f.write("# Bandgap = %0.6f au = %0.6f eV\n"%(bandgap, bandgap*27.2114))
        f.write("# Special points:\n")
        for point, label in zip(sp_points,['L', 'G', 'X', 'W', 'K', 'G']):
            f.write("# %0.6f %s\n"%(point,label))
        for kk, ek, qpk in zip(kpath, e_kn, qp_kn):
            f.write("%0.6f "%(kk))
            for ekn, qpkn in zip(ek,qpk):
                f.write("%0.6f %0.6f "%(ekn, qpkn))
            f.write("\n")
        f.close()
