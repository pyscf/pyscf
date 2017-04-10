import numpy as np
import shutil
import os.path
import sys
import os
from pyscf.pbc import cc as pbccc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_eom_krccsd_bands(cell, nmp, kpts_red, suffix="-band.dat"):
    from scf import run_khf
    e_kn = []
    qp_kn = []
    vbmax = -99
    cbmin = 99

    nip = 3
    nea = 3

    for kpt in kpts_red:
        if rank == 0:
            filename = "kpt_%.4f_%.4f_%.4f"%(kpt[0], kpt[1], kpt[2])
            filename += suffix

            print filename

            if os.path.isfile(filename):
                f = open(filename,"r")
                lines = f.readlines()
                lines = [lines[i].strip(' \n') for i in range(len(lines))]
                lines = filter(None,lines)
                comment_lines = [x.find('#') == -1 for x in lines]
                # Finding beginning and end of IP values, ignoring comments
                #
                for pos,line in enumerate(comment_lines):
                    if line == True:
                        IP_begin = pos
                        break
                for pos,line in enumerate(comment_lines[IP_begin:]):
                    if line == False:
                        IP_end = IP_begin + pos
                        break
                # Finding beginning and end of EA values, ignoring comments
                #
                comment_lines = comment_lines[::-1]
                for pos,line in enumerate(comment_lines):
                    if line == True:
                        EA_end = pos
                        break
                print comment_lines
                for pos,line in enumerate(comment_lines[EA_end:]):
                    if line == False:
                        EA_begin = EA_end + pos
                        break
                EA_end = len(lines) - EA_end
                EA_begin = len(lines) - EA_begin
                IP_lines = lines[IP_begin:IP_end]
                EA_lines = lines[EA_begin:EA_end]
                print "IP lines ", lines[IP_begin:IP_end]
                print "EA lines ", lines[EA_begin:EA_end]

                IP_lines = np.array([x.split() for x in IP_lines])
                EA_lines = np.array([x.split() for x in EA_lines])
                f.close()
            else:
                IP_lines = np.ones((nip,2))*999.9
                IP_lines[:,1] *= 0.0
                EA_lines = np.ones((nea,2))*999.9
                EA_lines[:,1] *= 0.0

            eip  = np.array(IP_lines[:,0], dtype=float)
            qpip = np.array(IP_lines[:,1], dtype=float)
            eea  = np.array(EA_lines[:,0], dtype=float)
            qpea = np.array(EA_lines[:,1], dtype=float)
            print eip
            print eea

            e_kn.append( np.append(-eip[::-1], eea) )
            qp_kn.append( np.append(qpip[::-1], qpea) )

        if np.max(-eip) > vbmax:
            vbmax = np.max(-eip)
        if np.min(eea) < cbmin:
            cbmin = np.min(eea)

    for k, ek in enumerate(e_kn):
        e_kn[k] = ek-vbmax
    bandgap = cbmin - vbmax

    return e_kn, qp_kn, bandgap

def main():
    import sys
    sys.path.append('/home/jmcclain/pyscf/pyscf/pbc/examples/')
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
    cell = build_cell(ase_atom, ke=40.0, basis=bas, incore_anyway=True)

    kpts_red, kpts_cart, kpath, sp_points = get_bandpath_fcc(ase_atom,npoints=30)

    e_kn, qp_kn, bandgap = read_eom_krccsd_bands(cell, nmp, kpts_red[start_band:end_band,:])
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

    e_kn, qp_kn, bandgap = read_eom_krccsd_bands(cell, nmp, kpts_red[start_band:end_band,:],suffix="-STAR-band.dat")
    if rank == 0:
        filename = "%s_%s_%d%d%d-STAR-bands.dat"%(formula.lower(), bas[4:],
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

if __name__ == '__main__':
    main()
