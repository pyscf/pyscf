from pyscf.nao.m_numba_utils import comp_coeffs_numba
from pyscf.nao.m_rsphar import rsphar
import numba as nb
import numpy as np

rsphar_nb = nb.jit(nopython=True)(rsphar)
comp_coeffs_nb = nb.jit(nopython=True)(comp_coeffs_numba)

def get_spatial_density_numba(dn_spatial, mu2dn, mesh, atom2sp, atom2coord, 
        atom2s, sp_mu2j, psi_log_rl, sp_mu2s, sp2rcut, rr, res, coeffs, gammin_jt,
        dg_jt, nr):

    for ix, x in enumerate(mesh[0]):
        print(ix, mesh[0].shape)
        for iy, y in enumerate(mesh[1]):
            for iz, z in enumerate(mesh[2]):
                br = np.array([x, y, z])
                for atm, sp in enumerate(atom2sp):
                    brp = br - atom2coord[atm, :]
                    r = np.sqrt(np.dot(brp, brp))
                    rcut = sp2rcut[sp]

                    si = atom2s[atm]
                    fi = atom2s[atm+1]


                    if r>rcut: continue
                    jmx_sp = sp_mu2j[sp].max()
                    rsh = np.zeros((jmx_sp+1)**2)
                    rsphar_nb(brp, jmx_sp, rsh)

                    ir = comp_coeffs_nb(gammin_jt, dg_jt, nr, r, coeffs)
                    for ij, j in enumerate(sp_mu2j[sp]):
                        ff = psi_log_rl[sp][ij]
                        s = sp_mu2s[sp][ij]
                        f = sp_mu2s[sp][ij+1]
                        if j == 0:
                            fval = np.sum(ff[ir:ir+6]*coeffs)
                        else: 
                            fval = np.sum(ff[ir:ir+6]*coeffs)*r**j
                        
                        res[si+s:si+f] = rsh[j*(j+1)-j:j*(j+1)+j+1]*fval

                dn_spatial[ix, iy, iz] += np.sum(res*mu2dn)

def get_spatial_density_numba_parallel(dn_spatial, mu2dn, mesh, atom2sp, atom2coord, 
        atom2s, sp_mu2j, psi_log_rl, sp_mu2s, sp2rcut, rr, res, coeffs, gammin_jt,
        dg_jt, nr):

    a = 0.0
    for ix in nb.prange(mesh[0].size):
        x = mesh[0][ix]
        print(ix, mesh[0].shape)
        for iy in range(mesh[1].size):
            y = mesh[1][iy]
            for iz in range(mesh[2].size):
                z = mesh[2][iz]
                br = np.array([x, y, z])
                for atm in range(atom2sp.size):
                    sp = atom2sp[atm]
                    brp = br - atom2coord[atm, :]
                    r = np.sqrt(np.dot(brp, brp))
                    rcut = sp2rcut[sp]

                    si = atom2s[atm]
                    fi = atom2s[atm+1]

                    if r>rcut: continue
#                    jmx_sp = np.max(sp_mu2j[sp])
#                    rsh = np.zeros((jmx_sp+1)**2)
#                    rsphar_nb(brp, jmx_sp, rsh)
#
#                    ir = comp_coeffs_nb(gammin_jt, dg_jt, nr, r, coeffs)
#                    for ij, j in enumerate(sp_mu2j[sp]):
#                        ff = psi_log_rl[sp][ij]
#                        s = sp_mu2s[sp][ij]
#                        f = sp_mu2s[sp][ij+1]
#                        if j == 0:
#                            fval = np.sum(ff[ir:ir+6]*coeffs)
#                        else: 
#                            fval = np.sum(ff[ir:ir+6]*coeffs)*r**j
#                        
#                        res[si+s:si+f] = rsh[j*(j+1)-j:j*(j+1)+j+1]*fval
#
#                dn_spatial[ix, iy, iz] += np.sum(res*mu2dn)
