from __future__ import print_function, division
import numpy as np
from pyscf.nao.m_tools import find_nearrest_index
from pyscf.nao.m_ao_matelem import ao_matelem_c
from pyscf.nao.m_csphar import csphar

#try:
from pyscf.nao.m_comp_vext_tem_numba import get_tem_potential_numba
use_numba = True
#except:
#    use_numba = False


def comp_vext_tem(self, ao_log=None):

    def c2r_lm(conv, clm, clmm, m):
        """
            clm: sph harmonic l and m
            clmm: sph harmonic l and -m
            convert from real to complex spherical harmonic
            for an unique value of l and m
        """
        rlm = 0.0
        if m == 0:
            rlm = conv._c2r[conv._j, conv._j]*clm
        else:
            rlm = conv._c2r[m+conv._j, m+conv._j]*clm +\
                    conv._c2r[m+conv._j, -m+conv._j]*clmm

        if rlm.imag > 1e-10:
            print(rlm)
            raise ValueError("Non nul imaginary paert for c2r conversion")
        return rlm.real

    def get_index_lm(l, m):
        """
            return the index of an array ordered as 
            [l=0 m=0, l=1 m=-1, l=1 m=0, l=1 m=1, ....]
        """
        return (l+1)**2 -1 -l + m


    V_time = np.zeros((self.time.size), dtype=np.complex64)

    aome = ao_matelem_c(self.ao_log.rr, self.ao_log.pp)
    me = ao_matelem_c(self.ao_log) if ao_log is None else aome.init_one_set(ao_log)
    atom2s = np.zeros((self.natm+1), dtype=np.int64)
    for atom,sp in enumerate(self.atom2sp): 
        atom2s[atom+1]= atom2s[atom] + me.ao1.sp2norbs[sp]

    R0 = self.vnorm*self.time[0]*self.vdir + self.beam_offset
    rr = self.ao_log.rr
    dr = (np.log(rr[-1])-np.log(rr[0]))/(rr.size-1)
    dt = self.time[1]-self.time[0]
    dw = self.freq_symm[1] - self.freq_symm[0]
    wmin = self.freq_symm[0]
    tmin = self.time[0]
    nff = self.freq.size
    ub = self.freq_symm.size//2 - 1
    l2m = [] # list storing m value to corresponding l
    fact_fft = np.exp(-1.0j*self.freq_symm[ub:ub+nff]*tmin)
    for l in range(me.jmx+1):
        lm = []
        for m in range(-l, l+1):
            lm.append(m)
        l2m.append(np.array(lm))

    for atm, sp in enumerate(self.atom2sp):
        rcut = self.ao_log.sp2rcut[sp]
        center = self.atom2coord[atm, :]
        rmax = find_nearrest_index(rr, rcut)

        si = atom2s[atm]
        fi = atom2s[atm+1]

        #print("sum(V_freq) = ", np.sum(abs(self.V_freq.real)), np.sum(abs(self.V_freq.imag)))
        #print("si, fi = ", si, fi)
        for mu, l in enumerate(self.pb.prod_log.sp_mu2j[sp]):
            s = self.pb.prod_log.sp_mu2s[sp][mu]
            f = self.pb.prod_log.sp_mu2s[sp][mu+1]

            fr_val = self.pb.prod_log.psi_log[sp][mu, :]
            inte1 = np.sum(fr_val[0:rmax+1]*rr[0:rmax+1]**(l+2)*rr[0:rmax+1]*dr)

            for k in range(s, f):
                V_time.fill(0.0)

                m = l2m[l][k-s]
                ind_lm = get_index_lm(l, m)
                ind_lmm = get_index_lm(l, -m)

                if use_numba:
                    get_tem_potential_numba(self.time, R0, self.vnorm, self.vdir, center, rcut, inte1,
                        rr, dr, fr_val, me._c2r, l, m, me._j, ind_lm, ind_lmm, V_time)
                else:
                    for it, t in enumerate(self.time):
                        R_sub = R0 + self.vnorm*self.vdir*(t - self.time[0]) - center
                        norm = np.sqrt(np.dot(R_sub, R_sub))

                        if norm > rcut:
                            I1 = inte1/(norm**(l+1))
                            I2 = 0.0
                        else:
                            rsub_max = find_nearrest_index(rr, norm)

                            I1 = np.sum(fr_val[0:rsub_max+1]*
                                    rr[0:rsub_max+1]**(l+2)*rr[0:rsub_max+1])
                            I2 = np.sum(fr_val[rsub_max+1:]*
                                    rr[rsub_max+1:]/(rr[rsub_max+1:]**(l-1)))

                            I1 = I1*dr/(norm**(l+1))
                            I2 = I2*(norm**l)*dr
                        clm_tem = csphar(R_sub, l)
                        clm = (4*np.pi/(2*l+1))*clm_tem[ind_lm]*(I1 + I2)
                        clmm = (4*np.pi/(2*l+1))*clm_tem[ind_lmm]*(I1 + I2)
                        rlm = c2r_lm(me, clm, clmm, m)
                        V_time[it] = rlm + 0.0j
                
                V_time *= dt*np.exp(-1.0j*wmin*(self.time-tmin))

                FT = np.fft.fft(V_time)
                self.V_freq[:, si + k] = FT[ub:ub+nff]*fact_fft
