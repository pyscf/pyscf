/* Copyright 2022 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>

/*
 * tmp = fg[...,[[0,1],[1,2]],...]
 * tmp[...,0,0,...] *= 2
 * tmp[...,1,1,...] *= 2
 * qg = np.einsum('nabmg,bxg->naxmg', tmp, rho[:,1:4])
 */
void VXCfg_to_direct_deriv(double *qg, double *fg, double *rho,
                           int ncounts, int nvar, int mcounts, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t mg = mcounts * Ngrids;
        double *rho_ax = rho + Ngrids;
        double *rho_ay = rho_ax + Ngrids;
        double *rho_az = rho_ay + Ngrids;
        double *rho_bx = rho_ax + nvar * Ngrids;
        double *rho_by = rho_ay + nvar * Ngrids;
        double *rho_bz = rho_az + nvar * Ngrids;
        double *qg_ax, *qg_ay, *qg_az, *qg_bx, *qg_by, *qg_bz;
        double *fg_aa, *fg_ab, *fg_bb;
        double vaa, vab, vbb;
        int n, m, g;
        for (n = 0; n < ncounts; n++) {
                qg_ax = qg + n * 2 * 3 * mg;
                qg_ay = qg_ax + mg;
                qg_az = qg_ay + mg;
                qg_bx = qg_ax + 3 * mg;
                qg_by = qg_ay + 3 * mg;
                qg_bz = qg_az + 3 * mg;
                fg_aa = fg + n * 3 * mg;
                fg_ab = fg_aa + mg;
                fg_bb = fg_ab + mg;
                for (m = 0; m < mcounts; m++) {
#pragma GCC ivdep
                for (g = 0; g < ngrids; g++) {
                        vaa = fg_aa[m*Ngrids+g] * 2;
                        vab = fg_ab[m*Ngrids+g];
                        vbb = fg_bb[m*Ngrids+g] * 2;
                        qg_ax[m*Ngrids+g] = vaa * rho_ax[g] + vab * rho_bx[g];
                        qg_ay[m*Ngrids+g] = vaa * rho_ay[g] + vab * rho_by[g];
                        qg_az[m*Ngrids+g] = vaa * rho_az[g] + vab * rho_bz[g];
                        qg_bx[m*Ngrids+g] = vbb * rho_bx[g] + vab * rho_ax[g];
                        qg_by[m*Ngrids+g] = vbb * rho_by[g] + vab * rho_ay[g];
                        qg_bz[m*Ngrids+g] = vbb * rho_bz[g] + vab * rho_az[g];
                } }
        }
}

void VXCud2ts_deriv1(double *v_ts, double *v_ud, int nvar, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t vg = nvar * Ngrids;
        double *vu = v_ud;
        double *vd = v_ud + vg;
        double *vt = v_ts;
        double *vs = v_ts + vg;
        size_t n;
#pragma GCC ivdep
        for (n = 0; n < vg; n++) {
                vt[n] = (vu[n] + vd[n]) * .5;
                vs[n] = (vu[n] - vd[n]) * .5;
        }
}

void VXCud2ts_deriv2(double *v_ts, double *v_ud, int nvar, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t vg = nvar * Ngrids;
        size_t vg2 = vg * 2;
        size_t vvg = nvar * vg2;
        double *vuu = v_ud;
        double *vud = v_ud + vg;
        double *vdu = vuu + vvg;
        double *vdd = vud + vvg;
        double *vtt = v_ts;
        double *vts = v_ts + vg;
        double *vst = vtt + vvg;
        double *vss = vts + vvg;
        double ut, us, dt, ds;
        size_t i, n;
        for (i = 0; i < nvar; i++) {
#pragma GCC ivdep
                for (n = 0; n < vg; n++) {
                        ut = vuu[i*vg2+n] + vud[i*vg2+n];
                        us = vuu[i*vg2+n] - vud[i*vg2+n];
                        dt = vdu[i*vg2+n] + vdd[i*vg2+n];
                        ds = vdu[i*vg2+n] - vdd[i*vg2+n];
                        vtt[i*vg2+n] = (ut + dt) * .25;
                        vts[i*vg2+n] = (us + ds) * .25;
                        vst[i*vg2+n] = (ut - dt) * .25;
                        vss[i*vg2+n] = (us - ds) * .25;
                }
        }
}

void VXCud2ts_deriv3(double *v_ts, double *v_ud, int nvar, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t vg = nvar * Ngrids;
        size_t vg2 = vg * 2;
        size_t vvg = nvar * vg2;
        size_t vvg2 = vvg * 2;
        size_t vvvg = nvar * vvg2;
        double *vuuu = v_ud;
        double *vuud = v_ud + vg;
        double *vudu = vuuu + vvg;
        double *vudd = vuud + vvg;
        double *vduu = vuuu + vvvg;
        double *vdud = vuud + vvvg;
        double *vddu = vudu + vvvg;
        double *vddd = vudd + vvvg;
        double *vttt = v_ts;
        double *vtts = v_ts + vg;
        double *vtst = vttt + vvg;
        double *vtss = vtts + vvg;
        double *vstt = vttt + vvvg;
        double *vsts = vtts + vvvg;
        double *vsst = vtst + vvvg;
        double *vsss = vtss + vvvg;
        double uut, uus, udt, uds, dut, dus, ddt, dds;
        double utt, uts, ust, uss, dtt, dts, dst, dss;
        size_t i, j, ij, n;
        for (i = 0; i < nvar; i++) {
        for (j = 0; j < nvar; j++) {
                ij = (i * nvar * 2 + j) * vg2;
#pragma GCC ivdep
                for (n = 0; n < vg; n++) {
                        uut = vuuu[ij+n] + vuud[ij+n];
                        uus = vuuu[ij+n] - vuud[ij+n];
                        udt = vudu[ij+n] + vudd[ij+n];
                        uds = vudu[ij+n] - vudd[ij+n];
                        dut = vduu[ij+n] + vdud[ij+n];
                        dus = vduu[ij+n] - vdud[ij+n];
                        ddt = vddu[ij+n] + vddd[ij+n];
                        dds = vddu[ij+n] - vddd[ij+n];
                        utt = uut + udt;
                        uts = uus + uds;
                        ust = uut - udt;
                        uss = uus - uds;
                        dtt = dut + ddt;
                        dts = dus + dds;
                        dst = dut - ddt;
                        dss = dus - dds;
                        vttt[ij+n] = (utt + dtt) * .125;
                        vtts[ij+n] = (uts + dts) * .125;
                        vtst[ij+n] = (ust + dst) * .125;
                        vtss[ij+n] = (uss + dss) * .125;
                        vstt[ij+n] = (utt - dtt) * .125;
                        vsts[ij+n] = (uts - dts) * .125;
                        vsst[ij+n] = (ust - dst) * .125;
                        vsss[ij+n] = (uss - dss) * .125;
                }
        } }
}
