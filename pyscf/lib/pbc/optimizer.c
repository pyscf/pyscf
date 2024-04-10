/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
  
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
#include <math.h>
#include "cint.h"
#include "pbc/optimizer.h"

#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])

void PBCinit_optimizer(PBCOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        PBCOpt *opt0 = malloc(sizeof(PBCOpt));
        opt0->rrcut = NULL;
        opt0->rcut = NULL;
        opt0->fprescreen = &PBCnoscreen;
        *opt = opt0;
}

void PBCdel_optimizer(PBCOpt **opt)
{
        PBCOpt *opt0 = *opt;
        if (opt0 == NULL) {
                return;
        }

        if (opt0->rrcut != NULL) {
                free(opt0->rrcut);
        }
        if (!opt0->rcut) {
                free(opt0->rcut);
        }
        free(opt0);
        *opt = NULL;
}

int PBCnoscreen(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        return 1;
}

int PBCrcut_screen(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const double *ri = env + atm[bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double *rj = env + atm[bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        double rr = SQUARE(rirj);
        return (rr < opt->rrcut[ish] || rr < opt->rrcut[jsh]);
}

int PBCrcut_screen_loose(int *shls, PBCOpt *opt, int *atm, int *bas, double *env)
{
        if (opt == NULL) {
                return 1; // no screen
        }
        const int ish = shls[0];
        const int jsh = shls[1];
        const double *ri = env + atm[bas[ATOM_OF+ish*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        const double *rj = env + atm[bas[ATOM_OF+jsh*BAS_SLOTS]*ATM_SLOTS+PTR_COORD];
        double rirj[3];
        rirj[0] = ri[0] - rj[0];
        rirj[1] = ri[1] - rj[1];
        rirj[2] = ri[2] - rj[2];
        double r = sqrt(SQUARE(rirj));
        return r < opt->rcut[ish] + opt->rcut[jsh];
}

void PBCset_rcut_cond(PBCOpt *opt, double *rcut,
                      int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->rrcut != NULL) {
                free(opt->rrcut);
        }
        opt->rrcut = (double *)malloc(sizeof(double) * nbas);
        opt->fprescreen = &PBCrcut_screen;

        int i;
        for (i = 0; i < nbas; i++) {
                opt->rrcut[i] = rcut[i] * rcut[i];
        }
}

void PBCset_rcut_cond_loose(PBCOpt *opt, double *rcut,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->rcut != NULL) {
                free(opt->rcut);
        }
        opt->rcut = (double *)malloc(sizeof(double) * nbas);
        opt->fprescreen = &PBCrcut_screen_loose;

        int i;
        for (i = 0; i < nbas; i++) {
                opt->rcut[i] = rcut[i];
        }
}
