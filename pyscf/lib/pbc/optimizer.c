#include <stdlib.h>
#include "cint.h"
#include "pbc/optimizer.h"

#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])

void PBCinit_optimizer(PBCOpt **opt, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        PBCOpt *opt0 = malloc(sizeof(PBCOpt));
        opt0->rrcut = NULL;
        opt0->fprescreen = &PBCnoscreen;
        *opt = opt0;
}

void PBCdel_optimizer(PBCOpt **opt)
{
        PBCOpt *opt0 = *opt;
        if (!opt0) {
                return;
        }

        if (!opt0->rrcut) {
                free(opt0->rrcut);
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
        if (!opt) {
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

int PBCset_rcut_cond(PBCOpt *opt, double *rcut,
                     int *atm, int natm, int *bas, int nbas, double *env)
{
        if (opt->rrcut) {
                free(opt->rrcut);
        }
        opt->rrcut = (double *)malloc(sizeof(double) * nbas);
        opt->fprescreen = &PBCrcut_screen;

        int i;
        for (i = 0; i < nbas; i++) {
                opt->rrcut[i] = rcut[i] * rcut[i];
        }
}
