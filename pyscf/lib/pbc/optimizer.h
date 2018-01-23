#if !defined(HAVE_DEFINED_CVHFOPT_H)
#define HAVE_DEFINED_CVHFOPT_H
typedef struct PBCOpt_struct {
    double *rrcut;
    int (*fprescreen)(int *shls, struct PBCOpt_struct *opt,
                      int *atm, int *bas, double *env);
} PBCOpt;
#endif

int PBCnoscreen(int *shls, PBCOpt *opt, int *atm, int *bas, double *env);
int PBCrcut_screen(int *shls, PBCOpt *opt, int *atm, int *bas, double *env);

