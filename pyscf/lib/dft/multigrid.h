#include <stdbool.h>
#if !defined(HAVE_DEFINED_MULTIGRID_H)
#define HAVE_DEFINED_MULTIGRID_H

#define BINOMIAL(n, i)  (_BINOMIAL_COEF[_LEN_CART0[n]+i])

extern const int _LEN_CART[];
extern const int _LEN_CART0[];
extern const int _BINOMIAL_COEF[];

typedef struct GridLevel_Info_struct {
    int nlevels;
    double rel_cutoff;
    double *cutoff;
    int *mesh;
} GridLevel_Info;

typedef struct RS_Grid_struct {
    int nlevels;
    GridLevel_Info* gridlevel_info;
    int comp;
    double** data;
} RS_Grid;

typedef struct PGFPair_struct {
    int ish;
    int ipgf;
    int jsh;
    int jpgf;
    int iL;
    double radius;
} PGFPair;

bool pgfpairs_with_same_shells(PGFPair*, PGFPair*);

typedef struct Task_struct {
    size_t buf_size;
    size_t ntasks;
    PGFPair** pgfpairs;
    double radius;
} Task;

typedef struct TaskList_struct {
    int nlevels;
    GridLevel_Info* gridlevel_info;
    Task** tasks;
} TaskList;
#endif
