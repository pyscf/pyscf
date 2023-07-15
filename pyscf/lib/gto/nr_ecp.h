#ifndef HAVE_DEFINED_ECP_H
#define HAVE_DEFINED_ECP_H
#include <stdint.h>

//#define ATOM_OF         0
//#define ANG_OF          1
#define RADI_POWER      3
#define SO_TYPE_OF      4

#define ECP_LMAX        5
//#define PTR_EXP         5
//#define PTR_COEFF       6
#define SIM_ZERO        1e-50
#define EXPCUTOFF       39   // 1e-17
#define CUTOFF          460  // ~ 1e200
#define CLOSE_ENOUGH(x, y)      (fabs(x-y) < 1e-12*fabs(y) || fabs(x-y) < 1e-12)
#define SQUARE(r)       (r[0]*r[0]+r[1]*r[1]+r[2]*r[2])
#define CART_CUM        (455+1) // upto l = 12
#define K_TAYLOR_MAX    7
#define K_TAB_COL       24      // >= (7*2+1+K_TAYLOR_MAX)
#define K_TAB_ENTRIES   400
#define K_TAB_INTERVAL  (16./K_TAB_ENTRIES)    // [0,16], interval 0.04
#define MALLOC_INSTACK(var, n) \
                var = (void *)cache; \
                cache = (void *)(((uintptr_t)(var + (n)) + 7) & (-(uintptr_t)8));
#define MARK_STACK      cache0 = cache;
#define RESTORE_STACK   cache = cache0;

// Held in env, to get *ecpbas, necpbas
#define AS_RINV_ORIG_ATOM       17
#define AS_ECPBAS_OFFSET        18
#define AS_NECPBAS              19

// for radial grids
#define LEVEL0          5
//#define LEVEL_MAX       11      // 2047 points
#define LEVEL_MAX       11
#endif

typedef struct {
    double *u_ecp;
} ECPOpt;

typedef int Function_cart(double *gctr, int *shls, int *ecpbas, int necpbas,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          ECPOpt *opt, double *cache);
