/*
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

// 2 slots of int param[]
#define POS_E1   0
#define TENSOR   1

// 128s42p21d12f8g6h4i3j 
#define NCTR_CART      128
//  72s24p14d10f8g6h5i4j 
#define NCTR_SPH        72
#define NPRIMAX         64
#define BLKSIZE         64
#define EXPCUTOFF       50  // 1e-22
#define NOTZERO(e)      ((e)>1e-18 || (e)<-1e-18)

void GTOnabla1(double *fx1, double *fy1, double *fz1,
               double *fx0, double *fy0, double *fz0, int l, double a);
void GTOx1(double *fx1, double *fy1, double *fz1,
           double *fx0, double *fy0, double *fz0, int l, double *ri);
int GTOprim_exp(double *eprim, double *coord, double *alpha, double *coeff,
                int l, int nprim, int nctr, int blksize, double fac);

void GTOeval_sph_drv(void (*feval)(),  int (*fexp)(),
                     int param[], int *shls_slice, int *ao_loc, int ngrids,
                     double *ao, double *coord, char *non0table,
                     int *atm, int natm, int *bas, int nbas, double *env);

void GTOeval_cart_drv(void (*feval)(),  int (*fexp)(),
                      int param[], int *shls_slice, int *ao_loc, int ngrids,
                      double *ao, double *coord, char *non0table,
                      int *atm, int natm, int *bas, int nbas, double *env);

#define GTO_D_I(o, i, l) \
        GTOnabla1(fx##o, fy##o, fz##o, fx##i, fy##i, fz##i, l, alpha[k])
/* r-R_0, R_0 is (0,0,0) */
#define GTO_R0I(o, i, l) \
        GTOx1(fx##o, fy##o, fz##o, fx##i, fy##i, fz##i, l, ri)
/* r-R_C, R_C is common origin */
#define GTO_RCI(o, i, l) \
        GTOx1(fx##o, fy##o, fz##o, fx##i, fy##i, fz##i, l, dri)
/* origin from center of each basis
 * x1(fx1, fy1, fz1, fx0, fy0, fz0, l, 0, 0, 0) */
#define GTO_R_I(o, i, l) \
        fx##o = fx##i + 1; \
        fy##o = fy##i + 1; \
        fz##o = fz##i + 1
