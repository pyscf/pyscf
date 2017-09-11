/*
 *
 */
#include <stdint.h>
#define MAX_THREADS     256

typedef struct {
        unsigned int addr;
        unsigned short ia;
        signed char sign;
        signed char _padding;
} _LinkTrilT;

typedef struct {
        unsigned int addr;
        unsigned char a;
        unsigned char i;
        char sign;
        char _padding;
} _LinkT;

#define EXTRACT_A(I)    (I.a)
#define EXTRACT_I(I)    (I.i)
#define EXTRACT_SIGN(I) (I.sign)
#define EXTRACT_ADDR(I) (I.addr)
#define EXTRACT_IA(I)   (I.ia)

#define EXTRACT_CRE(I)  EXTRACT_A(I)
#define EXTRACT_DES(I)  EXTRACT_I(I)

void FCIcompress_link(_LinkT *clink, int *link_index,
                      int norb, int nstr, int nlink);
void FCIcompress_link_tril(_LinkTrilT *clink, int *link_index,
                           int nstr, int nlink);
int FCIcre_des_sign(int p, int q, uint64_t string0);
int FCIcre_sign(int p, uint64_t string0);
int FCIdes_sign(int p, uint64_t string0);
int FCIpopcount_1(uint64_t x);

void FCIprog_a_t1(double *ci0, double *t1,
                  int bcount, int stra_id, int strb_id,
                  int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);
void FCIprog_b_t1(double *ci0, double *t1,
                  int bcount, int stra_id, int strb_id,
                  int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);
void FCIspread_a_t1(double *ci0, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);
void FCIspread_b_t1(double *ci0, double *t1,
                    int bcount, int stra_id, int strb_id,
                    int norb, int nstrb, int nlinka, _LinkTrilT *clink_indexa);

double FCIrdm2_a_t1ci(double *ci0, double *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa);
double FCIrdm2_b_t1ci(double *ci0, double *t1,
                      int bcount, int stra_id, int strb_id,
                      int norb, int nstrb, int nlinka, _LinkT *clink_indexa);

void FCIaxpy2d(double *out, double *in, size_t count, size_t no, size_t ni);
