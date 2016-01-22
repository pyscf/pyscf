/*
 *
 */

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
