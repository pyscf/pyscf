#define BeginTimeRevLoop(I, J) \
        for (I##0 = I##start; I##0 < I##end;) { \
                I##1 = abs(tao[I##0]); \
                for (J##0 = J##start; J##0 < J##end;) { \
                        J##1 = abs(tao[J##0]);
#define EndTimeRevLoop(I, J) \
                        J##0 = J##1; } \
                I##0 = I##1; }

void CVHFtimerev_map(int *tao, int *bas, int nbas);

void CVHFtimerev_block(double complex *block, double complex *mat, int *tao,
                       int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_blockT(double complex *block, double complex *mat, int *tao,
                        int istart, int iend, int jstart, int jend, int nao);

void CVHFtimerev_i(double complex *block, double complex *mat, int *tao,
                   int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_iT(double complex *block, double complex *mat, int *tao,
                    int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_j(double complex *block, double complex *mat, int *tao,
                   int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_jT(double complex *block, double complex *mat, int *tao,
                    int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_ijplus(double complex *block, double complex *mat, int *tao,
                        int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_ijminus(double complex *block, double complex *mat, int *tao,
                         int istart, int iend, int jstart, int jend, int nao);

void CVHFtimerev_adbak_block(double complex *block, double complex *mat, int *tao,
                             int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_adbak_blockT(double complex *block, double complex *mat, int *tao,
                              int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_adbak_i(double complex *block, double complex *mat, int *tao,
                         int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_adbak_iT(double complex *block, double complex *mat, int *tao,
                          int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_adbak_j(double complex *block, double complex *mat, int *tao,
                         int istart, int iend, int jstart, int jend, int nao);
void CVHFtimerev_adbak_jT(double complex *block, double complex *mat, int *tao,
                          int istart, int iend, int jstart, int jend, int nao);

void CVHFreblock_mat(double complex *a, double complex *b,
                     int *bloc, int nbloc, int nao);
void CVHFunblock_mat(double complex *a, double complex *b,
                     int *bloc, int nbloc, int nao);
