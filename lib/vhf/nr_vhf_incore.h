/*
 *
 */

void CVHFcompress_nr_dm(double *tri_dm, double *dm, int nao);

void CVHFnr_k(int n, double *eri, double *dm, double *vk);
void CVHFnr_incore_o3(int n, double *eri, double *dm, double *vj, double *vk);
void CVHFnr_eri8fold_vj_o2(double *tri_vj, const int ij,
                           const double *eri, const double *tri_dm);
void CVHFnr_eri8fold_vj_o3(double *tri_vj, const int ij,
                           const double *eri, const double *tri_dm);
void CVHFnr_eri8fold_vk_o0(double *vk, int i, int j, int n,
                           const double *eri, const double *tri_dm);
void CVHFnr_eri8fold_vk_o1(double *vk, int i, int j, int n,
                           const double *eri, const double *tri_dm);
void CVHFnr_eri8fold_vk_o2(double *vk, int i, int j, int n,
                           const double *eri, const double *tri_dm);
void CVHFnr_eri8fold_vk_o3(double *vk, int i, int j, int n,
                           const double *eri, const double *tri_dm);
void CVHFnr_eri8fold_vk_o4(double *vk, int i, int j, int n,
                           const double *eri, const double *tri_dm);
void CVHFnr_incore_o4(int n, double *eri, double *dm, double *vj, double *vk);

