/*
 *
 */

void compress_dm(double *tri_dm, double *dm, unsigned int nao);
void set_ij2i(unsigned int *ij2i, unsigned int n);

void nr_vhf_k(int n, double *eri, double *dm, double *vk);
void nr_vhf_incore_o3(int n, double *eri, double *dm, double *vj, double *vk);
void nr_eri8fold_vj_o2(double *tri_vj, const unsigned int ij,
                       const double *eri, const double *tri_dm);
void nr_eri8fold_vj_o3(double *tri_vj, const unsigned int ij,
                       const double *eri, const double *tri_dm);
void nr_eri8fold_vk_o0(double *vk, int i, int j, int n,
                       const double *eri, const double *tri_dm);
void nr_eri8fold_vk_o1(double *vk, int i, int j, int n,
                       const double *eri, const double *tri_dm);
void nr_eri8fold_vk_o2(double *vk, int i, int j, int n,
                       const double *eri, const double *tri_dm);
void nr_eri8fold_vk_o3(double *vk, int i, int j, int n,
                       const double *eri, const double *tri_dm);
void nr_eri8fold_vk_o4(double *vk, int i, int j, int n,
                       const double *eri, const double *tri_dm);
void nr_vhf_incore_o4(int n, double *eri, double *dm, double *vj, double *vk);

