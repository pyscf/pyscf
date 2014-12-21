/*
 *
 */

/* japcv = apcv * 4
 *       - acpv.transpose(0,2,1,3)
 *       - avcp.transpose(0,3,2,1)
 */
// transform last three indices
void MCSCFinplace_apcv(double *japcv, double *ppp,
                       int ncore, int ncas, int nmo)
{
        int j, k, l;
        int d2 = nmo * nmo;
        int nvir = nmo - ncore;
        double *apcv;

        japcv -= ncore;
        for (j = 0; j < nmo; j++) {
                apcv = ppp + j * d2;
                for (k = 0; k < ncore; k++) {
                for (l = ncore; l < nmo; l++) {
                        japcv[k*nvir+l] = apcv[k*nmo+l] * 4
                                - ppp[j*nmo+k*d2+l] - ppp[j+k*nmo+l*d2];
                } }
                japcv += ncore * nvir;
        }
}

/* jcvcv += cvcv * 4
 *        - cvcv.transpose(0,3,2,1)
 *        - ccvv.transpose(0,2,1,3)
 */
void MCSCFinplace_cvcv(double *jcvcv, double *vcp, double *cpp,
                       int ncore, int ncas, int nmo)
{
        int j, k, l, n;
        int ncp = ncore * nmo;
        int nmo2 = nmo * nmo;
        double *pcp = vcp - ncore * ncp;

        for (n = 0, j = ncore; j < nmo; j++) {
                for (k = 0; k < ncore; k++) {
                for (l = ncore; l < nmo; l++, n++) {
                        jcvcv[n] += vcp[k*nmo+l] * 4 - cpp[j*nmo+k*nmo2+l]
                                  - pcp[l*ncp+k*nmo+j];
                } }
                vcp += ncp;
        }
}


/* japcv = apcv * 2
 *       - acpv.transpose(0,2,1,3)
 *       - avcp.transpose(0,3,2,1)
 */
void MCSCFinplace_apcv_uhf(double *japcv, double *ppp,
                           int ncore, int ncas, int nmo)
{
        int j, k, l;
        int d2 = nmo * nmo;
        int nvir = nmo - ncore;
        double *apcv;

        japcv -= ncore;
        for (j = 0; j < nmo; j++) {
                apcv = ppp + j * d2;
                for (k = 0; k < ncore; k++) {
                for (l = ncore; l < nmo; l++) {
                        japcv[k*nvir+l] = apcv[k*nmo+l] * 2
                                - ppp[j*nmo+k*d2+l] - ppp[j+k*nmo+l*d2];
                } }
                japcv += ncore * nvir;
        }
}

/* jcvcv += cvcv * 2
 *        - cvcv.transpose(0,3,2,1)
 *        - ccvv.transpose(0,2,1,3)
 */
void MCSCFinplace_cvcv_uhf(double *jcvcv, double *vcp, double *cpp,
                           int ncore, int ncas, int nmo)
{
        int j, k, l, n;
        int ncp = ncore * nmo;
        int nmo2 = nmo * nmo;
        double *pcp = vcp - ncore * ncp;

        for (n = 0, j = ncore; j < nmo; j++) {
                for (k = 0; k < ncore; k++) {
                for (l = ncore; l < nmo; l++, n++) {
                        jcvcv[n] += vcp[k*nmo+l] * 2 - cpp[j*nmo+k*nmo2+l]
                                  - pcp[l*ncp+k*nmo+j];
                } }
                vcp += ncp;
        }
}

