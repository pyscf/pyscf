c copied from https://github.com/nwchemgit/nwchem/blob/master/src/solvation/mnsol.F
c MN solvation models -->
C
C THIS IS A PROGRAM TO CALCULATE THE CDS TERM FOR MINNESOTA SOLVATION MODEL SMD (MARENICH, CRAMER, TRUHLAR)
C JUN-14-2010
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C
C    |__________________:__ CDSSET
C    |                  :     |
C    |                  :     |______ CDSCALC
C    |                  :                |
C    |                  :                |______ SMD_CDS_AQ
C    |                  :                |
C    |                  :                |______ SMD_CDS_NAQ
C    |                  :                |
C    |                  :                |______ VDWRAD
C    |                  :                |
C    |                  :                |______ CDS_EG
C    |                  :                          |_____ SMXCDS
C    |                  :                          |         |_____ IJ0
C    |                  :                          |
C    |                  :                          |_____ DAREAL
C    |                  :                                    |_____ SCOPYMN
C    |                  :                                    |_____ IJ0
C    |                  :                                    |_____ DSCALMN
C    |                  :                                    |_____ DOTMN
C    |                  :                                    |        |_____ DDOTMN
C    |                  :                                    |
C    |                  :                                    |_____ DSWAPMN
C    |                  :
C    |                  :
C
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C  * CDSSET
C CDS TERMS FOR MINNESOTA SOLVATION MODELS
       SUBROUTINE CDSSET(ICDS,GCDS,AREACDS,NAT,C,IAN,DCDS,X,
     &   SOLA,SOLB,SOLC,SOLG,SOLH,SOLN)
C
C     USE THE FOLLOWING VALUES OF ICDS:
C     ICDS=1 FOR WATER
C     ICDS=2 FOR ANY NONAQUEOUS SOLVENT
C     IF ICDS=2 YOU NEED TO PROVIDE THE FOLLOWING SOLVENT DESCRIPTORS (see http://comp.chem.umn.edu/solvation/mnsddb.pdf):
C       SOLA (Abraham's hydrogen-bond acidity parameter),
C       SOLB (Abraham's hydrogen-bond basicity parameter),
C       SOLC (carbon aromaticity),
C       SOLG (macroscopic surface tension in cal/(mol A^2),
C       SOLH (electronegative halogenicity),
C       SOLN (refractive index)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C     CAVITY DISPERSION SOLVENT STRUCTURE (CDS) DRIVER
      CALL CDSCALC(C,DCDS,IAN,ICDS,NAT,GCDS,AREACDS,X,
     &  SOLA,SOLB,SOLC,SOLG,SOLH,SOLN)
C
      RETURN
      END
C
C  * CDSCALC
      SUBROUTINE CDSCALC(C,DCDS,IAN,ICDS,NAT,GCDS,AREACDS,X,
     &  SOLA,SOLB,SOLC,SOLG,SOLH,SOLN)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      LOGICAL LGR
C
      DIMENSION C(3,*),DCDS(3,*),IAN(*),X(*)
      DIMENSION SIGMA(150),HSIGMA(150)
C
      GCDS=0.D0
      AREACDS=0.D0
C
C DO WE NEED CDS GRADIENTS? LGR=.FALSE. OR LGR=.TRUE.
C      LGR=.FALSE.
      LGR=.TRUE.
C
C FINDING THE CDS PARAMETERS
C
      DO I=1,150
      SIGMA(I)=0.D0
      HSIGMA(I)=0.D0
      ENDDO
      CSSIGM=0.D0
C
      IF (ICDS.EQ.1) CALL SMD_CDS_AQ(SIGMA,HSIGMA,CSSIGM)
      IF (ICDS.EQ.2) CALL SMD_CDS_NAQ(SIGMA,HSIGMA,CSSIGM,
     &    SOLA,SOLB,SOLC,SOLG,SOLH,SOLN)
      IF (ICDS.NE.1.AND.ICDS.NE.2.AND.ICDS.NE.3) RETURN
C
C MEMORY ALLOCATION

C
C     CALL SOME EXTERNAL FUNCTION HERE TO DEFINE THE BEGINNING OF X(*) AS LXMEM !!!!
C
      LXMEM = 0 ! set LXMEM = 0 for now XXX FIX ME XXXX
      LRAD = LXMEM + 1
      LCDSA = LRAD + NAT
      LAREA = LCDSA + NAT
      LDATAR = LAREA + NAT
      LSTS = LDATAR + 3 * NAT * NAT
      LCOT = LSTS + NAT
      LDSTS = LCOT + NAT * (NAT + 1) / 2
      LDCOTDR = LDSTS + 3 * NAT * NAT
      LNC = LDCOTDR + NAT * (NAT + 1) / 2
      LDAREA = LNC + NAT + 1
      LRLIO = LDAREA + 3 * (NAT + 1)
      LURLIO = LRLIO + NAT * (NAT + 1) / 2
      LLAB = LURLIO + 3 * NAT * NAT
      LNCNCT = LLAB + NAT
      LCONECT = LNCNCT + NAT * (2 * NAT + 1)
      LCTHETA = LCONECT + NAT * NAT
      LSTHETA = LCTHETA + NAT * NAT
      LSIT = LSTHETA + NAT
      LDCSIT = LSIT + NAT
      LDJCOSN = LDCSIT + 3 * NAT
      LCOSN = LDJCOSN + 3 * 3 * NAT * NAT
      LDSTETA = LCOSN + 3 * NAT * NAT
      LDCTETA = LDSTETA + 3 * NAT
      LDCOSN = LDCTETA + 3 * NAT * NAT
      LWORK = LDCOSN + 3 * 3 * NAT * NAT
      LDIWORK = LWORK + 2 * NAT + 1
      LDJWORK = LDIWORK + 3 * NAT
      LDKWORK = LDJWORK + 3 * NAT
      LD0WORK = LDKWORK + 3 * NAT
      LDWORKR = LD0WORK + 3 * NAT
      LDCAODD = LDWORKR + NAT
      LDSITR = LDCAODD + 3 * (NAT + 1)
      LDCAPLY = LDSITR + NAT
      LDICOSN = LDCAPLY + 3 * (NAT + 1)
      LDCASLC = LDICOSN + 3 * 3 * NAT * NAT
      LDCTETR = LDCASLC + 3 * (NAT + 1)
      LDSTETR = LDCTETR + NAT
      LDCOSNR = LDSTETR + NAT
      LEND = LDCOSNR + 3 * NAT * NAT
      INEED = LEND - LRAD
C
C     YOU NEED TO CHECK IF YOU HAVE INEED WORDS OF MEMORY BY CALLING SOME EXTERNAL FUNCTION
C
C
C VAN DER WAALS RADII FOR SASA: Bondi's radii + 0.4Angs
C
      SOLVRD=0.4D0
      CALL VDWRAD(X(LRAD),IAN,NAT,SOLVRD)
C
C CALCULATION OF CDS CONTRIBUTIONS TO ENERGY AND GRADIENT
C
      CALL CDS_EG(CSSIGM,CDST,TAREA,NAT,LGR,
     $C,DCDS,IAN,SIGMA,HSIGMA,X(LRAD),
     $X(LCDSA),X(LAREA),X(LDATAR),X(LSTS),X(LCOT),X(LDSTS),
     $X(LDCOTDR),X(LNC),X(LDAREA),X(LRLIO),X(LURLIO),
     $X(LLAB),X(LNCNCT),X(LCONECT),X(LCTHETA),X(LSTHETA),X(LSIT),
     $X(LDCSIT),X(LDJCOSN),X(LCOSN),X(LDSTETA),X(LDCTETA),X(LDCOSN),
     . X(LWORK),X(LDIWORK),X(LDJWORK),
     $X(LDKWORK),X(LD0WORK),X(LDWORKR),X(LDCAODD),X(LDSITR),
     . X(LDCAPLY),X(LDICOSN),X(LDCASLC),
     $X(LDCTETR),X(LDSTETR),X(LDCOSNR))
C
C MEMORY RELEASE
C     CALL SOME EXTERNAL FUNCTION TO RELEASE INEED WORDS OF MEMORY USED FOR X(*)
C
      GCDS=CDST
      AREACDS=TAREA
C
C     GCDS IS A NONELECTROSTATIC CONTRIBUTION IN KCAL/MOL TO THE TOTAL ENERGY IN SOLUTION, IT MUST BE ADDED TO THE TOTAL ENERGY
C     AREACDS IS THE TOTAL SURFACE AREA (FOR DEBUGGING PURPOSES ONLY)
C
      RETURN
      END
C
C  * SMD_CDS_AQ
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C    SMD_CDS_AQ CALCULATES SURFACE TENSION COEFFICIENTS
C    FOR AQUEOUS SOLUTION FOR SMD
C    ATOMIC SURFACE TENSION COEFFICIENTS ARE OF 3 TYPES:
C      1.) Zero order surface tension coefficients,
C        SIGMA(I): atomic surface tension of atom with nuclear number I<101
C      2.) First order modification for H,
C         HSIGMA(I): H bonded to a heavy atom with nuclear number I
C      3.) Other modifications,
C         SIGMA(I+100): modification for atom X bonded to Y.
C         I is the atomic pair index shown as following:
C              1:     C-C (1)
C              2:     C-C (2)
C              3:     O-C
C              4:     O-O
C              5:     N-C (1)
C              6:     O-N
C              7:     S-S
C              8:     ...
C              9:     ...
C             10:     C-N
C             11:     N-C (2)
C             12:     H-(NN)
C             13:     H-(OH)
C             14:     O-P
C             15:     S-P
C             16:     N-C (triple bonded to C)
C             17:     O-Si
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SMD_CDS_AQ(SIGMA,HSIGMA,CSSIGM)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
      DIMENSION SIGMA(150),HSIGMA(150)
      DIMENSION SIGMA_DATA(150),HSIGMA_DATA(150)
C
      DATA SIGMA_DATA/
     .    48.69,     0.00,     0.00,     0.00,     0.00,
     .   129.74,     0.00,     0.00,    38.18,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .    -9.10,     9.82,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,    -8.72,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .   -72.95,     0.00,    68.69,     0.00,   -48.22,
     .   121.98,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,    68.85,     0.00,
     .    84.10,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00/
      DATA HSIGMA_DATA/
     . 5*0.00, -60.77, 0.00, 0.00,
     $ 7*0.00, 0.00,134*0.00/
C
      CSSIGM=0.D0
C
      DO I=1,150
      SIGMA(I)=SIGMA_DATA(I)
      HSIGMA(I)=HSIGMA_DATA(I)
      ENDDO
C
      RETURN
      END
C
C  * SMD_CDS_NAQ
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C    SMD_CDS_NAQ CALCULATES SURFACE TENSION COEFFICIENTS
C    FOR NONAQUEOUS SOLUTION FOR SMD
C    ATOMIC SURFACE TENSION COEFFICIENTS ARE OF 3 TYPES:
C      1.) Zero order surface tension coefficients,
C        SIGMA_x(I): atomic surface tension of atom with nuclear number I<101
C        where if
C        x = N is multiplied by SolN
C        x = A is multiplied by SolA
C        x = B is multiplied by SolB
C      2.) First order modification for H,
C         HSIGMA_x(I): H bonded to a heavy atom with nuclear number I
C        where if
C        x = N is multiplied by SolN
C        x = A is multiplied by SolA
C        x = B is multiplied by SolB
C      3.) Other modifications,
C         SIGMA_x(I+100): modification for atom X bonded to Y.
C         I is the atomic pair index shown as following:
C              1:     C-C (1)
C              2:     C-C (2)
C              3:     O-C
C              4:     O-O
C              5:     N-C (1)
C              6:     O-N
C              7:     S-S
C              8:     ...
C              9:     ...
C             10:     C-N
C             11:     N-C (2)
C             12:     H-(NN)
C             13:     H-(OH)
C             14:     O-P
C             15:     S-P
C             16:     N-C (triple bonded to C)
C             17:     O-Si
C        where if
C        x = N is multiplied by SolN
C        x = A is multiplied by SolA
C        x = B is multiplied by SolB
C     MOLECULAR SURFACE TENSION COEFFICIENTS ARE STORED IN THE DATA
C     ARRAY SIGMA_MOL(I), where
C       element 1 is multiplied by SolG
C       element 2 is multiplied by (SolB)**2
C       element 3 is multiplied by (SolC)**2
C       element 4 is multiplied by (SolH)**2
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SMD_CDS_NAQ(SIGMA,HSIGMA,CSSIGM,
     &    SOLA,SOLB,SOLC,SOLG,SOLH,SOLN)
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C     YOU NEED TO PROVIDE THE FOLLOWING SOLVENT DESCRIPTORS (see http://comp.chem.umn.edu/solvation/mnsddb.pdf):
C       SOLA (Abraham's hydrogen-bond acidity parameter),
C       SOLB (Abraham's hydrogen-bond basicity parameter),
C       SOLC (carbon aromaticity),
C       SOLG (macroscopic surface tension in cal/(mol A^2),
C       SOLH (electronegative halogenicity),
C       SOLN (refractive index)
      DIMENSION SIGMA(150),HSIGMA(150)
      DIMENSION SIGMA_N(150),SIGMA_A(150),SIGMA_B(150)
      DIMENSION HSIGMA_N(150),HSIGMA_A(150),HSIGMA_B(150)
      DIMENSION SIGMA_MOL(4)
      DATA SIGMA_N/
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .    58.10,    32.62,   -17.56,     0.00,     0.00,
     .     0.00,     0.00,     0.00,   -18.04,     0.00,
     .   -33.17,   -24.31,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,   -35.42,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .   -62.05,     0.00,   -15.70,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,   -99.76,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00/
      DATA SIGMA_A/
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .    48.10,     0.00,   193.06,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,    95.99,     0.00,   -41.00,
     .     0.00,     0.00,     0.00,     0.00,   152.20,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00/
      DATA SIGMA_B/
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .    32.87,     0.00,   -43.79,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,  -128.16,     0.00,
     .    79.13,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00,
     .     0.00,     0.00,     0.00,     0.00,     0.00/
      DATA HSIGMA_N/
     . 5*0.00,-36.37,  0.00,-19.39,7*0.00,0.00,134*0.00/
      DATA HSIGMA_A/
     . 150*0.00/
      DATA HSIGMA_B/
     . 150*0.00/
      DATA SIGMA_MOL/
     . 0.35, 0.00, -4.19, -6.68/
C     Zero order atomic surface tensions SIGMA_x(1)-SIGMA_x(100)
C     First and higher order modification SIGMA_x(101)-SIGMA_x(150)
C
      DO I=1,150
      SIGMA(I)=SIGMA_N(I)*SOLN+SIGMA_A(I)*SOLA+SIGMA_B(I)*SOLB
      HSIGMA(I)=HSIGMA_N(I)*SOLN+HSIGMA_A(I)*SOLA+HSIGMA_B(I)*SOLB
      END DO
C
      CSSIGM=SIGMA_MOL(1)*SOLG+SIGMA_MOL(2)*SOLB*SOLB+
     $ SIGMA_MOL(3)*SOLC*SOLC+SIGMA_MOL(4)*SOLH*SOLH
C
      RETURN
      END
C
C  * VDWRAD
C  VAN DER WAALS RADII FOR CDS TERM
      SUBROUTINE VDWRAD(RAD,IAN,NAT,SOLVRD)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION RAD(*),IAN(*)
      DIMENSION BONDI(102)
C
C  SOURCE (except H):
C  Van der Waals radii of Mantina et al.
C  (Mantina, M.; Valero, R.; Cramer, C. J.; Truhlar, D. G.
C  "Atomic Radii of the Elements." In CRC Handbook of Chemistry
C   and Physics, 91st Edition, 2010-2011; Haynes, W. M., Ed.;
C   CRC Press: Boca Raton, FL, 2010; pp 9-49 -- 9-50)
C
C  SOURCE (H):
C  Bondi's radii (Bondi, A. J. Phys. Chem. 1964, 68, 441)
C
      DATA BONDI/
     $ 1.20,         ! H
     $ 1.40,         ! He
     $ 1.82,         ! Li
     $ 1.53,         ! Be
     $ 1.92,         ! B
     $ 1.70,         ! C
     $ 1.55,         ! N
     $ 1.52,         ! O
     $ 1.47,         ! F
     $ 1.54,         ! Ne
     $ 2.27,         ! Na
     $ 1.73,         ! Mg
     $ 1.84,         ! Al
     $ 2.10,         ! Si
     $ 1.80,         ! P
     $ 1.80,         ! S
     $ 1.75,         ! Cl
     $ 1.88,         ! Ar
     $ 2.75,         ! K
     $ 2.31,         ! Ca
     $ 2.16,         ! Sc
     $ 1.87,         ! Ti
     $ 1.79,         ! V
     $ 1.89,         ! Cr
     $ 1.97,         ! Mn
     $ 1.94,         ! Fe
     $ 1.92,         ! Co
     $ 1.84,         ! Ni
     $ 1.86,         ! Cu
     $ 2.10,         ! Zn
     $ 1.87,         ! Ga
     $ 2.11,         ! Ge
     $ 1.85,         ! As
     $ 1.90,         ! Se
     $ 1.85,         ! Br
     $ 2.02,         ! Kr
     $ 3.03,         ! Rb
     $ 2.49,         ! Sr
     $ 2.19,         ! Y
     $ 1.86,         ! Zr
     $ 2.07,         ! Nb
     $ 2.09,         ! Mo
     $ 2.09,         ! Tc
     $ 2.07,         ! Ru
     $ 1.95,         ! Rh
     $ 2.02,         ! Pd
     $ 2.03,         ! Ag
     $ 2.30,         ! Cd
     $ 1.93,         ! In
     $ 2.17,         ! Sn
     $ 2.06,         ! Sb
     $ 2.06,         ! Te
     $ 1.98,         ! I
     $ 2.16,         ! Xe
     $ 3.43,         ! Cs
     $ 2.68,         ! Ba
     $ 2.40,         ! La
     $ 2.35,         ! Ce
     $ 2.39,         ! Pr
     $ 2.29,         ! Nd
     $ 2.36,         ! Pm
     $ 2.29,         ! Sm
     $ 2.33,         ! Eu
     $ 2.37,         ! Gd
     $ 2.21,         ! Tb
     $ 2.29,         ! Dy
     $ 2.16,         ! Ho
     $ 2.35,         ! Er
     $ 2.27,         ! Tm
     $ 2.42,         ! Yb
     $ 2.21,         ! Lu
     $ 2.12,         ! Hf
     $ 2.17,         ! Ta
     $ 2.10,         ! W
     $ 2.17,         ! Re
     $ 2.16,         ! Os
     $ 2.02,         ! Ir
     $ 2.09,         ! Pt
     $ 2.17,         ! Au
     $ 2.09,         ! Hg
     $ 1.96,         ! Tl
     $ 2.02,         ! Pb
     $ 2.07,         ! Bi
     $ 1.97,         ! Po
     $ 2.02,         ! At
     $ 2.20,         ! Rn
     $ 3.48,         ! Fr
     $ 2.83,         ! Ra
     $ 2.60,         ! Ac
     $ 2.37,         ! Th
     $ 2.43,         ! Pa
     $ 2.40,         ! U
     $ 2.21,         ! Np
     $ 2.43,         ! Pu
     $ 2.44,         ! Am
     $ 2.45,         ! Cm
     $ 2.44,         ! Bk
     $ 2.45,         ! Cf
     $ 2.45,         ! Es
     $ 2.45,         ! Fm
     $ 2.46,         ! Md
     $ 2.46/         ! No
C
      DO I=1,NAT
      RAD(I)=0.D0
        DO K=1,102
          IF (IAN(I).EQ.K) THEN
          RAD(I)=BONDI(K)+SOLVRD
          ENDIF
        ENDDO
      ENDDO
      RETURN
      END
C
C   * CDS_EG
      SUBROUTINE CDS_EG(CSSIGM,CDST,TAREA,NAT,LGR,
     $C,DCDS,IAN,SIGMA,HSIGMA,RAD,
     $CDSA,AREA,DATAR,STS,COT,DSTS,DCOTDR,
     $NC,DAREA,RLIO,URLIO,
     $LAB,NCNCT,CONECT,CTHETA,STHETA,SIT,
     $DCSIT,DJCOSN,COSN,DSTETA,DCTETA,DCOSN,WORK,DIWORK,DJWORK,
     $DKWORK,D0WORK,DWORKR,DCAODD,DSITR,DCAPLY,DICOSN,DCASLC,
     $DCTETR,DSTETR,DCOSNR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (TOANGS=0.52917724924D+00,TOKCAL=627.509451D+00)
      LOGICAL LGR,CONECT
      DIMENSION C(3,*),DCDS(3,*),IAN(*)
      DIMENSION SIGMA(150),HSIGMA(150)
C
      DIMENSION CDSA(*),AREA(*)
      DIMENSION DATAR(3,NAT,*)
C
      DIMENSION STS(*),COT(*)
      DIMENSION DSTS(3,NAT,*),DCOTDR(*)
C
      DIMENSION RAD(*),NC(0:*),DAREA(3,0:*)
      DIMENSION RLIO(*),URLIO(3,NAT,*)
      DIMENSION LAB(*),NCNCT(2*NAT+1,*)
     1               ,CONECT(NAT,*)
      DIMENSION CTHETA(NAT,*),STHETA(*),SIT(*)
     1               ,DCSIT(3,*),DJCOSN(3,3,NAT,*)
     2               ,COSN(3,NAT,*),DSTETA(3,*)
     3               ,DCTETA(3,NAT,*),DCOSN(3,3,NAT,*)
     4               ,WORK(*),DIWORK(3,*),DJWORK(3,*)
     5               ,DKWORK(3,*),D0WORK(3,*),DWORKR(*)
     6               ,DCAODD(3,0:*),DSITR(*)
     6               ,DCAPLY(3,0:*)
     7               ,DICOSN(3,3,NAT,*),DCASLC(3,0:*)
     8               ,DCTETR(*),DSTETR(*)
     9               ,DCOSNR(3,NAT,*)
C
      DO I=1,NAT
      CDSA(I)=0.D0
      AREA(I)=0.D0
      STS(I)=0.D0
      DO J=1,NAT
      DO K=1,3
      DATAR(K,I,J)=0.D0
      ENDDO
      ENDDO
      ENDDO
C
C MATRIX OF INTERATOMIC DISTANCES IN ANGSTROMS
C
      K12=0
      DO K1=1,NAT
      DO K2=1,K1
      K12=K12+1
      IF (K1.EQ.K2) THEN
      RLIO(K12)=0.D0
      URLIO(1,K1,K2)=0.D0
      URLIO(2,K1,K2)=0.D0
      URLIO(3,K1,K2)=0.D0
                   ELSE
      X=(C(1,K1)-C(1,K2))*TOANGS
      Y=(C(2,K1)-C(2,K2))*TOANGS
      Z=(C(3,K1)-C(3,K2))*TOANGS
      R=DSQRT(X*X+Y*Y+Z*Z)
      URLIO(1,K1,K2)=-X/R
      URLIO(2,K1,K2)=-Y/R
      URLIO(3,K1,K2)=-Z/R
      URLIO(1,K2,K1)=X/R
      URLIO(2,K2,K1)=Y/R
      URLIO(3,K2,K1)=Z/R
      RLIO(K12)=R
      ENDIF
      ENDDO
      ENDDO
C
C CALCULATION OF CDS CONTRIBUTIONS TO ENERGY
C
      CALL SMXCDS
     $(IAN,SIGMA,HSIGMA,STS,COT,DSTS,DCOTDR,RLIO,URLIO,
     $NAT,LGR)
C
      TAREA=0.D0
      CDST=0.D0
      DO K=1,NAT
      AREA0=0.D0
      NCROSS=0
       CALL DAREAL
     $(NAT,AREA0,NCROSS,K,LGR,
     $RAD,NC,DAREA,RLIO,URLIO,LAB,NCNCT,CONECT,CTHETA,STHETA,SIT,
     $DCSIT,DJCOSN,COSN,DSTETA,DCTETA,DCOSN,WORK,DIWORK,DJWORK,
     $DKWORK,D0WORK,DWORKR,DCAODD,DSITR,DCAPLY,DICOSN,DCASLC,
     $DCTETR,DSTETR,DCOSNR)
       RAS2=RAD(K)*RAD(K)
       AREA(K)=AREA0*RAS2
       CDST=CDST+AREA(K)*(STS(K)+CSSIGM)*0.001D0
       TAREA=TAREA+AREA(K)
       CDSA(K)=AREA(K)*(STS(K)+CSSIGM)*0.001D0
       IF (LGR) THEN
        DO L=0,NCROSS
        DATAR(1,NC(L),K)=DAREA(1,L)*RAS2
        DATAR(2,NC(L),K)=DAREA(2,L)*RAS2
        DATAR(3,NC(L),K)=DAREA(3,L)*RAS2
        ENDDO
       ENDIF
      ENDDO
C     ----- COMPUTATION OF CDS GRADIENTS -----
C
      IF (LGR) THEN
       DO IAT=1,NAT
       DCDSX=0.D0
       DCDSY=0.D0
       DCDSZ=0.D0
         DO J=1,NAT
         DCDSX=DCDSX
     $    +(STS(J)+CSSIGM)*DATAR(1,IAT,J)*0.001D0
     $    +DSTS(1,IAT,J)*AREA(J)*0.001D0
         DCDSY=DCDSY
     $    +(STS(J)+CSSIGM)*DATAR(2,IAT,J)*0.001D0
     $    +DSTS(2,IAT,J)*AREA(J)*0.001D0
         DCDSZ=DCDSZ
     $    +(STS(J)+CSSIGM)*DATAR(3,IAT,J)*0.001D0
     $    +DSTS(3,IAT,J)*AREA(J)*0.001D0
         ENDDO
       DCDSX=DCDSX/TOKCAL*TOANGS
       DCDSY=DCDSY/TOKCAL*TOANGS
       DCDSZ=DCDSZ/TOKCAL*TOANGS
       DCDS(1,IAT)=DCDSX
       DCDS(2,IAT)=DCDSY
       DCDS(3,IAT)=DCDSZ
      ENDDO
      ENDIF
C
      RETURN
      END
C
C   * SMXCDS
      SUBROUTINE SMXCDS
     $(IAN,SIGMA,HSIGMA,STS,COT,DSTS,DCOTDR,RLIO,URLIO,
     $NAT,LGR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      LOGICAL LGR
      DIMENSION IAN(*)
      DIMENSION SIGMA(150),HSIGMA(150)
      DIMENSION NATCNV(100),RKKVAL(15,15)
      DIMENSION STS(*),COT(*)
      DIMENSION DSTS(3,NAT,*),DCOTDR(*),
     $RLIO(*),URLIO(3,NAT,*)
C                 H     C N O F     Si P S Cl     Br     I
      DATA NATCNV/1,4*0,2,3,4,5,4*0,11,9,6,7,17*0,8,17*0,10,47*0/
C
C --- RKKVAL EXPANDED TO 15 X 15 DATA ARRAY ---
C
C ---  TYPE X REFERS TO :                   ---
C
C    Hydrogen - type 1
C    Carbon   - type 2
C    Nitrogen - type 3
C    Oxygen   - type 4
C    Fluorine - type 5
C    Sulfur   - type 6
C    Chlorine - type 7
C    Bromine  - type 8
C    Phos.    - type 9
C    Iodine   - type 10
C    Silicon  - type 11
C
C
      DATA ((RKKVAL(J,I), I=1,15), J=1,15)
C
C --- J => TYPES DEFINED ABOVE    ---
C --- I => NATCNV ARRAY  ELEMENTS ---
C
C  Set for type 1
     .           /0.00D0, 1.55D0, 1.55D0, 1.55D0, 0.00D0,
     .            2.14D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 2
     .            1.55D0, 1.84D0, 1.84D0, 1.84D0, 1.84D0,
     .            2.20D0, 2.10D0, 2.30D0, 2.20D0, 2.60D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 3
     .            1.55D0, 1.84D0, 1.85D0, 1.50D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 4
     .            1.55D0, 1.84D0, 1.50D0, 2.75D0, 0.00D0,
     .            1.71D0, 0.00D0, 0.00D0, 2.10D0, 0.00D0,
     .            2.10D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 5
     .            0.00D0, 1.84D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 6
     .            2.14D0, 2.20D0, 0.00D0, 1.71D0, 0.00D0,
     .            2.75D0, 0.00D0, 0.00D0, 2.50D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 7
     .            0.00D0, 2.10D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 8
     .            0.00D0, 2.30D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 9
     .            0.00D0, 2.20D0, 0.00D0, 2.10D0, 0.00D0,
     .            2.50D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 10
     .            0.00D0, 2.60D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 11
     .            0.00D0, 0.00D0, 0.00D0, 2.10D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 12
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 13
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 14
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
C  Set for type 15
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0,
     .            0.00D0, 0.00D0, 0.00D0, 0.00D0, 0.00D0/
C
      ISTS=6
C
       DO I=1, NAT*(NAT+1)/2
       DCOTDR(I)=0.D0
       COT(I)=0.D0
       ENDDO
       DO I1=1,NAT
       DO I2=1,NAT
       DO I3=1,3
       DSTS(I3,I1,I2)=0.D0
       ENDDO
       ENDDO
       ENDDO
C
C-----COMPUTE COTS AND DERIVATIVES
C
      RHLD=1.4d0
      DELTAR=0.2d0
      BCCC=1.0d0
      EXPVAL=DEXP(1.00D0)
C
      DO 100 I=1,NAT
        II=IAN(I)
        ITPC=NATCNV(II)
        DO 100 J=1,NAT
        IF (I.NE.J) THEN
          JJ=IAN(J)
          JTPC=NATCNV(JJ)
          IF(ITPC.EQ.0.OR.JTPC.EQ.0) THEN
            COT(IJ0(I,J))=0.0D0
          ELSE
            COT(IJ0(I,J))=0.0D0
            DELTAR=0.30D0
            RHLD=RKKVAL(ITPC,JTPC)
            IF((II.EQ.8.and.JJ.EQ.6).OR.(II.EQ.6.and.JJ.EQ.8)) THEN
              DELTAR=0.10D0
              RHLD=1.330D0
            END IF
            IF (ISTS.EQ.6) THEN
              IF(II.EQ.8.and.JJ.EQ.8) THEN
                DELTAR=0.30D0
                RHLD=1.80D0
              END IF
            END IF
            CUTOF1=RHLD+DELTAR
            IF(RLIO(IJ0(I,J)).LT.CUTOF1) THEN
             EXPONT=DELTAR/(RLIO(IJ0(I,J))-CUTOF1)
             COT(IJ0(I,J))=DEXP(EXPONT)
C
              if (LGR)
     $DCOTDR(IJ0(I,J))=-COT(IJ0(I,J))*EXPONT/(RLIO(IJ0(I,J))-CUTOF1)
            END IF
          END IF
        ENDIF
 100    CONTINUE
C
C-----CDS SIGMAS CAN BE ANY OF THESE FORMS: a, a-b, a-b(2), a-b(3) (a AND b - ARE ELEMENTS)
C-----SOME OF THE EXPRESSIONS BELOW ARE NOT USED BY THE SMD MODEL BUT KEPT HERE FOR DEBUGGING PURPOSES
C
      DO 950 I=1,NAT
C
C-----ADD CDS FOR SIGMA a
        STS(I)=SIGMA(IAN(I))
C
        IF (IAN(I).EQ.1) THEN
C
C-----ADD CDS FOR SIGMA H-a (USED BY SMD ONLY FOR H-C, H-O)
          DO 150 J=1,NAT
            NTP=IAN(J)
            STS(I)=STS(I)+COT(IJ0(I,J))*HSIGMA(NTP)
C-----ADD DCDS FOR SIGMA H-a
            if (LGR) then
              do L=1,3
                DSTS(L,I,I)=DSTS(L,I,I)
     .              +HSIGMA(NTP)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                DSTS(L,J,I)=DSTS(L,J,I)
     .              -HSIGMA(NTP)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
              end do
            end if
C
C-----ADD CDS FOR SIGMA H-N(2); FOR H-N-N BOND (NOT USED BY SMD)
            IF (NTP.EQ.7) THEN
              DO 262 K=1,NAT
                KTP=IAN(K)
                IF (J.NE.K.AND.KTP.EQ.7) THEN
                 STS(I)=STS(I)+COT(IJ0(I,J))*COT(IJ0(J,K))*SIGMA(112)
C-----ADD DCDS FOR SIGMA H-N(2); FOR H-N-N BOND (NOT USED BY SMD)
                  if (LGR) then
                    do L=1,3
                      DSTS(L,I,I)=DSTS(L,I,I)
     .                           +SIGMA(112)*COT(ij0(J,K))
     .                           *DCOTDR(ij0(I,J))*URLIO(L,J,I)
                      DSTS(L,J,I)=DSTS(L,J,I)
     .                           -SIGMA(112)*COT(ij0(J,K))
     .                           *DCOTDR(ij0(I,J))*URLIO(L,J,I)
     .                           +SIGMA(112)*COT(ij0(I,J))
     .                           *DCOTDR(ij0(J,K))*URLIO(L,K,J)
                      DSTS(L,K,I)=DSTS(L,K,I)
     .                           -SIGMA(112)*COT(ij0(I,J))
     .                           *DCOTDR(ij0(J,K))*URLIO(L,K,J)
                    end do
                  end if
                ENDIF
 262          CONTINUE
C
C-----ADD CDS FOR SIGMA H-O(2); FOR H-O-H BOND (NOT USED BY SMD)
            ELSE IF (NTP.EQ.8) THEN
              DO 263 K=1,NAT
                KTP=IAN(K)
                IF (I.NE.K.AND.KTP.EQ.1) THEN
                  STS(I)=STS(I)+COT(IJ0(I,J))*COT(IJ0(J,K))*SIGMA(113)
C-----ADD DCDS FOR SIGMA H-O(2); FOR H-O-H BOND (NOT USED BY SMD)
                  if (LGR) then
                    do L=1,3
                      DSTS(L,I,I)=DSTS(L,I,I)
     .                           +SIGMA(113)*COT(ij0(J,K))
     .                           *DCOTDR(ij0(I,J))*URLIO(L,J,I)
                      DSTS(L,J,I)=DSTS(L,J,I)
     .                           -SIGMA(113)*COT(ij0(J,K))
     .                           *DCOTDR(ij0(I,J))*URLIO(L,J,I)
     .                           +SIGMA(113)*COT(ij0(I,J))
     .                           *DCOTDR(ij0(J,K))*URLIO(L,K,J)
                      DSTS(L,K,I)=DSTS(L,K,I)
     .                           -SIGMA(113)*COT(ij0(I,J))
     .                           *DCOTDR(ij0(J,K))*URLIO(L,K,J)
                    end do
                  end if
                ENDIF
 263          CONTINUE
            ENDIF
 150      CONTINUE
        ENDIF
C
        IF (IAN(I).EQ.8) THEN
          RTKK=0.0D0
          KOO=0
          DO 200 J=1,NAT
            NTP=IAN(J)
C
C-----ADD CDS FOR SIGMA O-C
            IF(NTP.EQ.6)THEN
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(103)
C-----ADD DCDS FOR SIGMA O-C
              if (LGR) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(103)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(103)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
C
C-----ADD CDS FOR SIGMA O-N
            ELSE IF(NTP.EQ.7) THEN
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(106)
C-----ADD DCDS FOR SIGMA O-N
              if (LGR) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(106)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(106)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
C
C-----ADD CDS FOR SIGMA O-S (NOT USED BY SMD)
            ELSE IF(NTP.EQ.16) THEN
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(118)
C-----ADD DCDS FOR SIGMA O-S (NOT USED BY SMD)
              if (LGR) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(118)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(118)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
C
C-----ADD CDS FOR SIGMA O-SI (NOT USED BY SMD)
            Else If (NTP .EQ. 14) Then
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(117)
C-----ADD DCDS FOR SIGMA O-SI (NOT USED BY SMD)
              If (LGR) Then
                Do L = 1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     &                +SIGMA(117)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     &                -SIGMA(117)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                 End Do
              End If
C
C-----ADD CDS FOR SIGMA O-P
            ELSE IF (NTP.EQ.15) THEN
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(114)
C-----ADD DCDS FOR SIGMA O-P
              if (LGR) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(114)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(114)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
C
C-----ADD CDS FOR SIGMA O-O (SM6 FORM ONLY)
            ELSE IF(NTP.EQ.8.AND.I.NE.J) THEN
              If (ISTS.eq.6) then
                STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(104)
C-----ADD DCDS FOR SIGMA O-O (SM6 FORM ONLY)
                if (LGR) then
                  do L=1,3
                    DSTS(L,I,I)=DSTS(L,I,I)
     .                  +SIGMA(104)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                    DSTS(L,J,I)=DSTS(L,J,I)
     .                  -SIGMA(104)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  end do
                end if
              End if
            ENDIF
 200      CONTINUE
        ENDIF
C
C-----ADD CDS FOR SIGMA N-C FOR -N-C- BOND AND N-C(2) FOR N-C=O BOND; N-C(2) IS NOT USED BY SMD
        IF (IAN(I).EQ.7) THEN
          RTKKS=0.0D0
          RTKKS2=0.D0
          DO 250 J=1,NAT
            NTP=IAN(J)
            IF(NTP.EQ.6)THEN
              RTKK3=0.0D0
              RTKK5=0.0D0
              DO 251 K=1,NAT
                IF (K.NE.I.AND.K.NE.J) THEN
                  KTP=IAN(K)
                  NKTPC=NATCNV(IAN(K))
                  NTPC =NATCNV(IAN(J))
                  RHLD2=RKKVAL(NKTPC,NTPC)
                  CUTOF2=RHLD2+0.3D0
                  IF(RLIO(IJ0(J,K)).LT.CUTOF2)THEN
                    RTKK2=DEXP(1.0D0-(BCCC/(1.0D0-
     .                    ((RLIO(IJ0(J,K))-RHLD2)/0.3D0))))/(EXPVAL)
                    IF(KTP.EQ.8) RTKK5=RTKK5+RTKK2
                    RTKK3=RTKK3+RTKK2
                  ENDIF
                ENDIF
 251          CONTINUE
              RTKKS=RTKKS+COT(IJ0(I,J))*RTKK3**2
              RTKKS2=RTKKS2+COT(IJ0(I,J))*RTKK5
            ENDIF
 250      CONTINUE
          DHOLDER=(RTKKS**1.3D0)
          STS(I)=STS(I)+DHOLDER*SIGMA(105)
          STS(I)=STS(I)+RTKKS2*SIGMA(111)
C-----ADD DCDS FOR SIGMA N-C FOR -N-C- BOND AND N-C(2) FOR N-C=O BOND; N-C(2) IS NOT USED BY SMD
          if (LGR) then
            do J=1,NAT
              NTP=IAN(J)
              if (NTP.EQ.6) then
                SCOTC=0.0D0
                SCOTO=0.0D0
                do K=1,NAT
                  if ((K.ne.I).and.(K.ne.J)) then
                    KTP=IAN(K)
                    COTJK=0.0D0
                    DCOTJK=0.0D0
                    NKTPC=NATCNV(IAN(K))
                    NTPC =NATCNV(IAN(J))
                    RHLD2=RKKVAL(NKTPC,NTPC)
                    CUTOF2=RHLD2+0.30
                    if (RLIO(ij0(J,K)).lt.CUTOF2) then
                      COTJK=exp(1.0D0-(BCCC/(1.0D0
     .                     -((RLIO(ij0(J,K))-RHLD2)/0.30))))/(EXPVAL)
                     DCOTJK=-COTJK*0.3/(RLIO(ij0(J,K))-CUTOF2)**2
                    end if
                    if (KTP.eq.8) then
                      SCOTO=SCOTO+COTJK
                      do L=1,3
                        DSTS(L,I,I)=DSTS(L,I,I)
     .                             +SIGMA(111)*COTJK
     .                             *DCOTDR(ij0(I,J))*URLIO(L,J,I)
                        DSTS(L,J,I)=DSTS(L,J,I)
     .                             -SIGMA(111)*COTJK
     .                             *DCOTDR(ij0(I,J))*URLIO(L,J,I)
     .                             +SIGMA(111)*COT(ij0(I,J))
     .                             *DCOTJK          *URLIO(L,K,J)
                        DSTS(L,K,I)=DSTS(L,K,I)
     .                             -SIGMA(111)*COT(ij0(I,J))
     .                             *DCOTJK          *URLIO(L,K,J)
                      end do
                    end if
                    SCOTC=SCOTC+COTJK
                  end if
                end do
C
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(105)*RTKKS**0.3D0*1.3D0
     .                *SCOTC**2*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(105)*RTKKS**0.3D0*1.3D0
     .                *SCOTC**2*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
C
                do K=1,NAT
                  if ((K.ne.I).and.(K.ne.J)) then
                    KTP=IAN(K)
                    COTJK=0.0D0
                    DCOTJK=0.0D0
                    NKTPC=NATCNV(IAN(K))
                    NTPC =NATCNV(IAN(J))
                    RHLD2=RKKVAL(NKTPC,NTPC)
                    CUTOF2=RHLD2+0.30
                    if (RLIO(ij0(J,K)).lt.CUTOF2) then
                      COTJK=exp(1.0D0-(1.0D0/(1.0D0
     .                     -((RLIO(ij0(J,K))-RHLD2)/0.30))))/(EXPVAL)
                      DCOTJK=-COTJK*0.3/(RLIO(ij0(J,K))-CUTOF2)**2
                    end if
                    do L=1,3
                      DSTS(L,J,I)=DSTS(L,J,I)+SIGMA(105)
     .                           *RTKKS**0.3D0*1.3D0
     .                           *2.0D0*SCOTC*COT(ij0(I,J))
     .                           *DCOTJK          *URLIO(L,K,J)
                      DSTS(L,K,I)=DSTS(L,K,I)-SIGMA(105)
     .                           *RTKKS**0.3D0*1.3D0
     .                           *2.0D0*SCOTC*COT(ij0(I,J))
     .                           *DCOTJK          *URLIO(L,K,J)
                    end do
                  end if
                end do
              end if
            end do
          end if
C
C-----ADD CDS FOR SIGMA N-C(3) FOR THE N-C TRIPLE BOND
          DRTKKU=0.D0
          RTKK=0.0D0
          DELTAR=0.065D0
          DO 461 J=1,NAT
            NTP=IAN(J)
            NTPC=NATCNV(NTP)
            IF(J.EQ.I) GOTO 461
            IF(NTP.EQ.6) THEN
              RHLD=1.225D0
              CUTOF1=RHLD+DELTAR
              IF(RLIO(IJ0(I,J)).LT.CUTOF1) THEN
                RTKK=RTKK+DEXP(1.0D0-(BCCC/(1.0D0-((RLIO(IJ0(I,J))-
     .               RHLD)/DELTAR))))/(EXPVAL)
C-----ADD DCDS FOR SIGMA N-C(3) FOR THE N-C TRIPLE BOND
                if (LGR) then
                  EXPONT=DELTAR/(RLIO(ij0(I,J))-CUTOF1)
                  DRTKK=-DEXP(EXPONT)*EXPONT/(RLIO(ij0(I,J))-CUTOF1)
                  do L=1,3
                    DSTS(L,I,I)=DSTS(L,I,I)
     .                         +SIGMA(116)*DRTKK*URLIO(L,J,I)
                    DSTS(L,J,I)=DSTS(L,J,I)
     .                         -SIGMA(116)*DRTKK*URLIO(L,J,I)
                   end do
                 end if
              END IF
            END IF
 461      CONTINUE
          STS(I)=STS(I)+RTKK*SIGMA(116)
        ENDIF
C
C-----ADD CDS FOR SIGMA C-C AND C-C(2); C-C(2) IS NOT USED BY SMD
        IF (IAN(I).EQ.6) THEN
          ITPC=NATCNV(6)
          DO 500 KCCC=1,2
            RTKK=0.0D0
            DO 450 J=1,NAT
              IF(J.EQ.I) GOTO 450
              NTP=IAN(J)
              IF(NTP.EQ.6)THEN
                NTPC=NATCNV(NTP)
                IF (KCCC.EQ.1) THEN
                  RHLD=RKKVAL(ITPC,NTPC)
                  DELTAR=0.30D0
                ELSE
                  RHLD=1.27D0
                  DELTAR=0.07D0
                ENDIF
                CUTOF1=RHLD+DELTAR
                IF(RLIO(IJ0(I,J)).LT.CUTOF1) THEN
                  RTKK=RTKK+DEXP(1.0D0-(BCCC/(1.0D0-((RLIO(IJ0(I,J))-
     .                 RHLD)/DELTAR))))/(EXPVAL)
C-----ADD DCDS FOR SIGMA C-C AND C-C(2); C-C(2) IS NOT USED BY SMD
                  if (LGR) then
                    EXPONT=DELTAR/(RLIO(ij0(I,J))-CUTOF1)
                    DRTKK=-DEXP(EXPONT)*EXPONT/(RLIO(ij0(I,J))-CUTOF1)
                    if(KCCC.EQ.1)then
                      do L=1,3
                        DSTS(L,I,I)=DSTS(L,I,I)
     .                             +SIGMA(101)*DRTKK*URLIO(L,J,I)
                        DSTS(L,J,I)=DSTS(L,J,I)
     .                             -SIGMA(101)*DRTKK*URLIO(L,J,I)
                      end do
                    else if (KCCC.EQ.2) then
                      do L=1,3
                        DSTS(L,I,I)=DSTS(L,I,I)
     .                             +SIGMA(102)*DRTKK*URLIO(L,J,I)
                        DSTS(L,J,I)=DSTS(L,J,I)
     .                             -SIGMA(102)*DRTKK*URLIO(L,J,I)
                      end do
                    end if
                  end if
                ENDIF
              ENDIF
 450        CONTINUE
            IF(KCCC.EQ.1)THEN
              STS(I)=STS(I)+RTKK*SIGMA(101)
            ELSE IF (KCCC.EQ.2) THEN
              STS(I)=STS(I)+RTKK*SIGMA(102)
            ENDIF
 500      CONTINUE
C
C-----ADD CDS FOR SIGMA C-N FOR ANY C-N BOND
          RTKK=0.0D0
          DO 460 J=1,NAT
            NTP=IAN(J)
            IF(NTP.EQ.7)THEN
              RTKK=RTKK+COT(IJ0(I,J))
            ENDIF
 460      CONTINUE
          HOLDRK=RTKK*RTKK
          STS(I)=STS(I)+HOLDRK*SIGMA(110)
C-----ADD DCDS FOR SIGMA C-N FOR ANY C-N BOND
          if (LGR) then
            do J=1,NAT
              NTP=IAN(J)
              if (NTP.eq.7) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(110)*2.0D0
     .                *RTKK*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(110)*2.0D0
     .                *RTKK*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
            end do
          end if
C
        ENDIF
C
C-----ADD CDS FOR SIGMA S-S FOR S-S BOND (NOT USED BY SMD)
        IF (IAN(I).EQ.16) THEN
          DO 550 J=1,NAT
            NTP=IAN(J)
            IF(NTP.EQ.16.AND.J.NE.I) THEN
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(107)
C-----ADD DCDS FOR SIGMA S-S FOR S-S BOND (NOT USED BY SMD)
              if (LGR) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(107)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(107)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
C
            ENDIF
 550      CONTINUE
        ENDIF
C
C-----ADD CDS FOR SIGMA S-P FOR S-P BOND (NOT USED BY SMD)
        IF (IAN(I).EQ.16) THEN
          DO 220 J=1,NAT
            NTP=IAN(J)
            IF(NTP.EQ.15)THEN
              STS(I)=STS(I)+COT(IJ0(I,J))*SIGMA(115)
C-----ADD DCDS FOR SIGMA S-P FOR S-P BOND (NOT USED BY SMD)
              if (LGR) then
                do L=1,3
                  DSTS(L,I,I)=DSTS(L,I,I)
     .                +SIGMA(115)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                  DSTS(L,J,I)=DSTS(L,J,I)
     .                -SIGMA(115)*DCOTDR(ij0(I,J))*URLIO(L,J,I)
                end do
              end if
C
            ENDIF
 220      CONTINUE
        ENDIF
 950  CONTINUE
      RETURN
      END
C
C   * IJ0
      FUNCTION IJ0(I,J)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      IF(I.GT.J) THEN
         IJ0=I*(I-1)/2+J
      ELSE
         IJ0=J*(J-1)/2+I
      END IF
      RETURN
      END
C
C   * DAREAL
C  ********************  DAREAL SUBROUTINE  **********************
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C     ACCESSIBLE SOLID ANGLE OF SPHERES (RADIANS). ANALYTICAL.
C     ALLOW MULTIPLE CONNECTED COMPONENTS ON A SPHERE.
C     REGULARIZE RARE EVENTS (POINTS SHARED BY MORE THAN 3 SPHERES)
C     BY A SMALL PERTURBATION ON RADIUS.
C
C  INPUT
C     RAD(I)           : RADII.
C     NAT              : NUMBER OF SPHERES.
C     K                : NO OF THE SPHERE STUDIED.
C     LGRX             : .TRUE. TO CALCULATE CARTESIAN DERIVATIVES.
C     LGRR             : .TRUE. TO CALCULATE DERIVATIVE VS RAD(K).
C  (REF)
C     RLIO(I): INTERCENTRE DISTANCES (LOW TRIANGLE),
C     URLIO(3,I,J)     : ASSOCIATED UNIT VECTORS: FROM i TO j.
C
C  OUTPUT
C     AREA          : ACCESSIBLE SOLID ANGLE FOR SPHERE NO K.
C     NCROSS        : NUMBER OF SPHERES OVERLAPPING K.
C     NC            : INDICE OF SPHERES CONNECTED TO K.
C   AND IF LGRX = .TRUE. :
C     DAREA         : DERIVATIVES OF THE ACCESSIBLE SOLID ANGLE WITH
C                     RESPECT TO THE CARTESIAN COORDINATES.
C     THAT IS: DAREA(I,J) = dAREA(K)/dCOORD(I,NC(J)) ... J=0,NCROSS.
C   AND IF LGRR = .TRUE. :
C     DAREAR        : DERIVATIVES OF THE ACCESSIBLE SOLID ANGLE WITH
C                     RESPECT TO RAD(K).
C
C     By D. Liotard, December 1992.
C     Derivatives by D. Rinaldi and D. Liotard
C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE DAREAL
     $(NAT,AREA,NCROSS,K,LGRX,
     $RAD,NC,DAREA,RLIO,URLIO,LAB,NCNCT,CONECT,CTHETA,STHETA,SIT,
     $DCSIT,DJCOSN,COSN,DSTETA,DCTETA,DCOSN,WORK,DIWORK,DJWORK,
     $DKWORK,D0WORK,DWORKR,DCAODD,DSITR,DCAPLY,DICOSN,DCASLC,
     $DCTETR,DSTETR,DCOSNR)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (PI=3.14159265358979D0)
      LOGICAL FREEIJ,FREEJI,CONECT,LPOLY,LGRX,LGRR
      DIMENSION RAD(*),NC(0:*),DAREA(3,0:*)
      DIMENSION RLIO(*),URLIO(3,NAT,*)
      DIMENSION LAB(*),NCNCT(2*NAT+1,*)
     1               ,CONECT(NAT,*)
      DIMENSION CTHETA(NAT,*),STHETA(*),SIT(*)
     1               ,DCSIT(3,*),DJCOSN(3,3,NAT,*)
     2               ,COSN(3,NAT,*),DSTETA(3,*)
     3               ,DCTETA(3,NAT,*),DCOSN(3,3,NAT,*)
     4               ,WORK(*),DIWORK(3,*),DJWORK(3,*)
     5               ,DKWORK(3,*),D0WORK(3,*),DWORKR(*)
     6               ,DCAODD(3,0:*),DSITR(*)
     6               ,DCAPLY(3,0:*)
     7               ,DICOSN(3,3,NAT,*),DCASLC(3,0:*)
     8               ,DCTETR(*),DSTETR(*)
     9               ,DCOSNR(3,NAT,*)
      DIMENSION DCNIJR(3),DVIJR(3)
     A               ,DCNIKR(3),DCC2I(3),DKCNIK(3,3),DAPHI(3),DBPHI(3)
     B               ,VIJ(3),DICOS(3,3),DJCOS(3,3),VN(3),DIVN(3,3)
     C               ,DJVN(3,3),DIVIJ(3,3),DJVIJ(3,3),DIWIJ(3,3)
     D               ,DJWIJ(3,3),DIAIJ(3),DJAIJ(3),DIBIJ(3),DJBIJ(3)
     E               ,DICIJ(3),DJCIJ(3),CNIJ(3),DICNIJ(3,3),DJCNIJ(3,3)
     F               ,CNIK(3),DICNIK(3,3)
      DATA EPSI /1.D-11/
C
      MXSS=2*NAT+1
C
      PCMIN=1000
      JPCNT=0
C
      LGRR=.FALSE.
C
      DASLIR=0.D0
      DCIJ=0.D0
      DSN2IJ=0.D0
      DAIJR=0.D0
      DBIJR=0.D0
      DCIJR=0.D0
      DAPOLR=0.D0
      DC2IR=0.D0
      DAODDR=0.D0
      DPHIR=0.D0
C
      DO I=1,NAT
      LAB(I)=0
      STHETA(I)=0.D0
      SIT(I)=0.D0
      DWORKR(I)=0.D0
      DSITR(I)=0.D0
      DCTETR(I)=0.D0
      DSTETR(I)=0.D0
      DO J=1,3
      DCSIT(J,I)=0.D0
      DSTETA(J,I)=0.D0
      DIWORK(J,I)=0.D0
      DJWORK(J,I)=0.D0
      DKWORK(J,I)=0.D0
      D0WORK(J,I)=0.D0
      ENDDO
      DO J=1,MXSS
      NCNCT(J,I)=0
      ENDDO
      DO J=1,NAT
      CONECT(I,J)=.FALSE.
      CTHETA(I,J)=0.D0
      DO KK=1,3
      COSN(KK,I,J)=0.D0
      DCTETA(KK,I,J)=0.D0
      DCOSNR(KK,I,J)=0.D0
      DO L=1,3
      DJCOSN(KK,L,I,J)=0.D0
      DCOSN(KK,L,I,J)=0.D0
      DICOSN(KK,L,I,J)=0.D0
      ENDDO
      ENDDO
      ENDDO
      ENDDO
      DO I=1,MXSS
      WORK(I)=0.D0
      ENDDO
      DO I=0,NAT
      DO J=1,3
      DCAODD(J,I)=0.D0
      DCAPLY(J,I)=0.D0
      DCASLC(J,I)=0.D0
      ENDDO
      ENDDO
C
C     EPSI:THRESHOLD TO AVOID DRAMATIC CONSEQUENCES OF ROUND-OFF ERRORS,
C     MUST BE GREATER THAN THE POOREST OF RELATIVE ERRORS ON FUNCTIONS:
C     SQRT, ATAN.
C
      TWOPI=PI+PI
      FOURPI=TWOPI+TWOPI
      RK=RAD(K)
      NC(0) = K
      NCROSS = 0
      IF (LGRX) CALL SCOPYMN(3,0D0,0,DAREA(1,0),1)
C
      IF (LGRR) DAREAR=0D0
      IF (RK.LE.0D0) THEN
         AREA=0D0
         RETURN
      ELSE
C
C
C     PRESET ACCESSIBLE SOLID ANGLES TO "FREE" SPHERES.
C
      AREA=FOURPI
      ENDIF
   10 NCROSS=0
      EPSK=EPSI*RK
C
C     CHECK IF SPHERE K NOT BURIED; SET UP FLAG FOR OVERLAP OF K.
C     -------------------------------------------------------------
C
      DO 20 I=1,NAT
      IF (I.EQ.K.OR.RAD(I).LE.0D0) GOTO 20
      IF (RK+RAD(I)-RLIO(IJ0(I,K)).LT.EPSK) GOTO 20
      IF (RLIO(IJ0(I,K))-ABS(RK-RAD(I)).LT.EPSK) THEN
         IF (RK.LE.RAD(I)) THEN
C           SPHERE K IMBEDDED IN SPHERE I
            AREA=0D0
            RETURN
         ENDIF
      ELSE
         NCROSS=NCROSS+1
         NC(NCROSS) = I
         NCNCT(NCROSS,1)=I
      ENDIF
   20 CONTINUE
      IF (NCROSS.EQ.0) THEN
C        THE SPHERE K IS FREE.
         RETURN
      ENDIF
C
C     SET UP DATA FOR SPHERICAL SEGMENTS (SS).
C     ----------------------------------------
C
      RKINV=0.5D0/RK
      RK2=RK**2
      DO 23 I=1,NCROSS
      LI=NCNCT(I,1)
      RIKINV=1.D0/RLIO(IJ0(LI,K))
C     COSINE OF HALF-ANGLE OF CONE.
      CTHETA(I,I)=RKINV*(RLIO(IJ0(LI,K))+(RK2-RAD(LI)**2)*RIKINV)
      STHETA(I)=SQRT(1.D0-CTHETA(I,I)**2)
      CONECT(I,I)=.FALSE.
C     DIRECTOR COSINES OF VECTOR KI.
      COSN(1,I,I)=URLIO(1,K,LI)
      COSN(2,I,I)=URLIO(2,K,LI)
      COSN(3,I,I)=URLIO(3,K,LI)
      IF (LGRX.OR.LGRR) THEN
         X=-CTHETA(I,I)/STHETA(I)
         IF (LGRX) THEN
            DRCTHT=RKINV*(1D0-(RK2-RAD(LI)**2)*RIKINV**2)
            DO 22 ICOR=1,3
            DCTETA(ICOR,I,I)=DRCTHT*COSN(ICOR,I,I)
            DSTETA(ICOR,I)=X*DCTETA(ICOR,I,I)
            COSNI=COSN(ICOR,I,I)*RIKINV
            DO 21 JCOR=1,ICOR
            COSNIJ=-COSNI*COSN(JCOR,I,I)
            DCOSN(ICOR,JCOR,I,I)=COSNIJ
   21       DCOSN(JCOR,ICOR,I,I)=COSNIJ
   22       DCOSN(ICOR,ICOR,I,I)=DCOSN(ICOR,ICOR,I,I)+RIKINV
         ENDIF
         IF (LGRR) THEN
            DCTETR(I)=RIKINV-CTHETA(I,I)/RK
            DSTETR(I)=X*DCTETR(I)
         ENDIF
      ENDIF
   23 CONTINUE
C
C     CONNECTIVITY MATRIX BETWEEN SS.
C     -------------------------------
C     TRUE IIF CONNECTED. DIAGONAL=TRUE IIF IMBEDDED.
C
      DO 40 I=2,NCROSS
      DO 30 J=1,I-1
      IF (CONECT(J,J)) GOTO 30
      CISJ=CTHETA(I,I)*STHETA(J)
      SICJ=STHETA(I)*CTHETA(J,J)
      SISJ=STHETA(I)*STHETA(J)
C     COSINES OF ANGLES BETWEEN VECTOR KI AND KJ.
      CTHETA(J,I)=DOTMN(COSN(1,I,I),COSN(1,J,J),3)
      IF (LGRX) THEN
         DO 24 ICOR=1,3
         DCTETA(ICOR,I,J)=DOTMN(DCOSN(1,ICOR,I,I),COSN(1,J,J),3)
   24    DCTETA(ICOR,J,I)=DOTMN(COSN(1,I,I),DCOSN(1,ICOR,J,J),3)
      ENDIF
      TIJ=CTHETA(J,I)-CTHETA(I,I)*CTHETA(J,J)
      IF (TIJ.GT.SISJ-EPSI*ABS(CISJ-SICJ)) THEN
         IF (CTHETA(J,J).GT.CTHETA(I,I)) THEN
C           SS J IMBEDDED IN SS I
            CONECT(J,J)=.TRUE.
         ELSE
C           SS I IMBEDDED IN SS J
            CONECT(I,I)=.TRUE.
            GOTO 40
         ENDIF
      ELSE
         EPSIJ=EPSI*(SICJ+CISJ)
         IF (SICJ+CISJ.GE.0D0) THEN
            CONECT(J,I)=TIJ.GT.EPSIJ-SISJ
         ELSE
            IF (TIJ.LE.-SISJ-EPSIJ) THEN
C              SPHERE K FULLY COVERED BY SS I PLUS SS J
               NCROSS=0
               AREA=0D0
               RETURN
            ELSE
               CONECT(J,I)=.TRUE.
            ENDIF
         ENDIF
         CONECT(I,J)=CONECT(J,I)
      ENDIF
   30 CONTINUE
   40 CONTINUE
C
C     SUM THE CONTRIBUTIONS FROM ISOLATED SS.
C     ---------------------------------------
C
      ASLICE=0D0
      IF (LGRX) CALL SCOPYMN(3*(NCROSS+1),0D0,0,DCASLC(1,0),1)
      IF (LGRR) DASLIR=0D0
      NCLUST=0
      DO 60 I=1,NCROSS
      IF (CONECT(I,I)) GOTO 60
      DO 50 J=1,NCROSS
      IF (CONECT(J,J)) GOTO 50
      IF (CONECT(J,I)) THEN
         NCLUST=NCLUST+1
C        LABEL OF NON-ISOLATED SS: LAB
         LAB(NCLUST)=I
C        PREPARE DATA FOR FURTHER USE
         WORK(NCLUST)=CTHETA(I,I)+EPSI*STHETA(I)
         WORK(NCLUST+NCROSS)=CTHETA(I,I)-EPSI*STHETA(I)
         GOTO 60
      ENDIF
   50 CONTINUE
      ASLICE=ASLICE+1.0D0-CTHETA(I,I)
      IF (LGRX) THEN
         DO 55 ICOR=1,3
         DCASLC(ICOR,I)=DCASLC(ICOR,I)-DCTETA(ICOR,I,I)
   55    DCASLC(ICOR,0)=DCASLC(ICOR,0)+DCTETA(ICOR,I,I)
      ENDIF
      IF (LGRR) DASLIR=DASLIR-DCTETR(I)
   60 CONTINUE
      ASLICE=TWOPI*ASLICE
      IF (LGRX) CALL DSCALMN (3*(NCROSS+1),TWOPI,DCASLC(1,0),1)
      IF (LGRR) DASLIR=DASLIR*TWOPI
      IF (NCLUST.EQ.0) THEN
C        NO CLUSTER OF SS REMAINS ON SPHERE K.
         AREA=FOURPI-ASLICE
         IF (LGRX) THEN
            DO 61 I=0,NCROSS
            DO 61 ICOR=1,3
   61       DAREA(ICOR,I)=-DCASLC(ICOR,I)
         ENDIF
         IF (LGRR) DAREAR=-DASLIR
         RETURN
      ENDIF
C
C     "FREE" INTERSECTIONS BETWEEN CLUSTERED SS.
C     ------------------------------------------
C
      NFREE=0
      NCNCT(MXSS,LAB(1))=0
      DO 80 I=2,NCLUST
      LI=LAB(I)
      NCNCT(MXSS,LI)=0
      DO 80 J=1,I-1
      LJ=LAB(J)
      IF (CONECT(LJ,LI)) THEN
C        DIRECTOR COSINES OF INTERSECTION POINTS BETWEEN SS
         SIN2IJ=1.D0/(1.D0-CTHETA(LJ,LI)**2)
         AIJ=(CTHETA(LI,LI)-CTHETA(LJ,LJ)*CTHETA(LJ,LI))*SIN2IJ
         BIJ=(CTHETA(LJ,LJ)-CTHETA(LI,LI)*CTHETA(LJ,LI))*SIN2IJ
         CIJ=SQRT((1.D0-AIJ*CTHETA(LI,LI)-BIJ*CTHETA(LJ,LJ))*SIN2IJ)
C        COMPUTE  VN = COSN(LI,LI) X COSN(LJ,LJ).
         VN(1)=COSN(2,LI,LI)*COSN(3,LJ,LJ)-COSN(3,LI,LI)*COSN(2,LJ,LJ)
         VN(2)=COSN(3,LI,LI)*COSN(1,LJ,LJ)-COSN(1,LI,LI)*COSN(3,LJ,LJ)
         VN(3)=COSN(1,LI,LI)*COSN(2,LJ,LJ)-COSN(2,LI,LI)*COSN(1,LJ,LJ)
         IF (LGRX.OR.LGRR) THEN
            DCIJ=0.5D0/CIJ
            IF (LGRX) THEN
               DO 62 ICOR=1,3
               DIVN(1,ICOR)=DCOSN(2,ICOR,LI,LI)*COSN(3,LJ,LJ)
     1                     -DCOSN(3,ICOR,LI,LI)*COSN(2,LJ,LJ)
               DIVN(2,ICOR)=DCOSN(3,ICOR,LI,LI)*COSN(1,LJ,LJ)
     1                     -DCOSN(1,ICOR,LI,LI)*COSN(3,LJ,LJ)
               DIVN(3,ICOR)=DCOSN(1,ICOR,LI,LI)*COSN(2,LJ,LJ)
     1                     -DCOSN(2,ICOR,LI,LI)*COSN(1,LJ,LJ)
               DJVN(1,ICOR)=COSN(2,LI,LI)*DCOSN(3,ICOR,LJ,LJ)
     1                     -COSN(3,LI,LI)*DCOSN(2,ICOR,LJ,LJ)
               DJVN(2,ICOR)=COSN(3,LI,LI)*DCOSN(1,ICOR,LJ,LJ)
     1                     -COSN(1,LI,LI)*DCOSN(3,ICOR,LJ,LJ)
   62          DJVN(3,ICOR)=COSN(1,LI,LI)*DCOSN(2,ICOR,LJ,LJ)
     1                     -COSN(2,LI,LI)*DCOSN(1,ICOR,LJ,LJ)
               DSN2IJ=2.0D0*CTHETA(LJ,LI)*SIN2IJ
            ENDIF
            IF (LGRR) THEN
               DAIJR=(DCTETR(LI)-DCTETR(LJ)*CTHETA(LJ,LI))*SIN2IJ
               DBIJR=(DCTETR(LJ)-DCTETR(LI)*CTHETA(LJ,LI))*SIN2IJ
               DCIJR=-SIN2IJ*(DAIJR*CTHETA(LI,LI)+AIJ*DCTETR(LI)
     1         +DBIJR*CTHETA(LJ,LJ)+BIJ*DCTETR(LJ))*DCIJ
            ENDIF
         ENDIF
C
         DO 63 ICOR=1,3
C
C        ----- INTERSECTION PIJ -----
C
         COSN(ICOR,LI,LJ)=AIJ*COSN(ICOR,LI,LI)+BIJ*COSN(ICOR,LJ,LJ)
     1                   +CIJ*VN(ICOR)
C
C        ----- INTERSECTION PJI -----
C
         COSN(ICOR,LJ,LI)=AIJ*COSN(ICOR,LI,LI)+BIJ*COSN(ICOR,LJ,LJ)
     1                   -CIJ*VN(ICOR)
         IF (LGRX) THEN
            DIS2IJ=DSN2IJ*DCTETA(ICOR,LI,LJ)
            DJS2IJ=DSN2IJ*DCTETA(ICOR,LJ,LI)
            DIAIJ(ICOR) =
     1           (DCTETA(ICOR,LI,LI)-CTHETA(LJ,LJ)*DCTETA(ICOR,LI,LJ))
     2           *SIN2IJ+AIJ*DIS2IJ
            DJAIJ(ICOR) =
     1           (-DCTETA(ICOR,LJ,LJ)*CTHETA(LJ,LI)-CTHETA(LJ,LJ)
     2           *DCTETA(ICOR,LJ,LI))*SIN2IJ + AIJ*DJS2IJ
            DJBIJ(ICOR) =
     1           (DCTETA(ICOR,LJ,LJ)-CTHETA(LI,LI)*DCTETA(ICOR,LJ,LI))
     2           *SIN2IJ + BIJ*DJS2IJ
            DIBIJ(ICOR) =
     1           (-DCTETA(ICOR,LI,LI)*CTHETA(LJ,LI)-CTHETA(LI,LI)
     2           *DCTETA(ICOR,LI,LJ))*SIN2IJ + BIJ*DIS2IJ
            DICIJ(ICOR) =
     1         -DCIJ*((DIAIJ(ICOR)*CTHETA(LI,LI)+AIJ*DCTETA(ICOR,LI,LI)
     2         +DIBIJ(ICOR)*CTHETA(LJ,LJ))*SIN2IJ)+0.5D0*CIJ*DIS2IJ
            DJCIJ(ICOR) =
     1         -DCIJ*((DJBIJ(ICOR)*CTHETA(LJ,LJ)+BIJ*DCTETA(ICOR,LJ,LJ)
     2         +DJAIJ(ICOR)*CTHETA(LI,LI))*SIN2IJ)+0.5D0*CIJ*DJS2IJ
         ENDIF
         IF (LGRR) THEN
            DCOSNR(ICOR,LI,LJ)=DAIJR*COSN(ICOR,LI,LI)
     1                        +DBIJR*COSN(ICOR,LJ,LJ)+DCIJR*VN(ICOR)
            DCOSNR(ICOR,LJ,LI)=DAIJR*COSN(ICOR,LI,LI)
     1                        +DBIJR*COSN(ICOR,LJ,LJ)-DCIJR*VN(ICOR)
         ENDIF
   63    CONTINUE
C
         IF (LGRX) THEN
            DO 64 ICOR=1,3
            DO 64 JCOR=1,3
            DICOS(ICOR,JCOR)=DIAIJ(JCOR)*COSN(ICOR,LI,LI) +
     1      AIJ*DCOSN(ICOR,JCOR,LI,LI)+DIBIJ(JCOR)*COSN(ICOR,LJ,LJ)
            DJCOS(ICOR,JCOR)=DJAIJ(JCOR)*COSN(ICOR,LI,LI) +
     1      DJBIJ(JCOR)*COSN(ICOR,LJ,LJ)+BIJ*DCOSN(ICOR,JCOR,LJ,LJ)
            DIWIJ(ICOR,JCOR)=DICIJ(JCOR)*VN(ICOR)+CIJ*DIVN(ICOR,JCOR)
            DJWIJ(ICOR,JCOR)=DJCIJ(JCOR)*VN(ICOR)+CIJ*DJVN(ICOR,JCOR)
            DICOSN(ICOR,JCOR,LI,LJ)=DICOS(ICOR,JCOR)+DIWIJ(ICOR,JCOR)
            DJCOSN(ICOR,JCOR,LI,LJ)=DJCOS(ICOR,JCOR)+DJWIJ(ICOR,JCOR)
            DICOSN(ICOR,JCOR,LJ,LI)=DICOS(ICOR,JCOR)-DIWIJ(ICOR,JCOR)
   64       DJCOSN(ICOR,JCOR,LJ,LI)=DJCOS(ICOR,JCOR)-DJWIJ(ICOR,JCOR)
         ENDIF
C
C        ----- INTERSECTION PIJ ------
C
         FREEIJ=.TRUE.
C
C        ----- INTERSECTION PJI -----
C
         FREEJI=.TRUE.
C
C        ----- SET UP TABLE OF FREE INTERSECTIONS BETWEEN SS. -----
C        ----- HERE IS THE L**3 PART OF THE CODE -----
C        ----- (L= MEAN VALUE OF NEIGHBORS) -----
C
         DO 70 L=1,NCLUST
         IF (L.EQ.I.OR.L.EQ.J) GOTO 70
         LL=LAB(L)
         IF (CONECT(LL,LI).AND.CONECT(LL,LJ)) THEN
            IF (FREEJI) THEN
               CHEK=DOTMN(COSN(1,LJ,LI),COSN(1,LL,LL),3)
               IF (CHEK.GT.WORK(L)) THEN
                  FREEJI=.FALSE.
               ELSE
                  PCMIN=MIN(PCMIN,ABS(WORK(L+NCROSS)-CHEK))
                  IF (CHEK.GE.WORK(L+NCROSS)) THEN
C
C        ----- SPHERES K, LAB(I), LAB(J), LAB(L) SHARE A POINT -----
C        ----- (AT THRESHOLD EPSI IN ANGLE). INCREASE THE -----
C        ----- RADIUS OF SPHERE K AND RESTART STUDY OF SPHERE K -----
C
                     RK=RK*(1D0+4D0*EPSI)
                     JPCNT=JPCNT + 1
                     GOTO 10
                  ENDIF
               ENDIF
            ENDIF
            IF (FREEIJ) THEN
               CHEK=DOTMN(COSN(1,LI,LJ),COSN(1,LL,LL),3)
               IF (CHEK.GT.WORK(L)) THEN
                  FREEIJ=.FALSE.
               ELSE IF (CHEK.GE.WORK(L+NCROSS)) THEN
                  RK=RK*(1D0+4D0*EPSI)
                  GOTO 10
               ENDIF
            ENDIF
            IF (.NOT.(FREEIJ.OR.FREEJI)) GOTO 80
         ENDIF
   70    CONTINUE
C
C        ----- ONE OR TWO FREE INTERSECTIONS FOUND. UPDATE TABLE -----
C
         CTHETA(LI,LJ)=CTHETA(LJ,LI)
         IF (FREEJI) THEN
            NFREE=NFREE+1
            M=NCNCT(MXSS,LI)+1
            NCNCT(M,LI)=LJ
            NCNCT(MXSS,LI)=M
            M=NCNCT(MXSS,LJ)+1
            NCNCT(M,LJ)=-LI
            NCNCT(MXSS,LJ)=M
         ENDIF
         IF (FREEIJ) THEN
            NFREE=NFREE+1
            M=NCNCT(MXSS,LI)+1
            NCNCT(M,LI)=-LJ
            NCNCT(MXSS,LI)=M
            M=NCNCT(MXSS,LJ)+1
            NCNCT(M,LJ)=LI
            NCNCT(MXSS,LJ)=M
         ENDIF
      ENDIF
   80 CONTINUE
      IF (NFREE.EQ.0) THEN
C        SPHERE K BURIED BY A CLUSTER OF SS
         NCROSS = 0
         AREA=0D0
         RETURN
      ENDIF
C
C     ORIENTED ANGLES OF SLICES OF SS.
C     --------------------------------
C
      APOLY=0D0
      IF (LGRX) CALL SCOPYMN (3*(NCROSS+1),0D0,0,DCAPLY(1,0),1)
      IF (LGRR) DAPOLR=0D0
      DO 130 I=1,NCLUST
      IF (LGRX) CALL SCOPYMN (3*(NCROSS+1),0D0,0,DCAODD(1,0),1)
      LI=LAB(I)
      NPHI=NCNCT(MXSS,LI)
      IF (NPHI.EQ.0) GOTO 130
      LJ=NCNCT(1,LI)
      IF (LJ.GT.0) THEN
         DO 83 ICOR=1,3
         CNIJ(ICOR) = COSN(ICOR,LJ,LI)
         IF (LGRX.AND.LI.GE.LJ) THEN
            DO 81 JCOR=1,3
            DICNIJ(ICOR,JCOR)=DICOSN(ICOR,JCOR,LJ,LI)
  81        DJCNIJ(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LJ,LI)
         ELSE IF (LGRX) THEN
            DO 82 JCOR=1,3
            DICNIJ(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LJ,LI)
  82        DJCNIJ(ICOR,JCOR)=DICOSN(ICOR,JCOR,LJ,LI)
         ENDIF
         IF (LGRR) DCNIJR(ICOR)=DCOSNR(ICOR,LJ,LI)
  83     CONTINUE
      ELSE
         LJ=-LJ
         DO 86 ICOR=1,3
         CNIJ(ICOR)=COSN(ICOR,LI,LJ)
         IF (LGRX.AND.LI.GE.LJ) THEN
            DO 84 JCOR=1,3
            DICNIJ(ICOR,JCOR)=DICOSN(ICOR,JCOR,LI,LJ)
  84        DJCNIJ(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LI,LJ)
         ELSE IF (LGRX) THEN
            DO 85 JCOR=1,3
            DICNIJ(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LI,LJ)
  85        DJCNIJ(ICOR,JCOR)=DICOSN(ICOR,JCOR,LI,LJ)
         ENDIF
         IF (LGRR) DCNIJR(ICOR)=DCOSNR(ICOR,LI,LJ)
  86     CONTINUE
         NCNCT(1,LI)=LJ
      ENDIF
      SIT(LI)=1.0D0/STHETA(LI)
      C2I=CTHETA(LI,LI)**2
      VIJ(1)=COSN(2,LI,LI)*CNIJ(3)-COSN(3,LI,LI)*CNIJ(2)
      VIJ(2)=COSN(3,LI,LI)*CNIJ(1)-COSN(1,LI,LI)*CNIJ(3)
      VIJ(3)=COSN(1,LI,LI)*CNIJ(2)-COSN(2,LI,LI)*CNIJ(1)
      LPOLY =DOTMN(VIJ,COSN(1,LJ,LJ),3).GT.0D0
C
      IF (LGRX.OR.LGRR) THEN
         X=2D0*CTHETA(LI,LI)
         Y=-SIT(LI)**2
         IF (LGRX) THEN
            DO 87 ICOR=1,3
            DCSIT(ICOR,LI)=Y*DSTETA(ICOR,LI)
            DCC2I(ICOR)=X*DCTETA(ICOR,LI,LI)
            DIVIJ(1,ICOR)=
     1      DCOSN(2,ICOR,LI,LI)*CNIJ(3)-DCOSN(3,ICOR,LI,LI)*CNIJ(2)
     2      +COSN(2,LI,LI)*DICNIJ(3,ICOR)-COSN(3,LI,LI)*DICNIJ(2,ICOR)
            DIVIJ(2,ICOR)=
     1      DCOSN(3,ICOR,LI,LI)*CNIJ(1)-DCOSN(1,ICOR,LI,LI)*CNIJ(3)
     2      +COSN(3,LI,LI)*DICNIJ(1,ICOR)-COSN(1,LI,LI)*DICNIJ(3,ICOR)
            DIVIJ(3,ICOR)=
     1      DCOSN(1,ICOR,LI,LI)*CNIJ(2)-DCOSN(2,ICOR,LI,LI)*CNIJ(1)
     2      +COSN(1,LI,LI)*DICNIJ(2,ICOR)-COSN(2,LI,LI)*DICNIJ(1,ICOR)
            DJVIJ(1,ICOR)=
     1      COSN(2,LI,LI)*DJCNIJ(3,ICOR)-COSN(3,LI,LI)*DJCNIJ(2,ICOR)
            DJVIJ(2,ICOR)=
     1      COSN(3,LI,LI)*DJCNIJ(1,ICOR)-COSN(1,LI,LI)*DJCNIJ(3,ICOR)
  87        DJVIJ(3,ICOR)=
     1      COSN(1,LI,LI)*DJCNIJ(2,ICOR)-COSN(2,LI,LI)*DJCNIJ(1,ICOR)
         ENDIF
         IF (LGRR) THEN
            DSITR(LI)=Y*DSTETR(LI)
            DC2IR=X*DCTETR(LI)
            DVIJR(1)=COSN(2,LI,LI)*DCNIJR(3)-COSN(3,LI,LI)*DCNIJR(2)
            DVIJR(2)=COSN(3,LI,LI)*DCNIJR(1)-COSN(1,LI,LI)*DCNIJR(3)
            DVIJR(3)=COSN(1,LI,LI)*DCNIJR(2)-COSN(2,LI,LI)*DCNIJR(1)
         ENDIF
      ENDIF
C
C     MAKE A LOOP OVER THE SS I, COMPUTE DIHEDRALS BETWEEN 0 AND TWOPI
C
      DO 95 J=2,NPHI
      LJ=NCNCT(J,LI)
      IF (LJ.GT.0) THEN
         DO 90 ICOR=1,3
         CNIK(ICOR) = COSN(ICOR,LJ,LI)
         IF (LGRX.AND.LI.GE.LJ) THEN
            DO 88 JCOR=1,3
            DICNIK(ICOR,JCOR)=DICOSN(ICOR,JCOR,LJ,LI)
   88       DKCNIK(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LJ,LI)
         ELSE IF (LGRX) THEN
            DO 89 JCOR=1,3
            DICNIK(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LJ,LI)
   89       DKCNIK(ICOR,JCOR)=DICOSN(ICOR,JCOR,LJ,LI)
         ENDIF
         IF (LGRR) DCNIKR(ICOR)=DCOSNR(ICOR,LJ,LI)
   90    CONTINUE
      ELSE
         LJ =-LJ
         DO 93 ICOR=1,3
         CNIK(ICOR)=COSN(ICOR,LI,LJ)
         IF (LGRX.AND.LI.GE.LJ) THEN
            DO 91 JCOR=1,3
            DICNIK(ICOR,JCOR)=DICOSN(ICOR,JCOR,LI,LJ)
   91       DKCNIK(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LI,LJ)
         ELSE IF (LGRX) THEN
            DO 92 JCOR=1,3
            DICNIK(ICOR,JCOR)=DJCOSN(ICOR,JCOR,LI,LJ)
   92       DKCNIK(ICOR,JCOR)=DICOSN(ICOR,JCOR,LI,LJ)
         ENDIF
         IF (LGRR) DCNIKR(ICOR)=DCOSNR(ICOR,LI,LJ)
   93    CONTINUE
         NCNCT(J,LI)=LJ
      ENDIF
C
C     NOTE: ACCURACY IS CRUCIAL HERE (USE ATAN2, NOT ACOS + ASIN)
C
      X=DOTMN(VIJ,CNIK,3)
      Y=DOTMN(CNIK,CNIJ,3)-C2I
      WORK(J-1)=ATAN2(X,Y)
      IF (WORK(J-1).LE.0D0) WORK(J-1)=WORK(J-1)+TWOPI
      IF (LGRX.OR.LGRR) THEN
         DX=-X/(X**2+Y**2)
         DY= Y/(X**2+Y**2)
         IF (LGRX) THEN
C           CARTESIAN DERIVATIVES OF WORK
            DO 94 ICOR=1,3
            DIX=DOTMN(DIVIJ(1,ICOR),CNIK,3)+
     $ DOTMN(VIJ,DICNIK(1,ICOR),3)
            DIY=DOTMN(DICNIK(1,ICOR),CNIJ,3)+
     $ DOTMN(CNIK,DICNIJ(1,ICOR),3)-DCC2I(ICOR)
            DJX=DOTMN(DJVIJ(1,ICOR),CNIK,3)
            DJY=DOTMN(CNIK,DJCNIJ(1,ICOR),3)
            DKX=DOTMN(VIJ,DKCNIK(1,ICOR),3)
            DKY=DOTMN(DKCNIK(1,ICOR),CNIJ,3)
            DIWORK(ICOR,J-1)= DY*DIX+DX*DIY
            DJWORK(ICOR,J-1)= DY*DJX+DX*DJY
            DKWORK(ICOR,J-1)= DY*DKX+DX*DKY
   94       D0WORK(ICOR,J-1)=-DIWORK(ICOR,J-1)-DJWORK(ICOR,J-1)
     1                       -DKWORK(ICOR,J-1)
         ENDIF
         IF (LGRR) THEN
C           DERIVATIVE OF WORK VS RAD(K)
            DXR=DOTMN(DVIJR,CNIK,3)+DOTMN(VIJ,DCNIKR,3)
            DYR=DOTMN(DCNIKR,CNIJ,3)+DOTMN(CNIK,DCNIJR,3)-DC2IR
            DWORKR(J-1)=DY*DXR+DX*DYR
         ENDIF
      ENDIF
   95 CONTINUE
      IF (NPHI.EQ.2) THEN
C        TRIVIAL CASE, NO SORT NEEDED
         AODD=WORK(1)
         IF (LGRX) THEN
            LJ=NCNCT(1,LI)
            LK=NCNCT(2,LI)
            DO 96 ICOR=1,3
            DCAODD(ICOR,LI)=DCAODD(ICOR,LI)+DIWORK(ICOR,1)
            DCAODD(ICOR,LJ)=DCAODD(ICOR,LJ)+DJWORK(ICOR,1)
            DCAODD(ICOR,LK)=DCAODD(ICOR,LK)+DKWORK(ICOR,1)
   96       DCAODD(ICOR,0 )=DCAODD(ICOR,0 )+D0WORK(ICOR,1)
         ENDIF
         IF (LGRR) DAODDR=DWORKR(1)
      ELSE
C
C        ----- SORT THE DIHEDRALS OF SS NO I IN ASCENDING ORDER -----
C
         DO 100 J=1,NPHI-2
         DO 100 L=J+1,NPHI-1
         IF (WORK(J).GT.WORK(L)) THEN
            BUF=WORK(L)
            WORK(L)=WORK(J)
            WORK(J)=BUF
            M=NCNCT(L+1,LI)
            NCNCT(L+1,LI)=NCNCT(J+1,LI)
            NCNCT(J+1,LI)=M
            IF (LGRX) THEN
               CALL DSWAPMN(3,DIWORK(1,L),1,DIWORK(1,J),1)
               CALL DSWAPMN(3,DJWORK(1,L),1,DJWORK(1,J),1)
               CALL DSWAPMN(3,DKWORK(1,L),1,DKWORK(1,J),1)
               CALL DSWAPMN(3,D0WORK(1,L),1,D0WORK(1,J),1)
            ENDIF
            IF (LGRR) THEN
               BUF=DWORKR(L)
               DWORKR(L)=DWORKR(J)
               DWORKR(J)=BUF
            ENDIF
         ENDIF
  100    CONTINUE
C
C        COMPUTE THE SUM OF DIHEDRALS OF ORDERED SLICES (EVEN IN NUMBER)
C        ---------------------------------------------------------------
C
         AODD=WORK(1)
         IF (LGRX) THEN
            LJ=NCNCT(1,LI)
            LK=NCNCT(2,LI)
            DO 101 ICOR=1,3
            DCAODD(ICOR,LI)=DCAODD(ICOR,LI)+DIWORK(ICOR,1)
            DCAODD(ICOR,LJ)=DCAODD(ICOR,LJ)+DJWORK(ICOR,1)
            DCAODD(ICOR,LK)=DCAODD(ICOR,LK)+DKWORK(ICOR,1)
  101       DCAODD(ICOR,0 )=DCAODD(ICOR,0 )+D0WORK(ICOR,1)
         ENDIF
         IF (LGRR) DAODDR=DWORKR(1)
         DO 110 J=3,NPHI-1,2
         AODD=AODD+WORK(J)-WORK(J-1)
         IF (LGRX) THEN
            LK=NCNCT(J,LI)
            LL=NCNCT(J+1,LI)
            DO 102 ICOR=1,3
            DCAODD(ICOR,LI)=DCAODD(ICOR,LI)
     1                     +DIWORK(ICOR,J )-DIWORK(ICOR,J-1)
            DCAODD(ICOR,LJ)=DCAODD(ICOR,LJ)
     1                     +DJWORK(ICOR,J )-DJWORK(ICOR,J-1)
            DCAODD(ICOR,LK)=DCAODD(ICOR,LK)-DKWORK(ICOR,J-1)
            DCAODD(ICOR,LL)=DCAODD(ICOR,LL)+DKWORK(ICOR,J)
  102       DCAODD(ICOR,0 )=DCAODD(ICOR,0 )
     1                     +D0WORK(ICOR,J )-D0WORK(ICOR,J-1)
         ENDIF
         IF (LGRR) DAODDR=DAODDR+DWORKR(J)-DWORKR(J-1)
  110    CONTINUE
      ENDIF
      AEVEN=TWOPI-AODD
      X=1D0-CTHETA(LI,LI)
      IF (LPOLY) THEN
C        ODD DIHEDRALS ARE VERTICE OF POLYGONS
         APOLY=APOLY+AODD
         ASLICE=ASLICE+AEVEN*X
         IF (LGRX) THEN
            DO 111 LT=0,NCROSS
            DCAPLY(1,LT)=DCAPLY(1,LT)+DCAODD(1,LT)
            DCASLC(1,LT)=DCASLC(1,LT)-DCAODD(1,LT)*X
            DCAPLY(2,LT)=DCAPLY(2,LT)+DCAODD(2,LT)
            DCASLC(2,LT)=DCASLC(2,LT)-DCAODD(2,LT)*X
            DCAPLY(3,LT)=DCAPLY(3,LT)+DCAODD(3,LT)
  111       DCASLC(3,LT)=DCASLC(3,LT)-DCAODD(3,LT)*X
            DCASLC(1,LI)=DCASLC(1,LI)-AEVEN*DCTETA(1,LI,LI)
            DCASLC(1,0 )=DCASLC(1,0 )+AEVEN*DCTETA(1,LI,LI)
            DCASLC(2,LI)=DCASLC(2,LI)-AEVEN*DCTETA(2,LI,LI)
            DCASLC(2,0 )=DCASLC(2,0 )+AEVEN*DCTETA(2,LI,LI)
            DCASLC(3,LI)=DCASLC(3,LI)-AEVEN*DCTETA(3,LI,LI)
            DCASLC(3,0 )=DCASLC(3,0 )+AEVEN*DCTETA(3,LI,LI)
         ENDIF
         IF (LGRR) THEN
            DAPOLR=DAPOLR+DAODDR
            DASLIR=DASLIR-DAODDR*X-AEVEN*DCTETR(LI)
         ENDIF
      ELSE
C        EVEN DIHEDRALS ARE VERTICE OF POLYGONS
         APOLY=APOLY+AEVEN
         ASLICE=ASLICE+AODD*X
         IF (LGRX) THEN
            DO 112 LT=0,NCROSS
            DCAPLY(1,LT)=DCAPLY(1,LT)-DCAODD(1,LT)
            DCASLC(1,LT)=DCASLC(1,LT)+DCAODD(1,LT)*X
            DCAPLY(2,LT)=DCAPLY(2,LT)-DCAODD(2,LT)
            DCASLC(2,LT)=DCASLC(2,LT)+DCAODD(2,LT)*X
            DCAPLY(3,LT)=DCAPLY(3,LT)-DCAODD(3,LT)
  112       DCASLC(3,LT)=DCASLC(3,LT)+DCAODD(3,LT)*X
            DCASLC(1,LI)=DCASLC(1,LI)-AODD*DCTETA(1,LI,LI)
            DCASLC(1,0 )=DCASLC(1,0 )+AODD*DCTETA(1,LI,LI)
            DCASLC(2,LI)=DCASLC(2,LI)-AODD*DCTETA(2,LI,LI)
            DCASLC(2,0 )=DCASLC(2,0 )+AODD*DCTETA(2,LI,LI)
            DCASLC(3,LI)=DCASLC(3,LI)-AODD*DCTETA(3,LI,LI)
            DCASLC(3,0 )=DCASLC(3,0 )+AODD*DCTETA(3,LI,LI)
         ENDIF
         IF (LGRR) THEN
            DAPOLR=DAPOLR-DAODDR
            DASLIR=DASLIR+DAODDR*X-AODD*DCTETR(LI)
         ENDIF
C        REORDER LABELS SO THAT POLYGON'S VERTICE APPEAR AT ODD RANKS
         M=NCNCT(1,LI)
         DO 120 J=1,NPHI-1
  120    NCNCT(J,LI)=NCNCT(J+1,LI)
         NCNCT(NPHI,LI)=M
      ENDIF
  130 CONTINUE
C
C     NOW TO COUNT THE NUMBER OF POLYGONS AND TO SUM INTERSECTION ANGLES
C     ------------------------------------------------------------------
C
      NPOLY=0
      DO 170 I=1,NCLUST
      LI=LAB(I)
      DO 160 J=2,NCNCT(MXSS,LI),2
      IF (NCNCT(J,LI).NE.0) THEN
C        DEFINE THE "FIRST" VERTEX OF A POLYGON
         IA=NCNCT(J-1,LI)
         IB=LI
         NCNCT(J,LI)=0
         PHI1=CTHETA(IA,IB)-CTHETA(IA,IA)*CTHETA(IB,IB)
         PHI2=SIT(IA)*SIT(IB)
         PHI=ACOS(PHI1*PHI2)
         APOLY=APOLY+PHI
         IF (LGRX.OR.LGRR) THEN
            S1NPHI=1D0/SIN(PHI)
            PHI1=S1NPHI*PHI1
            PHI2=S1NPHI*PHI2
            IF (LGRX) THEN
               DO 131 ICOR=1,3
               DAPHI(ICOR)=(DCTETA(ICOR,IA,IB)-CTHETA(IB,IB)
     1         *DCTETA(ICOR,IA,IA))*PHI2+PHI1*DCSIT(ICOR,IA)*SIT(IB)
               DBPHI(ICOR)=(DCTETA(ICOR,IB,IA)-CTHETA(IA,IA)
     1         *DCTETA(ICOR,IB,IB))*PHI2+PHI1*SIT(IA)*DCSIT(ICOR,IB)
               DCAPLY(ICOR,IA)=DCAPLY(ICOR,IA)-DAPHI(ICOR)
               DCAPLY(ICOR,IB)=DCAPLY(ICOR,IB)-DBPHI(ICOR)
  131          DCAPLY(ICOR,0 )=DCAPLY(ICOR,0 )+DAPHI(ICOR)+DBPHI(ICOR)
            ENDIF
            IF (LGRR) THEN
               DPHIR=
     1         (DCTETR(IA)*CTHETA(IB,IB)+CTHETA(IA,IA)*DCTETR(IB))*PHI2
     2         -PHI1*(DSITR(IA)*SIT(IB)+SIT(IA)*DSITR(IB))
               DAPOLR=DAPOLR+DPHIR
            ENDIF
         ENDIF
C
C        FOLLOW AND ERASE VERTICES OF THE CURRENT POLYGON
C
  140    L=2
  141    IF (L.GT.NCNCT(MXSS,IA)) GOTO 150
         IF (NCNCT(L,IA).EQ.IB) THEN
            IBOLD=IB
            NCNCT(L,IA)=0
            IB=IA
            IA=NCNCT(L-1,IB)
            IF (IA.NE.IBOLD) THEN
               PHI1=CTHETA(IA,IB)-CTHETA(IA,IA)*CTHETA(IB,IB)
               PHI2=SIT(IA)*SIT(IB)
               PHI=ACOS(PHI1*PHI2)
               IF (LGRX.OR.LGRR) THEN
                  S1NPHI=1D0/SIN(PHI)
                  PHI1=S1NPHI*PHI1
                  PHI2=S1NPHI*PHI2
                  IF (LGRX) THEN
                     DO 142 ICOR=1,3
                     DAPHI(ICOR)=(DCTETA(ICOR,IA,IB)
     1               -CTHETA(IB,IB)*DCTETA(ICOR,IA,IA))*PHI2
     2               +PHI1*DCSIT(ICOR,IA)*SIT(IB)
  142                DBPHI(ICOR)=(DCTETA(ICOR,IB,IA)
     1               -CTHETA(IA,IA)*DCTETA(ICOR,IB,IB))*PHI2
     2               +PHI1*SIT(IA)*DCSIT(ICOR,IB)
                  ENDIF
                  IF (LGRR) THEN
                     DPHIR=(DCTETR(IA)*CTHETA(IB,IB)+
     1               CTHETA(IA,IA)*DCTETR(IB))*PHI2
     2               -PHI1*(DSITR(IA)*SIT(IB)+SIT(IA)*DSITR(IB))
                  ENDIF
               ENDIF
            ELSE
               IB=LI
               IA=NCNCT(J-1,LI)
            ENDIF
            APOLY=APOLY+PHI
            IF (LGRX) THEN
               DO 143 ICOR=1,3
               DCAPLY(ICOR,IA)=DCAPLY(ICOR,IA)-DAPHI(ICOR)
               DCAPLY(ICOR,IB)=DCAPLY(ICOR,IB)-DBPHI(ICOR)
  143          DCAPLY(ICOR,0 )=DCAPLY(ICOR,0 )+DAPHI(ICOR)+DBPHI(ICOR)
            ENDIF
            IF (LGRR) DAPOLR=DAPOLR+DPHIR
            GOTO 140
         ENDIF
         L=L+2
         GOTO 141
C        ORIENTED LOOP OVER CURRENT POLYGON COMPLETED
  150    NPOLY=NPOLY+1
      ENDIF
  160 CONTINUE
  170 CONTINUE
C
C     ACCESSIBLE SOLID ANGLE FOR SPHERE K.
C     ------------------------------------
C     POLYGONS IMBEDDED IN OTHERS TAKEN INTO ACCOUNT BY MODULO(4PI)
C
      APOLY=APOLY+(NPOLY-NFREE)*TWOPI
      AREA=FOURPI-ASLICE-MOD(APOLY,FOURPI)
      IF (LGRX) THEN
         DO 200 I=0,NCROSS
         DAREA(1,I)=-DCASLC(1,I)-DCAPLY(1,I)
         DAREA(2,I)=-DCASLC(2,I)-DCAPLY(2,I)
  200    DAREA(3,I)=-DCASLC(3,I)-DCAPLY(3,I)
      ENDIF
      IF (LGRR) DAREAR=-DASLIR-DAPOLR
      RETURN
      END
C
C  ************** LIBRARIES FOR DAREAL SUBROUTINE **************
C
C   * SCOPYMN
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     COPIES A VECTOR, X, TO A VECTOR, Y.
C     USES UNROLLED LOOPS FOR INCREMENTS EQUAL TO ONE.
C     JACK DONGARRA, LINPACK, 3/11/78.
C
C     LINPACK ROUTINE DCOPY CONVERTED TO SCOPY BY GCL 03/25/92
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      SUBROUTINE SCOPYMN(N,DX,INCX,DY,INCY)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION DX(1),DY(1)
C
      IF(N.LE.0)RETURN
      IF(INCX.EQ.1.AND.INCY.EQ.1)GO TO 20
C
C        CODE FOR UNEQUAL INCREMENTS OR EQUAL INCREMENTS
C          NOT EQUAL TO 1
C
      IX = 1
      IY = 1
      IF(INCX.LT.0)IX = (-N+1)*INCX + 1
      IF(INCY.LT.0)IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
        DY(IY) = DX(IX)
        IX = IX + INCX
        IY = IY + INCY
   10 CONTINUE
      RETURN
C
C        CODE FOR BOTH INCREMENTS EQUAL TO 1
C
C
C        CLEAN-UP LOOP
C
   20 M = MOD(N,7)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DY(I) = DX(I)
   30 CONTINUE
      IF( N .LT. 7 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,7
        DY(I) = DX(I)
        DY(I + 1) = DX(I + 1)
        DY(I + 2) = DX(I + 2)
        DY(I + 3) = DX(I + 3)
        DY(I + 4) = DX(I + 4)
        DY(I + 5) = DX(I + 5)
        DY(I + 6) = DX(I + 6)
   50 CONTINUE
      RETURN
      END
C
C   * DSCALMN
      SUBROUTINE DSCALMN(N,DA,DX,INCX)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C
C     SCALES A VECTOR BY A CONSTANT.
C     USES UNROLLED LOOPS FOR INCREMENT EQUAL TO ONE.
C     JACK DONGARRA, LINPACK, 3/11/78.
      DIMENSION DX(*)
      IF(N.LE.0)RETURN
      IF(INCX.EQ.1)GO TO 20
C
C        CODE FOR INCREMENT NOT EQUAL TO 1
C
      NINCX = N*INCX
      DO 10 I = 1,NINCX,INCX
        DX(I) = DA*DX(I)
   10 CONTINUE
      RETURN
C
C        CODE FOR INCREMENT EQUAL TO 1
C
C
C        CLEAN-UP LOOP
C
   20 M = MOD(N,5)
      IF( M .EQ. 0 ) GO TO 40
      DO 30 I = 1,M
        DX(I) = DA*DX(I)
   30 CONTINUE
      IF( N .LT. 5 ) RETURN
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
        DX(I) = DA*DX(I)
        DX(I + 1) = DA*DX(I + 1)
        DX(I + 2) = DA*DX(I + 2)
        DX(I + 3) = DA*DX(I + 3)
        DX(I + 4) = DA*DX(I + 4)
   50 CONTINUE
      RETURN
      END
C
C   * DOTMN
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     DOT PRODUCT
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      FUNCTION DOTMN(X,Y,N)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION X(*), Y(*)
      DOTMN=DDOTMN(N,X,1,Y,1)
      RETURN
      END
C
C   * DDOTMN
      DOUBLE PRECISION FUNCTION DDOTMN(N,DX,INCX,DY,INCY)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION DX(*),DY(*)
C  Purpose
C  =======
C     forms the dot product of two vectors.
C     uses unrolled loops for increments equal to one.
C     jack dongarra, linpack, 3/11/78.
C     modified 12/3/93, array(1) declarations changed to array(*)
      INTRINSIC MOD
      DDOTMN = 0.0d0
      DTEMP = 0.0d0
      IF (N.LE.0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) GO TO 20
      IX = 1
      IY = 1
      IF (INCX.LT.0) IX = (-N+1)*INCX + 1
      IF (INCY.LT.0) IY = (-N+1)*INCY + 1
      DO 10 I = 1,N
          DTEMP = DTEMP + DX(IX)*DY(IY)
          IX = IX + INCX
          IY = IY + INCY
   10 CONTINUE
      DDOTMN = DTEMP
      RETURN
   20 M = MOD(N,5)
      IF (M.EQ.0) GO TO 40
      DO 30 I = 1,M
          DTEMP = DTEMP + DX(I)*DY(I)
   30 CONTINUE
      IF (N.LT.5) GO TO 60
   40 MP1 = M + 1
      DO 50 I = MP1,N,5
          DTEMP = DTEMP + DX(I)*DY(I) + DX(I+1)*DY(I+1) +
     +            DX(I+2)*DY(I+2) + DX(I+3)*DY(I+3) + DX(I+4)*DY(I+4)
   50 CONTINUE
   60 DDOTMN = DTEMP
      RETURN
      END
C
C   * DSWAPMN
      SUBROUTINE DSWAPMN(N,DX,INCX,DY,INCY)
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
c
c     interchanges two vectors.
c     uses unrolled loops for increments equal one.
c     jack dongarra, linpack, 3/11/78.
c     modified 12/3/93, array(1) declarations changed to array(*)
c
      DIMENSION        dx(*),dy(*)
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c       code for unequal increments or equal increments not equal
c         to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dx(ix)
        dx(ix) = dy(iy)
        dy(iy) = dtemp
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c       code for both increments equal to 1
c
c
c       clean-up loop
c
   20 m = mod(n,3)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dx(i)
        dx(i) = dy(i)
        dy(i) = dtemp
   30 continue
      if( n .lt. 3 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,3
        dtemp = dx(i)
        dx(i) = dy(i)
        dy(i) = dtemp
        dtemp = dx(i + 1)
        dx(i + 1) = dy(i + 1)
        dy(i + 1) = dtemp
        dtemp = dx(i + 2)
        dx(i + 2) = dy(i + 2)
        dy(i + 2) = dtemp
   50 continue
      return
      end
c <-- MN solvation models
c $Id$
