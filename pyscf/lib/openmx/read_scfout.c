/**********************************************************************
  read_scfout.c:

     read_scfout.c is a subroutine to read a binary file,
     filename.scfout.

  Log of read_scfout.c:

     2/July/2003  Released by T.Ozaki 

***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "read_scfout.h"

#define MAX_LINE_SIZE 256
#define fp_bsize         1048576     /* buffer size for setvbuf */

static void Input( FILE *fp );

void read_scfout(char *argv[])
{
  static FILE *fp;
  char buf[fp_bsize];          /* setvbuf */

  if ((fp = fopen(argv[1],"r")) != NULL){

#ifdef xt3
    setvbuf(fp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    printf("\nRead the scfout file (%s)\n",argv[1]);fflush(stdout);
    Input(fp);
    fclose(fp);
  }
  else {
    printf("Failure of reading the scfout file (%s).\n",argv[1]);fflush(stdout);
  }

}


void Input( FILE *fp )
{
  static int Gc_AN,ct_AN,h_AN,i,j,can,Gh_AN;
  static int wan1,wan2,TNO1,TNO2,spin,Rn,num_lines;
  static int k,q_AN,Gq_AN;
  static int i_vec[20],*p_vec;
  static double d_vec[20];
  static char makeinp[100];
  static char strg[MAX_LINE_SIZE];
  FILE *fp_makeinp;
  char buf[fp_bsize];          /* setvbuf */

  /****************************************************
     atomnum
     spinP_switch 
  ****************************************************/

  fread(i_vec,sizeof(int),6,fp);
  atomnum      = i_vec[0];
  SpinP_switch = i_vec[1];
  Catomnum =     i_vec[2];
  Latomnum =     i_vec[3];
  Ratomnum =     i_vec[4];
  TCpyCell =     i_vec[5];

  /****************************************************
    allocation of arrays:

    double atv[TCpyCell+1][4];
  ****************************************************/

  atv = (double**)malloc(sizeof(double*)*(TCpyCell+1));
  for (Rn=0; Rn<=TCpyCell; Rn++){
    atv[Rn] = (double*)malloc(sizeof(double)*4);
  }

  /****************************************************
                read atv[TCpyCell+1][4];
  ****************************************************/

  for (Rn=0; Rn<=TCpyCell; Rn++){
    fread(atv[Rn],sizeof(double),4,fp);
  }  

  /****************************************************
    allocation of arrays:

    int atv_ijk[TCpyCell+1][4];
  ****************************************************/

  atv_ijk = (int**)malloc(sizeof(int*)*(TCpyCell+1));
  for (Rn=0; Rn<=TCpyCell; Rn++){
    atv_ijk[Rn] = (int*)malloc(sizeof(int)*4);
  }

  /****************************************************
            read atv_ijk[TCpyCell+1][4];
  ****************************************************/

  for (Rn=0; Rn<=TCpyCell; Rn++){
    fread(atv_ijk[Rn],sizeof(int),4,fp);
  }  

  /****************************************************
    allocation of arrays:

    int Total_NumOrbs[atomnum+1];
    int FNAN[atomnum+1];
  ****************************************************/

  Total_NumOrbs = (int*)malloc(sizeof(int)*(atomnum+1));
  FNAN = (int*)malloc(sizeof(int)*(atomnum+1));

  /****************************************************
         the number of orbitals in each atom
  ****************************************************/

  p_vec = (int*)malloc(sizeof(int)*atomnum);
  fread(p_vec,sizeof(int),atomnum,fp);
  Total_NumOrbs[0] = 1;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    Total_NumOrbs[ct_AN] = p_vec[ct_AN-1];
  }
  free(p_vec);

  /****************************************************
   FNAN[]:
   the number of first nearest neighbouring atoms
  ****************************************************/

  p_vec = (int*)malloc(sizeof(int)*atomnum);
  fread(p_vec,sizeof(int),atomnum,fp);
  FNAN[0] = 0;
  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    FNAN[ct_AN] = p_vec[ct_AN-1];
  }
  free(p_vec);

  /****************************************************
    allocation of arrays:

    int natn[atomnum+1][FNAN[ct_AN]+1];
    int ncn[atomnum+1][FNAN[ct_AN]+1];
  ****************************************************/

  natn = (int**)malloc(sizeof(int*)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    natn[ct_AN] = (int*)malloc(sizeof(int)*(FNAN[ct_AN]+1));
  }

  ncn = (int**)malloc(sizeof(int*)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    ncn[ct_AN] = (int*)malloc(sizeof(int)*(FNAN[ct_AN]+1));
  }

  /****************************************************
    natn[][]:
    grobal index of neighboring atoms of an atom ct_AN
   ****************************************************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp);
  }  

  /****************************************************
    ncn[][]:
    grobal index for cell of neighboring atoms
    of an atom ct_AN
  ****************************************************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp);
  }

  /****************************************************
    tv[4][4]:
    unit cell vectors in Bohr
  ****************************************************/

  fread(tv[1],sizeof(double),4,fp);
  fread(tv[2],sizeof(double),4,fp);
  fread(tv[3],sizeof(double),4,fp);

  /****************************************************
    rtv[4][4]:
    unit cell vectors in Bohr
  ****************************************************/

  fread(rtv[1],sizeof(double),4,fp);
  fread(rtv[2],sizeof(double),4,fp);
  fread(rtv[3],sizeof(double),4,fp);

  /****************************************************
    Gxyz[][1-3]:
    atomic coordinates in Bohr
  ****************************************************/

  Gxyz = (double**)malloc(sizeof(double*)*(atomnum+1));
  for (i=0; i<(atomnum+1); i++){
    Gxyz[i] = (double*)malloc(sizeof(double)*60);
  }

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    fread(Gxyz[ct_AN],sizeof(double),4,fp);
  }  

  /****************************************************
    allocation of arrays:

    Kohn-Sham Hamiltonian

     dooble Hks[SpinP_switch+1]
               [atomnum+1]
               [FNAN[ct_AN]+1]
               [Total_NumOrbs[ct_AN]]
               [Total_NumOrbs[h_AN]];

    Overlap matrix

     dooble OLP[atomnum+1]
               [FNAN[ct_AN]+1]
               [Total_NumOrbs[ct_AN]]
               [Total_NumOrbs[h_AN]]; 

    Overlap matrix with position operator x, y, z

     dooble OLPpox,y,z
                 [atomnum+1]
                 [FNAN[ct_AN]+1]
                 [Total_NumOrbs[ct_AN]]
                 [Total_NumOrbs[h_AN]]; 

    Density matrix

     dooble DM[SpinP_switch+1]
              [atomnum+1]
              [FNAN[ct_AN]+1]
              [Total_NumOrbs[ct_AN]]
              [Total_NumOrbs[h_AN]];
  ****************************************************/

  Hks = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){

    Hks[spin] = (double****)malloc(sizeof(double***)*(atomnum+1));
    for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
      TNO1 = Total_NumOrbs[ct_AN];
      Hks[spin][ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        Hks[spin][ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

        if (ct_AN==0){ 
          TNO2 = 1;
	}
        else{ 
          Gh_AN = natn[ct_AN][h_AN];
          TNO2 = Total_NumOrbs[Gh_AN];
	}
        for (i=0; i<TNO1; i++){
          Hks[spin][ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
        }
      }
    }
  }

  iHks = (double*****)malloc(sizeof(double****)*3);
  for (spin=0; spin<3; spin++){

    iHks[spin] = (double****)malloc(sizeof(double***)*(atomnum+1));
    for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
      TNO1 = Total_NumOrbs[ct_AN];
      iHks[spin][ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        iHks[spin][ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

        if (ct_AN==0){ 
          TNO2 = 1;
	}
        else{ 
          Gh_AN = natn[ct_AN][h_AN];
          TNO2 = Total_NumOrbs[Gh_AN];
	}
        for (i=0; i<TNO1; i++){
          iHks[spin][ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
          for (j=0; j<TNO2; j++) iHks[spin][ct_AN][h_AN][i][j] = 0.0;
        }
      }
    }
  }

  OLP = (double****)malloc(sizeof(double***)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    OLP[ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      OLP[ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

      if (ct_AN==0){ 
        TNO2 = 1;
      }
      else{ 
        Gh_AN = natn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
      }
      for (i=0; i<TNO1; i++){
        OLP[ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
      }
    }
  }


  OLPpox = (double****)malloc(sizeof(double***)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    OLPpox[ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      OLPpox[ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

      if (ct_AN==0){ 
        TNO2 = 1;
      }
      else{ 
        Gh_AN = natn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
      }
      for (i=0; i<TNO1; i++){
        OLPpox[ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
      }
    }
  }

  OLPpoy = (double****)malloc(sizeof(double***)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    OLPpoy[ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      OLPpoy[ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

      if (ct_AN==0){ 
        TNO2 = 1;
      }
      else{ 
        Gh_AN = natn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
      }
      for (i=0; i<TNO1; i++){
        OLPpoy[ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
      }
    }
  }

  OLPpoz = (double****)malloc(sizeof(double***)*(atomnum+1));
  for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    OLPpoz[ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      OLPpoz[ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

      if (ct_AN==0){ 
        TNO2 = 1;
      }
      else{ 
        Gh_AN = natn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
      }
      for (i=0; i<TNO1; i++){
        OLPpoz[ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
      }
    }
  }

  DM = (double*****)malloc(sizeof(double****)*(SpinP_switch+1));
  for (spin=0; spin<=SpinP_switch; spin++){

    DM[spin] = (double****)malloc(sizeof(double***)*(atomnum+1));
    for (ct_AN=0; ct_AN<=atomnum; ct_AN++){
      TNO1 = Total_NumOrbs[ct_AN];
      DM[spin][ct_AN] = (double***)malloc(sizeof(double**)*(FNAN[ct_AN]+1));
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        DM[spin][ct_AN][h_AN] = (double**)malloc(sizeof(double*)*TNO1);

        if (ct_AN==0){ 
          TNO2 = 1;
	}
        else{ 
          Gh_AN = natn[ct_AN][h_AN];
          TNO2 = Total_NumOrbs[Gh_AN];
	}
        for (i=0; i<TNO1; i++){
          DM[spin][ct_AN][h_AN][i] = (double*)malloc(sizeof(double)*TNO2);
        }
      }
    }
  }

  /****************************************************
                    Hamiltonian matrix
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        Gh_AN = natn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        for (i=0; i<TNO1; i++){
          fread(Hks[spin][ct_AN][h_AN][i],sizeof(double),TNO2,fp);
        }
      }
    }
  }

  /****************************************************
  iHks:
  imaginary Kohn-Sham matrix elements of basis orbitals
  for alpha-alpha, beta-beta, and alpha-beta spin matrices
  of which contributions come from spin-orbit coupling 
  and Hubbard U effective potential.
  ****************************************************/

  if (SpinP_switch==3){
    for (spin=0; spin<3; spin++){
      for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
	TNO1 = Total_NumOrbs[ct_AN];
	for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
	  Gh_AN = natn[ct_AN][h_AN];
	  TNO2 = Total_NumOrbs[Gh_AN];
	  for (i=0; i<TNO1; i++){
	    fread(iHks[spin][ct_AN][h_AN][i],sizeof(double),TNO2,fp);
	  }
	}
      }
    }
  }

  /****************************************************
                     Overlap matrix
  ****************************************************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      for (i=0; i<TNO1; i++){
        fread(OLP[ct_AN][h_AN][i],sizeof(double),TNO2,fp);
      }
    }
  }

  /****************************************************
          Overlap matrix with position operator x
  ****************************************************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      for (i=0; i<TNO1; i++){
        fread(OLPpox[ct_AN][h_AN][i],sizeof(double),TNO2,fp);
      }
    }
  }

  /****************************************************
          Overlap matrix with position operator y
  ****************************************************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      for (i=0; i<TNO1; i++){
        fread(OLPpoy[ct_AN][h_AN][i],sizeof(double),TNO2,fp);
      }
    }
  }

  /****************************************************
          Overlap matrix with position operator z
  ****************************************************/

  for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
    TNO1 = Total_NumOrbs[ct_AN];
    for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
      Gh_AN = natn[ct_AN][h_AN];
      TNO2 = Total_NumOrbs[Gh_AN];
      for (i=0; i<TNO1; i++){
        fread(OLPpoz[ct_AN][h_AN][i],sizeof(double),TNO2,fp);
      }
    }
  }

  /****************************************************
                    Density matrix
  ****************************************************/

  for (spin=0; spin<=SpinP_switch; spin++){
    for (ct_AN=1; ct_AN<=atomnum; ct_AN++){
      TNO1 = Total_NumOrbs[ct_AN];
      for (h_AN=0; h_AN<=FNAN[ct_AN]; h_AN++){
        Gh_AN = natn[ct_AN][h_AN];
        TNO2 = Total_NumOrbs[Gh_AN];
        for (i=0; i<TNO1; i++){
          fread(DM[spin][ct_AN][h_AN][i],sizeof(double),TNO2,fp);
        }
      }
    }
  }

  /****************************************************
      Solver
  ****************************************************/

  fread(i_vec,sizeof(int),1,fp);
  Solver = i_vec[0];

  /****************************************************
      ChemP
      Temp
  ****************************************************/

  fread(d_vec,sizeof(double),10,fp);
  ChemP  = d_vec[0];
  E_Temp = d_vec[1];
  dipole_moment_core[1] = d_vec[2]; 
  dipole_moment_core[2] = d_vec[3]; 
  dipole_moment_core[3] = d_vec[4]; 
  dipole_moment_background[1] = d_vec[5]; 
  dipole_moment_background[2] = d_vec[6]; 
  dipole_moment_background[3] = d_vec[7]; 
  Valence_Electrons = d_vec[8]; 
  Total_SpinS = d_vec[9];

  /****************************************************
      input file 
  ****************************************************/

  fread(i_vec, sizeof(int), 1, fp);
  num_lines = i_vec[0];

  sprintf(makeinp,"temporal_12345.input");

  if ((fp_makeinp = fopen(makeinp,"w")) != NULL){

#ifdef xt3
    setvbuf(fp_makeinp,buf,_IOFBF,fp_bsize);  /* setvbuf */
#endif

    for (i=1; i<=num_lines; i++){
      fread(strg, sizeof(char), MAX_LINE_SIZE, fp);
      fprintf(fp_makeinp,"%s",strg);
    }

    fclose(fp_makeinp);
  }
  else{
    printf("error in making temporal_12345.input\n"); 
  }

}





