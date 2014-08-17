#ifndef CX_LEBEDEV_GRID_H
#define CX_LEBEDEV_GRID_H

namespace ct {

   unsigned const
      nAngularGrids = 33;

   struct FAngularGridEntry {
      unsigned
         // maximum L of spherical harmonics still integrated exactly.
         MaxL,
         // number of points in the angular grid
         nPoints;
      bool
         // true if the grid contains negative weights.
         // these are inappropriate for some uses. Happens, e.g., for the
         // MaxL=17   110-point grid. There may be other grids available with
         // more points and the same L, but I do not have them atm.
         HasNegativeWeights;
   };
   extern FAngularGridEntry
      AngularGridInfo[nAngularGrids];

   // writes (x,y,z,weight) pairs to pOut[0...nPoints-1]. Returns either
   // the actual number of points (==nPoints) or 0 if no grid of the requested
   // size was found.
   //
   // Note:
   //   - The grid weights integrate to 1.0, not to 4pi!
   //   - All grids have octahedral symmetry.
   unsigned MakeAngularGrid(double (*pOut)[4], unsigned nPoints);
}

/*

ccgk: This code generates Lebedev grids. It is based on C files from
ccgk: Dmitri Laikov, which were converted to Fortran by Christoph van Wuellen.
ccgk: I (Gerald Knizia) subsequently converted them back to C++.
ccgk:
ccgk: The original distribution contained the following readme file:
ccgk:

      Lebedev grids of orders n=6m+5 where m=0,1,...,21 in 16 digit precision
      =======================================================================

      The file Lebedev-Laikov.F implements a set of subroutines providing 
      Lebedev-Laikov grids of order n=2m+1, where m=1,2,...,15, and additionally
      grids of order n=6m+5, where m=5,6,...,21. The parameters ensure 
      that angular integration of polynomials x**k * y**l * z**m, where k+l+m <= 131 
      can be performed with a relative accuracy of 2e-14 [1]. Note that the weights
      are normalised to add up to 1.0.

      For each order n a separate subroutine is provided named 
      LD. The parameters X, Y, Z are arrays for the 
      cartesian components of each point, and the parameter W is an array for the
      weights. The subroutines increase the integer parameter N by number of grid
      points generated. All these routines use the subroutine gen_oh which takes care 
      of the octahedral symmetry of the grids.

      Christoph van Wuellen (Ruhr-Universitaet, Bochum, Germany) generated the 
      routines in Lebedev-Laikov.F by translating the original C-routines kindly 
      provided by Dmitri Laikov (Moscow State University, Moscow, Russia). We 
      are in debt to Dmitri Laikov for giving us permission to make these routines
      publically available.

      Huub van Dam
      Daresbury Laboratory, Daresbury, United Kingdom
      April, 2000

      References
      ==========

      [1] V.I. Lebedev, and D.N. Laikov
         "A quadrature formula for the sphere of the 131st
         algebraic order of accuracy"
         Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.

ccgk: and the following comments and references for the original of the subroutine SphGenOh:

      chvd
      chvd   This subroutine is part of a set of subroutines that generate
      chvd   Lebedev grids [1-6] for integration on a sphere. The original 
      chvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and 
      chvd   translated into fortran by Dr. Christoph van Wuellen.
      chvd   This subroutine was translated from C to fortran77 by hand.
      chvd
      chvd   Users of this code are asked to include reference [1] in their
      chvd   publications, and in the user- and programmers-manuals 
      chvd   describing their codes.
      chvd
      chvd   This code was distributed through CCL (http://www.ccl.net/).
      chvd
      chvd   [1] V.I. Lebedev, and D.N. Laikov
      chvd       "A quadrature formula for the sphere of the 131st
      chvd        algebraic order of accuracy"
      chvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
      chvd
      chvd   [2] V.I. Lebedev
      chvd       "A quadrature formula for the sphere of 59th algebraic
      chvd        order of accuracy"
      chvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286. 
      chvd
      chvd   [3] V.I. Lebedev, and A.L. Skorokhodov
      chvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
      chvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592. 
      chvd
      chvd   [4] V.I. Lebedev
      chvd       "Spherical quadrature formulas exact to orders 25-29"
      chvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107. 
      chvd
      chvd   [5] V.I. Lebedev
      chvd       "Quadratures on a sphere"
      chvd       Computational Mathematics and Mathematical Physics, Vol. 16,
      chvd       1976, pp. 10-24. 
      chvd
      chvd   [6] V.I. Lebedev
      chvd       "Values of the nodes and weights of ninth to seventeenth 
      chvd        order Gauss-Markov quadrature formulae invariant under the
      chvd        octahedron group with inversion"
      chvd       Computational Mathematics and Mathematical Physics, Vol. 15,
      chvd       1975, pp. 44-51.
      chvd
      cvw
      cvw    Given a point on a sphere (specified by a and b), generate all
      cvw    the equivalent points under Oh symmetry, making grid points with
      cvw    weight v.
      cvw    The variable num is increased by the number of different points
      cvw    generated.
      cvw
      cvw    Depending on code, there are 6...48 different but equivalent
      cvw    points.
      cvw
      cvw    code=1:   (0,0,1) etc                                (  6 points)
      cvw    code=2:   (0,a,a) etc, a=1/sqrt(2)                   ( 12 points)
      cvw    code=3:   (a,a,a) etc, a=1/sqrt(3)                   (  8 points)
      cvw    code=4:   (a,a,b) etc, b=sqrt(1-2 a^2)               ( 24 points)
      cvw    code=5:   (a,b,0) etc, b=sqrt(1-a^2), a input        ( 24 points)
      cvw    code=6:   (a,b,c) etc, c=sqrt(1-a^2-b^2), a/b input  ( 48 points)
      cvw

*/





#endif // CX_LEBEDEV_GRID_H
