c copied from https://github.com/nwchemgit/nwchem/blob/master/src/solvation/mnsol_mem.F
c
c     memory for the x array used in mnsol code
c
      subroutine mnsol_xmem(nat,ineed)
c
      implicit none
c
      integer nat
c
      integer LRAD
      integer LCDSA
      integer LAREA
      integer LDATAR
      integer LSTS
      integer LCOT
      integer LDSTS
      integer LDCOTDR
      integer LNC
      integer LDAREA
      integer LRLIO
      integer LURLIO
      integer LLAB
      integer LNCNCT
      integer LCONECT
      integer LCTHETA
      integer LSTHETA
      integer LSIT
      integer LDCSIT
      integer LDJCOSN
      integer LCOSN
      integer LDSTETA
      integer LDCTETA
      integer LDCOSN
      integer LWORK
      integer LDIWORK
      integer LDJWORK
      integer LDKWORK
      integer LD0WORK
      integer LDWORKR
      integer LDCAODD
      integer LDSITR
      integer LDCAPLY
      integer LDICOSN
      integer LDCASLC
      integer LDCTETR
      integer LDSTETR
      integer LDCOSNR
      integer LEND
      integer INEED
c
c     calculate memory
c
      LRAD = 1
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
c
      return
      end
c $Id$
