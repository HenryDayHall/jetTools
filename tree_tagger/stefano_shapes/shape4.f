c Contents;
c  - shape4
c  - order4_p (used in shape4)
c  - pxtth3 (get principle thrust axis)
c  - pxanxy (get azimuthal angle)
c  - pxrmx3  
c  - pxrob3
c  - pxrof3 
c  - pxlut3  
c  - pxplu3
c  - pxlth4
c  - ... 
c This one really dosn't seem to cooporate with python hooks
c gives bus error or segfault, depending on input
c will hae to be called as a subprocess instead
c compile with;
c     $ f77 shape2.f shape3.f shape4.f -o shape
c run with;
c     $ ./shape s p3(1) p3(2)... p4(1)... [p5(1)... [p6(1)...]]
      subroutine shape4(s,p3,p4,p5,p6,
     &                  thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan,
     &                  acopla,EE,THET,rmin,rmax,Dpar,sve)
      implicit real*8 (a-h,o-z)
      real*8 ljm
      common/taxis4/tv4(3)
      real*8, intent(in) :: s
      real*8, dimension(4), intent(in) :: p3,p4,p5,p6
      dimension tm1(4),tm2(4),tm3(4),tm4(4)
      dimension tmul1(4),tmul2(4),tmul3(4),tmul4(4)
      dimension ptrak(3,4),qtrak(1000,5),rtrak(4,4)
      dimension THRVAL(3),THRVEC(3,3),EVAL(3),EVEC(3,3) 
      dimension TVEC(3),AKOVEC(3)
      dimension tmm(3)
      dimension ahjm(4),bhjm(4)
      dimension T(3,3)
      real*8, dimension(4, 4), intent(out) :: EE,THET
      parameter (nmax=8,ia=nmax,iv=nmax)
      dimension a(ia,nmax),e(nmax),rr(nmax),v(iv,nmax)
      parameter (pi=3.141592653589793238d0)
      intent(out) :: thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan
      intent(out) :: acopla,rmin,rmax,Dpar,sve
c ordino impulsi secondo il modulo (tmd1>tmd2>tmd3>tmd4).
      call order4_p(p3,p4,p5,p6,tm1,tm2,tm3,tm4,tmd1,tmd2,tmd3,tmd4)
c serve per il thrust (tmm, tw asse di thrust).
      do i=1,3
        ptrak(i,1)=p3(i)
        ptrak(i,2)=p4(i)
        ptrak(i,3)=p5(i)
        ptrak(i,4)=p6(i)
      end do
      do i=1,3
        qtrak(1,i)=p3(i)
        qtrak(2,i)=p4(i)
        qtrak(3,i)=p5(i)
        qtrak(4,i)=p6(i)
      end do
      do i=1,4
        rtrak(i,1)=p3(i)
        rtrak(i,2)=p4(i)
        rtrak(i,3)=p5(i)
        rtrak(i,4)=p6(i)
      end do
      call PXTTH3(4,3,ptrak,thrust,tmm,IERR)
c  Pritingin like this gets through to python dict
      print *, "ThrustVector[1] ", tmm(1)
      print *, "ThrustVector[2] ", tmm(2)
      print *, "ThrustVector[3] ", tmm(3)
      do i=1,3
        tv4(i)=tmm(i)
      end do
      tmodm=sqrt(abs(tmm(1)*tmm(1)+tmm(2)*tmm(2)+tmm(3)*tmm(3)))
c serve per oblateness.
      call PXLUT3(4,1000,qtrak,thrust,oblateness)
c calcolo thrust.
      thr=thrust
c calcolo oblateness.
      obl=oblateness
c serve per minor and major
      call PXLTH4(4,4,rtrak,THRVAL,THRVEC,IERR)
c  Pritingin like this gets through to python dict
      print *, "Principle[1] ", THRVEC(1, 3)
      print *, "Principle[2] ", THRVEC(2, 3)
      print *, "Principle[3] ", THRVEC(3, 3)
      print *, "Major[1] ", THRVEC(1, 2)
      print *, "Major[2] ", THRVEC(2, 2)
      print *, "Major[3] ", THRVEC(3, 2)
      print *, "Minor[1] ", THRVEC(1, 1)
      print *, "Minor[2] ", THRVEC(2, 1)
      print *, "Minor[3] ", THRVEC(3, 1)
      rmax=THRVAL(2)
      rmin=THRVAL(1)      
c serve per C-, D-parameter.
      call PXLSP3(4,3,ptrak,EVAL,EVEC,IERR)
c calcolo in C- and D-parameter.
      Cpar=3.d0*(EVAL(1)*EVAL(2)+EVAL(2)*EVAL(3)+EVAL(3)*EVAL(1))
      Dpar=27.d0*EVAL(1)*EVAL(2)*EVAL(3)
c serve per sphericity, aplanarity e planarity.
      call PXJSP3(4,3,ptrak,EVAL,EVEC,IERR)
c calcolo sphericity.
      sfe=3.d0*(eval(1)+eval(2))/2.d0
c calcolo aplanarity.
      apla=3.d0/2.d0*eval(1)
c calcolo planarity.
      plan=eval(2)-eval(1)
c calcolo heavy and light jet mass squared.
      call PXMMBB(4,4,rtrak,tmm,AMH,AML,BT,BW,IERR)
      hjm=AMH*AMH
      ljm=AML*AML
c calcolo difference jet mass squared.
      djm=abs(hjm-ljm)
c calcolo acoplanarity.
      call PXAKO4(4,4,100,rtrak,80,AKOPL,AKOVEC,IERR) 
      acopla=AKOPL
c calcolo spherocity.
      x1=2.d0*p3(4)/sqrt(s)
      x2=2.d0*p4(4)/sqrt(s)
      xg=2.d0*(p5(4)+p6(4))/sqrt(s)
      eps=(p5(4)+p6(4))**2
      do i=1,3
        eps=eps-(p5(i)+p6(i))**2
      end do
      eps=eps/s
      xx=thr*thr
      sve=(4.d0/pi)**2
     &   *(4.d0/xx*(1.d0-x1)*(1.d0-x2)*(1.d0-xg)
     &    -2.d0*eps/xx*(x1*x1+x2*x2-xg*xg) 
     &    -4.d0*eps*eps/xx)       
c EEC calcolata nel main.
      return 
      end
c
c     ----------------------------------------------------------------------
c
      subroutine order4_p(a,b,c,d,aa,bb,cc,dd,pmodaa,pmodbb,pmodcc,
     &                    pmoddd)
      implicit real*8 (a-h,o-z)
      dimension a(4),b(4),c(4),d(4)
      dimension aa1(4),bb1(4),cc1(4),dd1(4)
      dimension aa2(4),bb2(4),cc2(4),dd2(4)
      dimension aa3(4),bb3(4),cc3(4),dd3(4)
      dimension aa4(4),bb4(4),cc4(4),dd4(4)
      dimension aa(4),bb(4),cc(4),dd(4)
      pmoda=sqrt(abs(a(1)*a(1)+a(2)*a(2)+a(3)*a(3)))
      pmodb=sqrt(abs(b(1)*b(1)+b(2)*b(2)+b(3)*b(3)))
      pmodc=sqrt(abs(c(1)*c(1)+c(2)*c(2)+c(3)*c(3)))
      do i=1,4
        aa1(i)=a(i)
      end do
      if(pmodb.gt.pmodaa1)then
        pmodaa2=pmodb
        pmodbb2=pmodaa1
        do i=1,4
          aa2(i)=b(i)
          bb2(i)=aa1(i)
        end do
      else
        pmodaa2=pmodaa1
        pmodbb2=pmodb
        do i=1,4
      	  aa2(i)=aa1(i)	
          bb2(i)=b(i)
        end do
      end if
      if(pmodc.gt.pmodaa2)then
        pmodaa3=pmodc
        pmodbb3=pmodaa2
        pmodcc3=pmodbb2
        do i=1,4
          aa3(i)=c(i)
          bb3(i)=aa2(i)
          cc3(i)=bb2(i)
        end do
      else
        if(pmodc.gt.pmodbb2)then
        pmodaa3=pmodaa2
        pmodbb3=pmodc
        pmodcc3=pmodbb2
          do i=1,4
            aa3(i)=aa2(i)
            bb3(i)=c(i)
            cc3(i)=bb2(i)
          end do
        else
        pmodaa3=pmodaa2
        pmodbb3=pmodbb2
        pmodcc3=pmodc
          do i=1,4
            aa3(i)=aa2(i)
            bb3(i)=bb2(i)
            cc3(i)=c(i)
          end do
        end if
      end if
      if(pmodd.gt.pmodaa3)then
        pmodaa4=pmodd
        pmodbb4=pmodaa3
        pmodcc4=pmodbb3
        pmoddd4=pmodcc3
        do i=1,4
          aa4(i)=d(i)
          bb4(i)=aa3(i)
          cc4(i)=bb3(i)
          dd4(i)=cc3(i)
        end do
      else
        if(pmodd.gt.pmodbb3)then
          pmodaa4=pmodaa3
          pmodbb4=pmodd
          pmodcc4=pmodbb3
          pmoddd4=pmodcc3
          do i=1,4
            aa4(i)=aa3(i)
            bb4(i)=d(i)
            cc4(i)=bb3(i)
            dd4(i)=cc3(i)
          end do
        else
          if(pmodd.gt.pmodcc3)then  
            pmodaa4=pmodaa3
            pmodbb4=pmodbb3
            pmodcc4=pmodd
            pmoddd4=pmodcc3
            do i=1,4
              aa4(i)=aa3(i)
              bb4(i)=bb3(i)
              cc4(i)=d(i)
              dd4(i)=cc3(i)
            end do
          else
            pmodaa4=pmodaa3
            pmodbb4=pmodbb3
            pmodcc4=pmodcc3
            pmoddd4=pmodd
            do i=1,4
              aa4(i)=aa3(i)
              bb4(i)=bb3(i)
              cc4(i)=cc3(i)
              dd4(i)=d(i)
            end do
          end if
        end if
      end if
      pmodaa=pmodaa4
      pmodbb=pmodbb4
      pmodcc=pmodcc4
      pmoddd=pmoddd4
      do i=1,4
        aa(i)=aa4(i)
        bb(i)=bb4(i)
        cc(i)=cc4(i)
        dd(i)=dd4(i)
      end do
      return
      end
c
c     ----------------------------------------------------------------------
c
      SUBROUTINE PXTTH3 (NTRAK,ITKDM,PTRAK,THR,TVEC,IERR)
*.*********************************************************
*. ------
*. PXTTH3
*. ------
*. Routine to determine the principle thrust axis and
*. thrust value of an event using the Tasso algorithm.
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRAK
*.      PARAMETER  (ITKDM=3.or.more,MXTRAK=1.or.more)
*.      INTEGER*4 NTRAK,IERR
*.      REAL*8  PTRAK (ITKDM,MXTRAK),
*.     +      TVEC (3.or.more)
*.      REAL*8  THR
*.
*.      NTRAK = 1.to.MXTRAK
*.      CALL  PXTTH3 (NTRAK,ITKDM,PTRAK,THR,TVEC,IERR)
*.
*. INPUT     : NTRAK    Total number of particles
*. INPUT     : ITKDM    First dimension of PTRAK array
*. INPUT     : PTRAK    Particle momentum array: Px,Py,Pz
*. OUTPUT    : THR      The thrust value
*. OUTPUT    : TVEC     The thrust vector
*. OUTPUT    : IERR     = 0 if all is OK ;   = -1 otherwise
*.
*. CALLS     : none
*. CALLED    : By User
*.
*. AUTHOR    :  D. Lueke (Tasso)
*. CREATED   :  25-Jun-79
*. LAST MOD  :  14-Nov-84
*.
*. Modification Log.
*. 01-Jul-88   J.W.Gary  Integrate into PX library
*.
*.*********************************************************
      INTEGER*4  IFIRST,ITKDM,NTRAK,NA1,IERR,
     +         I,J,K,L,M,K1,IX
      REAL*8  PTRAK (ITKDM,4),PTOT (3),PT (3),PTM (3),
     +      PC (3,4),TVEC (3)
      REAL*8  T,U,TMAX,THR,PP,SP
      REAL*8  AX,BX
      DATA  IFIRST / 0 /
 
      IERR = 0
*  no particles, skip event
*  -- ---------  ---- -----
      IF (NTRAK.LT.0) THEN
          IERR = -1
          GO TO 300
      END IF
*  one particle, buffer to output
*  --- --------  ------ -- ------
      IF (NTRAK.EQ.1) THEN
          THR = 1.
          TVEC (1) = PTRAK (1,1)
          TVEC (2) = PTRAK (2,1)
          TVEC (3) = PTRAK (3,1)
          GO TO 300
      END IF
      PTM (1) = 0.
      PTM (2) = 0.
      PTM (3) = 0.
      PTOT (1) = 0.
      PTOT (2) = 0.
      PTOT (3) = 0.
      SP = 0.
      DO 20 K = 1,NTRAK
          PP = SQRT (PTRAK (1,K)**2 + PTRAK (2,K)**2
     +             + PTRAK (3,K)**2)
          SP = SP + PP
          PTOT (1) = PTOT (1) + PTRAK (1,K)
          PTOT (2) = PTOT (2) + PTRAK (2,K)
          PTOT (3) = PTOT (3) + PTRAK (3,K)
   20 CONTINUE
      PTOT (1) = 0.5 * PTOT (1)
      PTOT (2) = 0.5 * PTOT (2)
      PTOT (3) = 0.5 * PTOT (3)
      TMAX = 0.
      NA1 = 2
      DO 200 K = NA1,NTRAK
          K1 = K - 1
          DO 190  J = 1,K1
*           cross product
*           ----- -------
              TVEC (1) = PTRAK (2,J) * PTRAK (3,K)
     +                 - PTRAK (3,J) * PTRAK (2,K)
              TVEC (2) = PTRAK (3,J) * PTRAK (1,K)
     +                 - PTRAK (1,J) * PTRAK (3,K)
              TVEC (3) = PTRAK (1,J) * PTRAK (2,K)
     +                 - PTRAK (2,J) * PTRAK (1,K)
              PT (1) = -PTOT (1)
              PT (2) = -PTOT (2)
              PT (3) = -PTOT (3)
              DO 100 L = 1,NTRAK
                  IF (L.EQ.K) GO TO 100
                  IF (L.EQ.J) GO TO 100
                  U = PTRAK (1,L) * TVEC (1)
     +              + PTRAK (2,L) * TVEC (2)
     +              + PTRAK (3,L) * TVEC (3)
                  IF (U.LT.0.) GO TO 100
                  PT (1) = PT (1) + PTRAK (1,L)
                  PT (2) = PT (2) + PTRAK (2,L)
                  PT (3) = PT (3) + PTRAK (3,L)
  100         CONTINUE
              DO 110 I = 1,3
                  PC (I,1) = PT (I)
                  PC (I,2) = PT (I) + PTRAK (I,K)
                  PC (I,3) = PT (I) + PTRAK (I,J)
                  PC (I,4) = PC (I,3) + PTRAK (I,K)
  110         CONTINUE
              DO 180 M = 1,4
                  T = PC (1,M) * PC (1,M) + PC (2,M) * PC (2,M)
     +              + PC (3,M) * PC (3,M)
                  IF (T.LE.TMAX) GO TO 180
                  TMAX = T
                  PTM (1) = PC (1,M)
                  PTM (2) = PC (2,M)
                  PTM (3) = PC (3,M)
  180         CONTINUE
  190     CONTINUE
  200 CONTINUE
      THR = 2.* SQRT (TMAX) / SP
      TVEC (1) = PTM (1)
      TVEC (2) = PTM (2)
      TVEC (3) = PTM (3)
  300 CONTINUE
      AX = 0.0
      DO 320  IX = 1,3
          AX = AX + TVEC (IX) * TVEC (IX)
  320 CONTINUE
      BX = DSQRT (AX)
      IF (BX.NE.0.0) THEN
          BX = 1.0 / BX
      ELSE
          IERR = -1
          RETURN
      END IF
      DO 340 IX = 1,3
          TVEC (IX) = BX * TVEC (IX)
  340 CONTINUE
      RETURN
      END

      SUBROUTINE PXANXY (XX,YY,ANG,IERR)
*.*********************************************************
*. ------
*. PXANXY
*. ------
*. SOURCE: Jetset7.1 (T. Sjostrand)
*. Reconstruct the azimuthal angle of a vector,
*. given the X and Y components of a vector
*. Usage     :
*.
*.      INTEGER*4  IERR
*.      REAL*8  XX,YY,ANG
*.
*.      CALL PXANXY (XX,YY,ANG,IERR)
*.
*. INPUT     : XX      The X component of a vector
*. INPUT     : YY      The Y component of a vector
*. OUTPUT    : ANG     The azimuthal angle
*. OUTPUT    : IERR    = 0 if all is OK ;   = -1 otherwise
*.
*.*********************************************************
      REAL*8  PIII
      PARAMETER  (PIII=3.1415927)
      INTEGER*4  IERR
      REAL*8  XX,YY,ANG
      REAL*8  ULANGL,RRR,XXX,YYY
 
      IERR = 0
      XXX = XX
      YYY = YY
      RRR = DSQRT (XXX**2 + YYY**2)
      IF (RRR.LT.1E-20) GO TO 990
      IF ((DABS (XXX)/RRR).LT.0.8) THEN
          ULANGL = DSIGN (DACOS (XXX/RRR),YYY)
      ELSE
          ULANGL = DASIN (YYY/RRR)
          IF (XXX.LT.0..AND.ULANGL.GE.0.) THEN
              ULANGL = PIII - ULANGL
          ELSE IF (XXX.LT.0.) THEN
              ULANGL = - PIII - ULANGL
          END IF
      END IF
      ANG = ULANGL
 
      RETURN
 990  IERR = -1
      RETURN
      END

      SUBROUTINE PXRMX3 (VECT,CP,SP,RMX)
*.*********************************************************
*. ------
*. PXRMX3
*. ------
*. SOURCE: HERWIG (B.Webber,G.Marchesini)
*. Calculate the rotation matrix to get from vector VECT
*. to the Z axis, followed by a rotation PHI about the Z axis,
*. where CP, SP are the cosine and sine of PHI, respectively.
*. Usage     :
*.
*.      REAL*8  VECT (3.or.more),
*.     +      RMX (3,3.or.more)
*.      REAL*8  CP,SP
*.
*.      CALL PXRMX3 (VECT,CP,SP,RMX)
*.
*. INPUT     : VECT   The vector for which the rotation matrix
*.                    is to be calculated
*. INPUT     : CP     Cosine of phi
*. INPUT     : SP     Sine of phi
*. OUTPUT    : RMX    The rotation matrix
*.
*.*********************************************************
      REAL*8  VECT (*),RMX (3,*)
      REAL*8  CT,ST,CF,SF,PP,PT
      REAL*8  CP,SP,PTCUT
      DATA  PTCUT / 1.E-10 /
      PT = VECT (1)**2 + VECT (2)**2
      IF (PT.LT.PTCUT) THEN
         CT = DSIGN (1.d0,VECT (3))
         ST = 0.
         CF = 1.
         SF = 0.
      ELSE
         PP = SQRT (VECT (3)**2 + PT)
         PT = SQRT (PT)
         CT = VECT (3) / PP
         ST = PT / PP
         CF = VECT (1) / PT
         SF = VECT (2) / PT
      END IF
      RMX (1,1) =  (CP * CF * CT) + (SP * SF)
      RMX (1,2) =  (CP * SF * CT) - (SP * CF)
      RMX (1,3) = -(CP * ST)
      RMX (2,1) = -(CP * SF) + (SP * CF * CT)
      RMX (2,2) =  (CP * CF) + (SP * SF * CT)
      RMX (2,3) = -(SP * ST)
      RMX (3,1) =  (CF * ST)
      RMX (3,2) =  (SF * ST)
      RMX (3,3) =  CT
      RETURN
      END

      SUBROUTINE PXROB3 (RMX,VECT,RVEC)
*.*********************************************************
*. ------
*. PXROB3
*. ------
*. SOURCE: HERWIG (B.Webber,G.Marchesini)
*. Rotate 3-vector VECT by inverse of rotation matrix RMX,
*.      RVEC = (RMX)-1 * VECT
*. Usage     :
*.
*.      REAL*8  VECT (3.or.more),
*.     +      RVEC (3.or.more),
*.     +      RMX  (3,3.or.more)
*.
*.      CALL PXROB3 (RMX,VECT,RVEC)
*.
*. INPUT     : RMX    The rotation matrix
*. INPUT     : VECT   The vector to be rotated
*. OUTPUT    : RVEC   The rotated vector
*.
*.*********************************************************
      REAL*8  S1,S2,S3
      REAL*8 RMX (3,*),VECT (*),RVEC (*)
      S1 = VECT (1) * RMX (1,1) + VECT (2) * RMX (2,1)
     +   + VECT (3) * RMX (3,1)
      S2 = VECT (1) * RMX (1,2) + VECT (2) * RMX (2,2)
     +   + VECT (3) * RMX (3,2)
      S3 = VECT (1) * RMX (1,3) + VECT (2) * RMX (2,3)
     +   + VECT (3) * RMX (3,3)
      RVEC (1) = S1
      RVEC (2) = S2
      RVEC (3) = S3
      RETURN
      END

      SUBROUTINE PXROF3 (RMX,VECT,RVEC)
*.*********************************************************
*. ------
*. PXROF3
*. ------
*. SOURCE: HERWIG (B.Webber,G.Marchesini)
*. Rotate 3-vector VECT by rotation matrix RMX,
*.      RVEC = RMX * VECT
*. Usage     :
*.
*.      REAL*8  VECT (3.or.more),
*.     +      RVEC (3.or.more),
*.     +      RMX (3,3.or.more)
*.
*.      CALL PXROF3 (RMX,VECT,RVEC)
*.
*. INPUT     : RMX    The rotation matrix
*. INPUT     : VECT   The vector to be rotated
*. OUTPUT    : RVEC   The rotated vector
*.
*.*********************************************************
      REAL*8  S1,S2,S3
      REAL*8 RMX (3,*),VECT (*),RVEC (*)
      S1 = RMX (1,1) * VECT (1) + RMX (1,2) * VECT (2)
     +   + RMX (1,3) * VECT (3)
      S2 = RMX (2,1) * VECT (1) + RMX (2,2) * VECT (2)
     +   + RMX (2,3) * VECT (3)
      S3 = RMX (3,1) * VECT (1) + RMX (3,2) * VECT (2)
     +   + RMX (3,3) * VECT (3)
      RVEC (1) = S1
      RVEC (2) = S2
      RVEC (3) = S3
      RETURN
      END

      SUBROUTINE PXLUT3 (N,NRLUDM,P,THR,OBL)
*.*********************************************************
*. ------
*. PXLUT3
*. ------
*. An "in-house" version of the Jetset thrust finding algorithm
*. which works entirely through an argument list rather than
*. with e.g. the Jetset common blocks.  This routine calculates
*. the standard linear thrust vectors and values. Its operation
*. is entirely decoupled from any MST or MSTJ variables etc.
*. which might be set by a user using Jetset.
*. The main purpose of developing an in-house version of the
*. Jetset thrust algorithm is so as to have a version
*. which is compatible with both Jetset6.3 and Jetset7.1 etc.
*. (because of the change in the Jetset common blocks between
*. these two versions, the Jetset version of this thrust
*. algorithm LUTHRU is version specific).
*.
*. The Jetset thrust algorithm implements an "approximate" method
*. for thrust determination because not all particle combinations
*. are tested.  It is therefore logically possible that the thrust
*. axes and values determined by this routine will correspond
*. to local rather than to absolute maxima of the thrust function.
*. However in practice this is unlikely because several starting
*. points are used and the algorithm iterated to cross check one
*. convergence vs. another.  Thus this routine offers a very fast
*. and in practice quite accurate algorithm for thrust (much faster
*. than so-called "exact" algorithms).
*. Usage     :
*.
*.      INTEGER*4  NRLUDM
*.      PARAMETER (NRLUDM=1000.or.so)
*.      REAL*8 PLUND (NRLUDM,5)
*.      INTEGER*4  NTRAK
*.      REAL*8  THRUST,OBLATE
*.
*.      (define NTRAK, fill PLUND)
*.      CALL PXLUT3 (NTRAK,NRLUDM,PLUND,THRUST,OBLATE)
*.
*. INPUT     : NTRAK    Number of tracks
*. INPUT     : NRLUDM   First dimension of PLUND
*. INPUT     : PLUND    4-momenta in Jetset format
*. OUTPUT    : THRUST   Thrust value
*. OUTPUT    : OBLATE   Oblateness value
*.
*. CALLS     : PXANXY,PXPLU3,PXRMX3,PXROF3,PXROB3
*. CALLED    : PXLTH4
*.
*. AUTHOR    : Modified from LUTHRU (T.Sjostrand) by J.W.Gary
*. CREATED   : 31-Jan-89
*. LAST MOD  : 27-Nov-95
*.
*. Modification Log.
*. 04-Feb-89  In-house version for PX library  J.W.Gary
*. 12-Mar-89  Get rid of calls to RLU          J.W.Gary
*. 27-Nov-95  M.Schroder Clear part of the array P above tracks
*.
*.*********************************************************
      INTEGER*4  N,NP,MSTU44,MSTU45,ILC,ILD,ILF,ILG,I,J,
     +         IAGR,NC,IPP,IERR,NRLUDM
      REAL*8  P (NRLUDM,*),TDI (3),TPR (3),PVEC (3),
     +      RVEC (3),RMX (3,3)
      REAL*8  PS,PARU42,PARU48,TDS,SGN,OBL,THP,THR,RLU,
     +      THPS,SG,PHI,CP,SP
      DATA  PARU42 / 1. /, PARU48 / 0.0001 /,
     +      MSTU44 / 4  /, MSTU45 / 2 /
 
      IF(2*N+MSTU44+15.GE.NRLUDM-5) THEN
          WRITE (6,FMT='('' PXLUT3: Error, not enough buffer'',
     +           '' space for Thrust calculation'')')
          THR=-2.
          OBL=-2.
          GO TO 990
      ENDIF
C  M.Schroder (these elements are always used, but sometimes not set...)
      DO 50 I = N+1, 2*N+2
          P(I,1) = 0.
          P(I,2) = 0.
          P(I,3) = 0.
          P(I,4) = 0.
          P(I,5) = 0.
   50 CONTINUE
C...Take copy of particles that are to be considered in thrust analysis.
      NP = 0
      PS = 0.
      DO 100 I = 1,N
          NP = NP + 1
          P (N+NP,1) = P (I,1)
          P (N+NP,2) = P (I,2)
          P (N+NP,3) = P (I,3)
          P (N+NP,4) = SQRT (P (I,1)**2 +P (I,2)**2 + P (I,3)**2)
          P (N+NP,5) = 1.
          IF (ABS (PARU42-1.).GT.0.001)
     +        P (N+NP,5) = P (N+NP,4)**(PARU42-1.)
          PS = PS + P (N+NP,4) * P (N+NP,5)
  100 CONTINUE
C...Loop over thrust and major. T axis along z direction in latter case.
      DO 280 ILD=1,2
          IF (ILD.EQ.2) THEN
              CALL PXANXY (P (N+NP+1,1),P (N+NP+1,2),PHI,IERR)
              CALL PXPLU3 (N+NP+1,NRLUDM,P,PVEC,'U')
              CP = COS (PHI)
              SP = SIN (PHI)
              CALL PXRMX3 (PVEC,CP,SP,RMX)
              DO 105 IPP = N+1,N+NP+1
                  CALL PXPLU3 (IPP,NRLUDM,P,PVEC,'U')
                  CALL PXROF3 (RMX,PVEC,RVEC)
                  CALL PXPLU3 (IPP,NRLUDM,P,RVEC,'P')
  105         CONTINUE
          ENDIF
C...Find and order particles with highest p (pT for major).
          DO 110 ILF = N+NP+4,N+NP+MSTU44+4
              P (ILF,4) = 0.
  110     CONTINUE
          DO 150 I = N+1,N+NP
              IF (ILD.EQ.2) P(I,4) = SQRT (P (I,1)**2 + P (I,2)**2)
              DO 120 ILF = N+NP+MSTU44+3,N+NP+4,-1
                  IF (P (I,4).LE.P (ILF,4)) GO TO 130
                  DO 115 J=1,5
                      P(ILF+1,J)=P(ILF,J)
  115             CONTINUE
  120         CONTINUE
              ILF = N + NP + 3
  130         DO 140 J=1,5
                  P (ILF+1,J) = P (I,J)
  140         CONTINUE
  150     CONTINUE
C...Find and order initial axes with highest thrust (major).
          DO 160 ILG=N+NP+MSTU44+5,N+NP+MSTU44+15
              P(ILG,4)=0.
  160     CONTINUE
          NC = 2**(MIN (MSTU44,NP) - 1)
          DO 220 ILC=1,NC
              DO 170 J=1,3
                  TDI(J)=0.
  170         CONTINUE
              DO 180 ILF=1,MIN(MSTU44,NP)
                  SGN = P (N+NP+ILF+3,5)
                  IF (2**ILF*((ILC+2**(ILF-1)-1)/2**ILF).GE.ILC)
     +                SGN = -SGN
                  DO 175 J = 1,4-ILD
                      TDI (J) = TDI (J) + SGN * P (N+NP+ILF+3,J)
  175             CONTINUE
  180         CONTINUE
              TDS = TDI (1)**2 + TDI (2)**2 + TDI (3)**2
              DO 190 ILG = N+NP+MSTU44+MIN(ILC,10)+4,N+NP+MSTU44+5,-1
                  IF (TDS.LE.P (ILG,4)) GO TO 200
                  DO 185 J=1,4
                      P (ILG+1,J) = P (ILG,J)
  185             CONTINUE
  190         CONTINUE
              ILG=N + NP + MSTU44 + 4
  200         DO 210 J=1,3
                  P (ILG+1,J) = TDI (J)
  210         CONTINUE
              P (ILG+1,4) = TDS
  220     CONTINUE
C...Iterate direction of axis until stable maximum.
          P (N+NP+ILD,4) = 0.
          ILG = 0
  230     ILG = ILG + 1
          THP = 0.
  240     THPS = THP
          DO 250 J=1,3
              IF (THP.LE.1E-10) TDI (J) = P (N+NP+MSTU44+4+ILG,J)
              IF (THP.GT.1E-10) TDI (J) = TPR (J)
              TPR (J) = 0.
  250     CONTINUE
          DO 260 I = N+1,N+NP
              SGN = DSIGN (P(I,5),
     +                TDI(1)*P(I,1)+TDI(2)*P(I,2)+TDI(3)*P(I,3))
              DO 255 J = 1,4-ILD
                  TPR (J) = TPR (J) + SGN * P (I,J)
  255         CONTINUE
  260     CONTINUE
          THP = SQRT (TPR (1)**2 + TPR (2)**2 + TPR (3)**2) / PS
          IF (THP.GE.THPS+PARU48) GO TO 240
C...Save good axis. Try new initial axis until a number of tries agree.
          IF (THP.LT.P(N+NP+ILD,4)-PARU48.AND.ILG.LT.MIN(10,NC))
     +          GO TO 230
          IF (THP.GT.P(N+NP+ILD,4)+PARU48) THEN
              IAGR = 0
**JWG              SGN = (-1.)**INT (RLU(0)+0.5)
              SGN = 1.
              DO 270 J=1,3
                  P (N+NP+ILD,J) = SGN * TPR (J) / (PS*THP)
  270         CONTINUE
              P(N+NP+ILD,4) = THP
              P(N+NP+ILD,5) = 0.
          ENDIF
          IAGR = IAGR + 1
          IF (IAGR.LT.MSTU45.AND.ILG.LT.MIN(10,NC)) GO TO 230
  280 CONTINUE
C...Find minor axis and value by orthogonality.
**JWG      SGN = (-1.)**INT (RLU(0)+0.5)
      SGN = 1.
      P (N+NP+3,1) = -SGN * P (N+NP+2,2)
      P (N+NP+3,2) = SGN * P (N+NP+2,1)
      P (N+NP+3,3) = 0.
      THP = 0.
      DO 290 I = N+1,N+NP
          THP = THP + P (I,5)
     +        * ABS (P (N+NP+3,1) * P (I,1) + P (N+NP+3,2) * P (I,2))
  290 CONTINUE
      P (N+NP+3,4) = THP / PS
      P (N+NP+3,5) = 0.
C...Fill axis information. Rotate back to original coordinate system.
      DO 300 ILD = 1,3
          DO 295 J = 1,5
              P (N+ILD,J) = P (N+NP+ILD,J)
  295     CONTINUE
  300 CONTINUE
      DO 305 IPP = N+1,N+3
          CALL PXPLU3 (IPP,NRLUDM,P,PVEC,'U')
          CALL PXROB3 (RMX,PVEC,RVEC)
          CALL PXPLU3 (IPP,NRLUDM,P,RVEC,'P')
  305 CONTINUE
C...Select storing option. Calculate thurst and oblateness.
      THR = P (N+1,4)
      OBL = P (N+2,4) - P (N+3,4)
 
  990 RETURN
      END

      SUBROUTINE PXPLU3 (INDX,NRLUDM,PLUND,PVEC,CHAR)
*.*********************************************************
*. ------
*. PXPLU3
*. ------
*. A utility routine to repack a Jetset 3-momentum as a
*. standard ordered array or vice-versa.  This routine
*. is used to translate between the array conventions
*. of the Jetset thrust algorithm and of the other routines
*. in this library
*. Usage     :
*.
*.      CALL PXPLU3 (INDX,NRLUDM,PLUND,PVEC,CHAR)
*.
*. INPUT     : INDX     The Jetset vector number
*. IN/OUT    : NRLUDM   First argument of PLUND
*. IN/OUT    : PLUND    The Jetset 3-momentum array
*. IN/OUT    : PVEC     The input or output array
*. CONTROL   : CHAR     ='U' means unpack Jetset array
*.                      = anything else means pack Jetset array
*.
*. CALLS     : none
*. CALLED    : PXLUT3
*.
*. AUTHOR    :  J.W.Gary
*. CREATED   :  04-Feb-89
*. LAST MOD  :  04-Feb-89
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4  INDX,NRLUDM,IX
      REAL*8  PVEC (*),PLUND (NRLUDM,*)
      CHARACTER*1  CHAR
      DO 120  IX = 1,3
          IF (CHAR.EQ.'U') THEN
              PVEC (IX) = PLUND (INDX,IX)
          ELSE
              PLUND (INDX,IX) = PVEC (IX)
          END IF
 120  CONTINUE
      RETURN
      END

      SUBROUTINE PXLTH4 (NTRAK,ITKDM,PTRAK,THRVAL,THRVEC,IERR)
*.*********************************************************
*. ------
*. PXLTH4
*. ------
*. Routine to determine the Thrust Principal, Major and
*. Minor axes and values using the Jetset algorithm.
*. The implementation here is without a common block, however.
*. Thus this routine may be used regardless of whether the
*. Jetset6.3 or Jetset7.1 library might be linked.  It is
*. not necessary to link to Jetset, however.
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRAK
*.      PARAMETER  (ITKDM=3.or.more,MXTRAK=1.or.more)
*.      INTEGER*4 NTRAK,IERR
*.      REAL*8  PTRAK (ITKDM,MXTRAK),
*.     +      THRVEC (3,3.or.more),
*.     +      THRVAL (3.or.more)
*.
*.      NTRAK = 1.to.MXTRAK
*.      CALL  PXLTH4 (NTRAK,ITKDM,PTRAK,THRVAL,THRVEC,IERR)
*.
*. The thrust vectors THRVEC are ordered according to the
*. corresponding thrust values THRVAL such that
*.     THRVAL (1) < THRVAL (2) < THRVAL (3)
*. Thus THRVEC (*,3) is the Thrust Principal axis;
*. Thus THRVEC (*,2) is the Thrust Major axis;
*. Thus THRVEC (*,1) is the Thrust Minor axis;
*.
*. INPUT     : NTRAK    Total number of particles
*. INPUT     : ITKDM    First dimension of PTRAK array
*. INPUT     : PTRAK    Particle momentum array: Px,Py,Pz,E
*. OUTPUT    : THRVAL   Thrust values
*. OUTPUT    : THRVEC   Associated Thrust vectors
*. OUTPUT    : IERR     = 0 if all is OK ;   = -1 otherwise
*.
*. CALLS     : PXLUT3
*. CALLED    : By User
*.
*. AUTHOR    :  J.W.Gary
*. CREATED   :  18-Jun-88
*. LAST MOD  :  04-Feb-89
*.
*. Modification Log.
*. 04-Feb-89  Integrate with PXLUT3  J.W.Gary
*.
*.*********************************************************
      INTEGER*4  NRLUDM,IOLUN
      PARAMETER (NRLUDM=1000,IOLUN=6)
      INTEGER*4  NTRAK,AXIS,IX1,IX2,ITKDM,IERR
      REAL*8  PTRAK (ITKDM,*),PLUND (NRLUDM,5),
     +      THRVEC (3,*),THRVAL (*)
      REAL*8  THRUST,OBLATE
      LOGICAL  LPRT
      DATA  LPRT / .FALSE. /
 
      IERR = 0
      IF (NTRAK.LE.1.OR.NTRAK.GT.NRLUDM) THEN
          IERR = -1
          WRITE (IOLUN,FMT='('' PXLTH4: Error, NTRAK ='',I6/
     +           ''  Max. allowed ='',I6)') NTRAK,NRLUDM
          GO TO 990
      END IF
*  Pack 4-momenta in Jetset format
*  ---- --------- -- ------ ------
      DO 110  IX1 = 1,NTRAK
          DO 100  AXIS = 1,4
              PLUND (IX1,AXIS) = PTRAK (AXIS,IX1)
 100      CONTINUE
 110  CONTINUE
*  Jetset algorithm for Thrust
*  ------ --------- --- ------
      CALL PXLUT3 (NTRAK,NRLUDM,PLUND,THRUST,OBLATE)
      IF (LPRT) WRITE (IOLUN,FMT='('' PXLTH4:  THRUST,'',
     +     ''OBLATE ='',2E12.4)') THRUST,OBLATE
      IF (THRUST.LT.0) THEN
          IERR = -1
          GO TO 990
      END IF
*  Buffer eigenvectors for output
*  ------ ------------ --- ------
      DO 210  IX1 = 1,3
          IX2 = NTRAK + (4 - IX1)
          THRVAL (IX1) = PLUND (IX2,4)
          DO 200  AXIS = 1,3
              THRVEC (AXIS,IX1) = PLUND (IX2,AXIS)
 200      CONTINUE
 210  CONTINUE
      IF (LPRT) THEN
          WRITE (IOLUN,FMT='('' PXLTH4: THRVAL,THRVEC ='',
     +          3(/5X,4E12.4))') (THRVAL (IX1),
     +          (THRVEC (IX2,IX1),IX2=1,3),IX1=1,3)
      END IF
 
 990  RETURN
      END

      SUBROUTINE PXLSP3 (NTRAK,ITKDM,PTRAK,EVAL,EVEC,IERR)
*.*********************************************************
*. ------
*. PXLSP3
*. ------
*. A routine to evaluate the momentum tensor, eigenvectors,
*. and eigenvalues belonging to the C parameter family.
*.
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRAK
*.      PARAMETER  (ITKDM=3.or.more,MXTRAK=1.or.more)
*.      INTEGER*4 NTRAK,IERR
*.      REAL*8  PTRAK (ITKDM,MXTRAK),
*.     +      EVEC (3,3.or.more),
*.     +      EVAL (3.or.more)
*.
*.      NTRAK = 1.to.MXTRAK
*.      CALL PXLSP3 (NTRAK,ITKDM,PTRAK,EVAL,EVEC,IERR)
*.
*. INPUT     : NTRAK    Total number of particles
*. INPUT     : ITKDM    First dimension of PTRAK array
*. INPUT     : PTRAK    Particle 3-momentum array: Px,Py,Pz
*. OUTPUT    : EVAL     C Parameter eigenvalues
*. OUTPUT    : EVEC     Associated C Eigenvectors;
*. OUTPUT    : IERR     = 0 if all is OK ;   = -1 otherwise
*.
*. Note:
*. (i)    C  = 3 * (EVAL(1)*EVAL(2) + EVAL(2)*EVAL(3) +
*.                  EVAL(3)*EVAL(1))
*. (ii)   D  = 27 * EVAL(1)*EVAL(2)*EVAL(3)
*. (iii)  EVAL (1) < EVAL (2) < EVAL (3)
*.
*. CALLS     : PXDIAM
*. CALLED    : By User
*.
*. Author    : Sukhpal Sanghera, Carleton U, Ottawa.
*. Created   : 24-Sep-89
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4  IOLUN
      PARAMETER  (IOLUN=6)
      INTEGER*4  NTRAK,ITKDM,IP,IX1,IX2,IERR,ISW,ICNT,IVMAX,NDIM
      REAL*8  PTRAK (ITKDM,*),EVEC (3,*),EVAL (*),SVBUFF (3)
      REAL*8  PMATRX (3,3),SPBUFF (3,3),WORK1 (3,3),
     +                  WORK2 (3,3),WORK3 (3,3),P1SUM,P2SUM
      REAL*8  EVMAX,P2AB,P1AB
      CHARACTER*1  CDONE (3)
      LOGICAL  LPRT
      DATA  LPRT / .FALSE. /
 
      IERR = 0
      IF (NTRAK.LE.1) THEN
          WRITE (IOLUN,FMT='('' PXLSP3: Error, NTRAK ='',I4)') NTRAK
          IERR = -1
          GO TO 990
      END IF
*  construct momentum tensor
*  --------- -------- ------
      P1SUM = 0.0D0
      DO 100 IX1 = 1,3
          DO 110 IX2 = 1,3
              PMATRX (IX1,IX2) = 0.0D0
 110      CONTINUE
 100  CONTINUE
      DO 120 IP = 1,NTRAK
          P2AB  = PTRAK(1,IP)**2 + PTRAK(2,IP)**2 + PTRAK(3,IP)**2
          P1AB  = SQRT(P2AB)
          P1SUM = P1SUM + P1AB
          DO 130 IX1 = 1,3
              DO 140 IX2 = 1,3
                  PMATRX (IX2,IX1) = PMATRX (IX2,IX1)
     +             + PTRAK (IX1,IP) * PTRAK (IX2,IP)/P1AB
 140          CONTINUE
 130      CONTINUE
 120  CONTINUE
      DO 160 IX1 = 1,3
          DO 150 IX2 = 1,3
              PMATRX (IX1,IX2) = PMATRX (IX1,IX2) / P1SUM
 150      CONTINUE
 160  CONTINUE
*  diagonalize matrix
*  ----------- ------
      NDIM = 3
      CALL PXDIAM (NDIM,PMATRX,SPBUFF,IERR,WORK1,WORK2,WORK3)
      IF (IERR.NE.0) GO TO 990
*  Eigenvalues are the diagonal elements of the diagonalized matrix
*  ----------- --- --- -------- -------- -- --- ------------ ------
      DO 200  IX1 = 1,3
          CDONE (IX1) = 'U'
          SVBUFF (IX1) = PMATRX (IX1,IX1)
 200  CONTINUE
*  put the eigenvectors/values into standard order
*  --- --- ------------ ------ ---- -------- -----
      ICNT = 0
 205  ICNT  = ICNT + 1
          EVMAX = 99999.
          DO 210  IX1 = 1,3
              IF (SVBUFF (IX1).LT.EVMAX
     +            .AND.CDONE (IX1).NE.'D') THEN
                  EVMAX = SVBUFF (IX1)
                  IVMAX = IX1
              END IF
 210      CONTINUE
          CDONE (IVMAX) = 'D'
          EVAL (ICNT) = SVBUFF (IVMAX)
          DO 220 IX1 = 1,3
              EVEC (IX1,ICNT) = SPBUFF (IX1,IVMAX)
 220      CONTINUE
      IF (ICNT.LT.3) GO TO 205
      IF (LPRT) THEN
          WRITE (IOLUN,FMT='(
     +           '' PXLSP3: EVAL, EVEC ='',3(/5X,4E12.4))')
     +           (EVAL (IX1),(EVEC (IX2,IX1),IX2=1,3),IX1=1,3)
      END IF
 
 990  RETURN
      END

      SUBROUTINE PXDIAM (NDIM,DUMMTX,EIGVEC,IERR,UMATRX,WORK2,WORK3)
*.*********************************************************
*. ------
*. PXDIAM
*. ------
*. Routine to diagonalize a real, symmetric matrix, i.e. to find
*. its eigenvalues and eigenvectors. This routine diagonalizes
*. matrices of arbitrary order.
*. The technique employed is the Jacobi iterative method.
*. Briefly, a unitary transformation is constructed to "annihilate,"
*. i.e. set to zero, all off-diagonal elements of the matrix above
*. a certain threshold.  The iterations are necessary, first of all,
*. to annihilate all the off diagonal elements and, second of all,
*. because each subsequent annihilation messes up the annihilation
*. due to a previous transformation (to a certain extent).
*. The diagonalization is mathematically guaranteed to converge,
*. however.
*. By following this procedure one therefore solves the equation
*.                 (Ut) M (U) = D
*. for the unitary matrix U and for the diagonal matrix D, where
*. Ut is the transpose of U and M is the input matrix to be
*. diagonalized.
*. The eigenvectors of M are the columns of U;
*. the eigenvalues of M are the diagonal elements of D.
*. Note:  This routine operates on  REAL*8 MATRICES
*. Usage     :
*.
*.      INTEGER*4  NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      INTEGER*4  IERR
*.      REAL*8  DUMMTX (NDIM,NDIM),
*.     +                  UMATRX (NDIM,NDIM),
*.     +                  WORK2  (NDIM,NDIM),
*.     +                  WORK3  (NDIM,NDIM),
*.     +                  EIGVEC (NDIM,NDIM)
*.
*.      CALL PXDIAM (NDIM,DUMMTX,EIGVEC,IERR,UMATRX,WORK2,WORK3)
*.
*. INPUT     : NDIM     The order of the NDIM x NDIM matrix
*. IN/OUTPUT : DUMMTX   Input: the matrix to be diagonalized
*.                      Output: the diagonalized matrix
*. OUTPUT    : EIGVEC   The eigenvectors of the matrix
*. OUTPUT    : IERR     = 0 if all is OK ;   = -1 otherwise
*. DUMMY     : UMATRX   Dummy working space, dimension NDIM x NDIM
*. DUMMY     : WORK2    Dummy working space, dimension NDIM x NDIM
*. DUMMY     : WORK3    Dummy working space, dimension NDIM x NDIM
*.
*. CALLS     : PXUNIM,PXMULM
*. CALLED    : PXJSP3
*.
*. AUTHOR    :  J.W.Gary
*. CREATED   :  18-Mar-88
*. LAST MOD  :  18-Mar-88
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4  IOLUN
      PARAMETER  (IOLUN=6)
      INTEGER*4  MXITER
      PARAMETER  (MXITER=100)
      REAL*8  XXMIN
      PARAMETER  (XXMIN=0.0D0)
      REAL*8  XISTOP
      PARAMETER  (XISTOP=0.0001)
      INTEGER*4  NDIM,IERR,IX1,IX2,IX3,IX4,NITER
      REAL*8
     +      DUMMTX (NDIM,NDIM),UMATRX (NDIM,NDIM),
     +      EIGVEC (NDIM,NDIM),WORK2 (NDIM,NDIM),
     +      WORK3  (NDIM,NDIM)
      REAL*8  XXCUT,AX0,AX1,COSPHI,S2THET,
     +                  STHETA,CTHETA
      CHARACTER*6  CDIAG,CSTAT
      LOGICAL  LPRT
      DATA  LPRT / .FALSE. /
 
      IERR = 0
*   construct the initial threshold for annihilation
*   --------- --- ------- --------- --- ------------
      XXCUT = 0.D0
      DO 180  IX1 = 1,NDIM
          DO 170  IX2 = IX1+1,NDIM
              XXCUT = XXCUT + DUMMTX (IX1,IX2)**2
 170      CONTINUE
 180  CONTINUE
      XXCUT = DSQRT (MAX (XXMIN,2.*XXCUT)) / FLOAT (NDIM)
*  start iteration
*  ----- ---------
      NITER = 0
      CDIAG = 'NODIAG'
 300  CSTAT = 'INIT'
*       annihilate off-diagonal elements above threshold
*       ---------- --- -------- -------- ----- ---------
          DO 380 IX1 = 1,NDIM
              DO 370  IX2 =  IX1+1,NDIM
                  IF (DABS (DUMMTX (IX1,IX2)).GT.XXCUT) THEN
                      NITER = NITER + 1
                      CSTAT = 'AGAIN'
*                   construct unitary matrix
*                   --------- ------- ------
                      DO 320 IX3 = 1,NDIM
                          DO 310 IX4 = 1,NDIM
                              UMATRX (IX3,IX4) = 0.D0
                              IF (IX3.EQ.IX4) UMATRX (IX3,IX4) = 1.
 310                      CONTINUE
 320                  CONTINUE
                      AX0 = 2.* DUMMTX (IX1,IX2)
                      AX1 = DUMMTX (IX1,IX1) - DUMMTX (IX2,IX2)
                      COSPHI = AX1 / DSQRT (AX0**2 + AX1**2)
                      S2THET = (1.-COSPHI) /2.
                      IF (S2THET.LT.0.) THEN
                          WRITE (IOLUN,FMT='(
     +                           '' PXDIAM:  Error,  S2THET ='',
     +                           E12.4)') S2THET
                          IERR = -1
                          RETURN
                      END IF
                      STHETA = DSQRT (S2THET)
                      CTHETA = DSQRT (1.-S2THET)
                      UMATRX (IX1,IX1) =   CTHETA
                      UMATRX (IX2,IX2) =   CTHETA
                      UMATRX (IX1,IX2) =   STHETA
                      UMATRX (IX2,IX1) = - STHETA
                      IF (LPRT) WRITE (IOLUN,FMT='(
     +                    ''   NITER ='',I5/
     +                    ''   IX1,IX2,XXCUT,STHETA,CTHETA ='',
     +                    2I5,3E12.4/
     +                    ''   UMATRX =''/3(10X,3E12.4/))')
     +                    NITER,IX1,IX2,XXCUT,STHETA,CTHETA,
     +                    ((UMATRX (IX3,IX4),IX4=1,NDIM),IX3=1,NDIM)
*                   perform unitary transformation
*                   ------- ------- --------------
                      CALL PXUNIM (NDIM,DUMMTX,UMATRX,WORK2,WORK3)
                      IF (LPRT) WRITE (IOLUN,FMT='(
     +                    '' PXDIAM:  DUMMTX =''/3(10X,3E12.4/))')
     +                    ((DUMMTX (IX3,IX4),IX4=1,NDIM),IX3=1,NDIM)
*                   update eigenvector matrix
*                   ------ ----------- ------
                      IF (NITER.EQ.1) THEN
                          DO 340 IX3 = 1,NDIM
                              DO 330  IX4 = 1,NDIM
                                  EIGVEC (IX3,IX4)
     +                              = UMATRX (IX3,IX4)
 330                          CONTINUE
 340                      CONTINUE
                      ELSE
                          CALL PXMULM (NDIM,EIGVEC,UMATRX,WORK2)
                          DO 360 IX3 = 1,NDIM
                              DO 350  IX4 = 1,NDIM
                                  EIGVEC (IX3,IX4)
     +                              = WORK2 (IX3,IX4)
 350                          CONTINUE
 360                      CONTINUE
                      END IF
                  END IF
 370          CONTINUE
 380      CONTINUE
*       cut off after maximum number of iterations
*       --- --- ----- ------- ------ -- ----------
          IF (NITER.GT.MXITER.OR.
     +       (DABS (CTHETA).EQ.1..AND.STHETA.EQ.0.)) THEN
              CDIAG = 'DIAG'
              CSTAT = 'QUIT'
          END IF
*       repeat until all off-diagonal elements below threshold
*       ------ ----- --- --- -------- -------- ----- ---------
          IF (CSTAT.EQ.'AGAIN') GO TO 300
*       criteron for whether matrix is sufficiently diagonal
*       -------- --- ------- ------ -- ------------ --------
          IF (CSTAT.EQ.'QUIT') THEN
          ELSE IF (XXCUT.LE.XISTOP) THEN
              CDIAG = 'DIAG'
          ELSE
              XXCUT = XXCUT / FLOAT (NDIM)
          END IF
      IF (CDIAG.NE.'DIAG') GO TO 300
*  matrix now diagonalized (I hope !!!)
*  ------ --- ------------  - ----
      IF (LPRT) WRITE (IOLUN,FMT='(
     +    '' PXDIAM:  EIGVEC =''/3(10X,3E12.4/))')
     +      ((EIGVEC (IX3,IX4),IX4=1,NDIM),IX3=1,NDIM)
 
      RETURN
      END

      SUBROUTINE PXUNIM (NDIM,XMTX,UMX,UMT,WORK3)
*.*********************************************************
*. ------
*. PXUNIM
*. ------
*. Routine to perform a unitary transfomation of a real,
*. symmetric matrix.
*. Note:  This routine operates on REAL*8 MATRICES
*. Usage     :
*.
*.      INTEGER*4  NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  XMTX (NDIM,NDIM),
*.     +                  UMX  (NDIM,NDIM),
*.     +                  UMT  (NDIM,NDIM),
*.     +                  WORK3  (NDIM,NDIM)
*.
*.      CALL PXUNIM (NDIM,XMTX,UMX,UMT,WORK3)
*.
*. INPUT     : NDIM    The order of the NDIM x NDIM matrix
*. IN/OUTPUT : XMTX    Input:  the matrix to be transformed
*.                     Output: the transformed matrix
*. INPUT     : UMX     The unitary transformation matrix
*. DUMMY     : UMT     Working space, dimension NDIM x NDIM
*. DUMMY     : WORK3   Working space, dimension NDIM x NDIM
*.
*. CALLS     : PXMULM
*. CALLED    : PXDIAM
*.
*. AUTHOR    :  J.W.Gary
*. CREATED   :  18-Mar-88
*. LAST MOD  :  18-Mar-88
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4  NDIM,IX1,IX2,IX3
      REAL*8  XMTX (NDIM,NDIM),UMX (NDIM,NDIM),
     +                  UMT (NDIM,NDIM),WORK3 (NDIM,NDIM)
 
*  transpose of the unitary matrix
*  --------- -- --- ------- ------
      DO 180  IX1 = 1,NDIM
          DO 170  IX2 = 1,NDIM
              UMT (IX2,IX1) = UMX (IX1,IX2)
 170      CONTINUE
 180  CONTINUE
*  perform unitary transformation
*  ------- ------- --------------
      CALL PXMULM (NDIM,XMTX,UMX,WORK3)
      CALL PXMULM (NDIM,UMT,WORK3,XMTX)
 
      RETURN
      END

      SUBROUTINE PXMULM (NDIM,XMX1,XMX2,XMXX)
*.*********************************************************
*. ------
*. PXMULM
*. ------
*. Routine to perform matrix multiplication
*. Note:  This routine operates on REAL*8 MATRICES
*. Usage     :
*.
*.      INTEGER*4  NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  XMX1 (NDIM,NDIM),
*.     +                  XMX2 (NDIM,NDIM),
*.     +                  XMXX (NDIM,NDIM)
*.
*.      CALL PXMULM (NDIM,XMX1,XMX2,XMXX)
*.
*. INPUT     :  NDIM       The order of the NDIM x NDIM matrices
*. INPUT     :  XMX1       1st matrix in the product (MX1) x (MX2)
*. INPUT     :  XMX2       2nd matrix in the product (MX1) x (MX2)
*. OUTPUT    :  XMXX       The result =  (MX1) x (MX2)
*.
*. CALLS     : none
*. CALLED    : PXDIAM,PXUNIM
*.
*. AUTHOR    :  J.W.Gary
*. CREATED   :  18-Mar-88
*. LAST MOD  :  18-Mar-88
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4  NDIM,IX1,IX2,IX3
      REAL*8  XMX1 (NDIM,NDIM),XMX2 (NDIM,NDIM),
     +                  XMXX (NDIM,NDIM)
      DO 180  IX1 = 1,NDIM
          DO 170  IX2 = 1,NDIM
              XMXX (IX1,IX2) = 0.D0
              DO 160  IX3 = 1,NDIM
                  XMXX (IX1,IX2) = XMXX (IX1,IX2)
     +               + XMX1 (IX1,IX3) * XMX2 (IX3,IX2)
 160          CONTINUE
 170      CONTINUE
 180  CONTINUE
      RETURN
      END

      SUBROUTINE PXJSP3 (NTRAK,ITKDM,PTRAK,EVAL,EVEC,IERR)
*.*********************************************************
*. ------
*. PXJSP3
*. ------
*. Routine to calculate the eigenvectors and eigenvalues of the
*. momentum tensor. The eigenvectors of the momentum tensor are
*. the same as the eigenvectors of the Sphericity matrix;
*. the eigenvalues are related as noted below.
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRAK
*.      PARAMETER  (ITKDM=3.or.more,MXTRAK=1.or.more)
*.      INTEGER*4 NTRAK,IERR
*.      REAL*8 PTRAK (ITKDM,MXTRAK),
*.     +      EVEC (3,3.or.more),
*.     +      EVAL (3.or.more)
*.
*.      NTRAK = 1.to.MXTRAK
*.      CALL PXJSP3 (NTRAK,ITKDM,PTRAK,EVAL,EVEC,IERR)
*.
*. INPUT     : NTRAK    Total number of particles
*. INPUT     : ITKDM    First dimension of PTRAK array
*. INPUT     : PTRAK    Particle 3-momentum array: Px,Py,Pz
*. OUTPUT    : EVAL     Sphericity Eigenvalues
*. OUTPUT    : EVEC     Associated Sphericity Eigenvectors;
*. OUTPUT    : IERR     = 0 if all is OK ;   = -1 otherwise
*.
*. Note:
*. (i)    Sphericity  = (3./2.) * (EVAL (1) + EVAL (2))
*. (ii)   Aplanarity  = (3./2.) *  EVAL (1)
*. (iii)  EVAL (1) < EVAL (2) < EVAL (3)
*. (iv)   EVEC (*,3) is the principal sphericity axis
*.
*. CALLS     : PXDIAM
*. CALLED    : By User
*.
*. AUTHOR    :  J.W.Gary
*. CREATED   :  18-Mar-88
*. LAST MOD  :  18-Mar-88
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4  IOLUN
      PARAMETER  (IOLUN=6)
      INTEGER*4  NTRAK,ITKDM,IP,IX1,IX2,IERR,ISW,ICNT,IVMAX,NDIM
      REAL*8 PTRAK (ITKDM,*)
      REAL*8  PMATRX (3,3),SPBUFF (3,3),WORK1 (3,3),
     +                  WORK2 (3,3),WORK3 (3,3)
      REAL*8 EVEC (3,*),EVAL (*),SVBUFF (3)
      REAL*8  P2SUM
      REAL*8 EVMAX
      CHARACTER*1  CDONE (3)
      LOGICAL  LPRT
      DATA  LPRT / .FALSE. /

      IERR = 0
      IF (NTRAK.LE.1) THEN
          WRITE (IOLUN,FMT='('' PXJSP3: Error, NTRAK ='',I4)')
     +           NTRAK
          IERR = -1
          GO TO 990
      END IF
*  construct momentum tensor
*  --------- -------- ------
      P2SUM = 0.
      DO 130 IX1 = 1,3
          DO 120 IX2 = 1,3
              PMATRX (IX1,IX2) = 0.D0
 120      CONTINUE
 130  CONTINUE
      DO 180 IP = 1,NTRAK
          P2SUM = P2SUM
     +          + PTRAK(1,IP)**2 + PTRAK(2,IP)**2 + PTRAK(3,IP)**2
          DO 170 IX1 = 1,3
              DO 160 IX2 = 1,3
                  PMATRX (IX2,IX1) = PMATRX (IX2,IX1)
     +                + PTRAK (IX1,IP) * PTRAK (IX2,IP)
 160          CONTINUE
 170      CONTINUE
 180  CONTINUE
      DO 380 IX1 = 1,3
          DO 370 IX2 = 1,3
              PMATRX (IX1,IX2) = PMATRX (IX1,IX2) / P2SUM
              IF (LPRT) WRITE (IOLUN,FMT='(
     +            '' PXJSP3: IX1,IX2,PMATRX ='',2I5,E12.4)')
     +            IX1,IX2,PMATRX (IX1,IX2)
 370      CONTINUE
 380  CONTINUE
*  diagonalize matrix
*  ----------- ------
      NDIM = 3
      CALL PXDIAM (NDIM,PMATRX,SPBUFF,IERR,WORK1,WORK2,WORK3)
      IF (IERR.NE.0) GO TO 990
*  eigenvalues are the diagonal elements of the diagonalized matrix
*  ----------- --- --- -------- -------- -- --- ------------ ------
      DO 450  IX1 = 1,3
          CDONE (IX1) = 'U'
          SVBUFF (IX1) = PMATRX (IX1,IX1)
 450  CONTINUE
*  put the eigenvectors/values into standard order
*  --- --- ------------ ------ ---- -------- -----
      ICNT = 0
 500  ICNT  = ICNT + 1
          EVMAX = 99999.
          DO 540  IX1 = 1,3
              IF (SVBUFF (IX1).LT.EVMAX
     +            .AND.CDONE (IX1).NE.'D') THEN
                  EVMAX = SVBUFF (IX1)
                  IVMAX = IX1
              END IF
 540      CONTINUE
          CDONE (IVMAX) = 'D'
          EVAL (ICNT) = SVBUFF (IVMAX)
          DO 580 IX1 = 1,3
              EVEC (IX1,ICNT) = SPBUFF (IX1,IVMAX)
 580      CONTINUE
      IF (ICNT.LT.3) GO TO 500
      IF (LPRT) THEN
          WRITE (IOLUN,FMT='(
     +           '' PXJSP3: EVAL, EVEC ='',3(/5X,4E12.4))')
     +           (EVAL (IX1),(EVEC (IX2,IX1),IX2=1,3),IX1=1,3)
      END IF

 990  RETURN
      END

      SUBROUTINE PXMMBB (NTRAK,ITKDM,PTRAK,TVEC,AMH,AML,
     +                   BT,BW,IERR)
*.*********************************************************
*. ------
*. PXMMBB
*. ------
*. SOURCE:  D.R.Ward
*. Calculate Heavy and light jet masses using Thrust method,
*  and also the jet broadening measures of Webber et al.
*. Usage     :
*.
*.      INTEGER*4  NTRAK,ITKDM,IERR,MXTRK
*.      PARAMETER (ITKDM=4.or.more,MXTRK=1.or.more)
*.      REAL*8 PTRAK (ITKDM,MXTRK),TVEC (3),AMH,AML,BT,BW
*.
*.      CALL PXMMBB (NTRAK,ITKDM,PTRAK,TVEC,AMH,AML,BT,BW,IERR)
*.
*. INPUT     : NTRAK      Number of "tracks"
*. INPUT     : ITKDM      First dimension of PTRAK array
*. INPUT     : PTRAK      Array of  4-momenta (Px,Py,Pz,E)
*. INPUT     : TVEC       Vector defining the Thrust axis.
*. OUTPUT    : AMH        Invariant mass of heavy hemisphere/Evis
*. OUTPUT    : AML        Invariant mass of light hemisphere/Evis
*. OUTPUT    : BT         Sum of transverse energies/pvis
*. OUTPUT    : BW         Sum of transverse energies/pvis wide jet
*. OUTPUT    : IERR       = 0 if all is OK ;   = -1 otherwise
*.
*.*********************************************************
*
*
      INTEGER*4 NTRAK,IERR,I,J,IHEM,ITKDM
      REAL*8    PTRAK(ITKDM,*),AMH,AML,PHEM(4,2),CTH,TVEC(*),
     +        SAVE,BB(2),BT,BW,SUMP,PL,PT
*
      AMH=0.D0
      AML=0.D0
      IERR=0
*
      IF(NTRAK.LE.0) THEN
         WRITE (6,FMT='('' PXMMBB: Error, NTRAK=0 '')')
         IERR=-1
         GO TO 99
      ENDIF
*
      CALL PXZERV (8,PHEM)
      CALL PXZERV (2,BB)
      SUMP=0.D0
      DO 211 I=1,NTRAK
         CTH=TVEC(1)*PTRAK(1,I)+TVEC(2)*PTRAK(2,I)+TVEC(3)*PTRAK(3,I)
         IHEM=1
         IF(CTH.LT.0) IHEM=2
         DO 212 J=1,4
            PHEM(J,IHEM)=PHEM(J,IHEM)+PTRAK(J,I)
  212    CONTINUE
         CALL PXPLT3(PTRAK(1,I),TVEC,PL,PT,IERR)
         BB(IHEM)=BB(IHEM)+PT
         SUMP=SUMP+SQRT(PL**2+PT**2)
*
  211 CONTINUE
*
      AMH=SQRT(ABS(PHEM(4,1)**2-PHEM(1,1)**2-PHEM(2,1)**2-PHEM(3,1)**2))
      AML=SQRT(ABS(PHEM(4,2)**2-PHEM(1,2)**2-PHEM(2,2)**2-PHEM(3,2)**2))
      IF(AML.GT.AMH) THEN
         SAVE=AMH
         AMH=AML
         AML=SAVE
      ENDIF
      AMH=AMH/(PHEM(4,1)+PHEM(4,2))
      AML=AML/(PHEM(4,1)+PHEM(4,2))
      BT=(BB(1)+BB(2))*0.5D0/SUMP
      BW=MAX(BB(1),BB(2))*0.5D0/SUMP
*
  99  RETURN
      END

      SUBROUTINE PXPLT3 (VEC1,VEC2,PPL,PPT,IERR)
*.*********************************************************
*. ------
*. PXPLT3
*. ------
*. SOURCE: VECSUB (V. Blobel)
*. calculate the magnitude of the component of VEC1
*. parallel to VEC2 (=PPL) and the magnitude of the
*. component of VEC1 perpendicular to VEC2 (=PPT)
*. Usage     :
*.
*.      REAL*8  VEC1 (3.or.more),
*.     +      VEC2 (3.or.more)
*.      REAL*8  PPL,PPT
*.      INTEGER*4  IERR
*.
*.      CALL PXPLT3 (VEC1,VEC2,PPL,PPT,IERR)
*.
*. INPUT     : VEC1    The vector whose components are to
*.                     be calculated
*. INPUT     : VEC2    The reference vector
*. OUTPUT    : PPL     The parallel component of VEC1
*. OUTPUT    : PPT     The perpendicular component of VEC1
*. OUTPUT    : IERR    = 0 if all is OK ;   = -1 otherwise
*.
*.*********************************************************
      INTEGER*4  IX,IERR
      REAL*8  VEC1 (*),VEC2 (*)
      REAL*8  PPL,PPT
      REAL*8  BX,CX,DX,EX,GX (3)
      IERR = 0
      BX = 0D0
      CX = 0D0
      DO 120  IX = 1,3
          BX = BX + VEC2 (IX) * VEC2 (IX)
          CX = CX + VEC1 (IX) * VEC2 (IX)
 120  CONTINUE
      DX = DSQRT (BX)
      GX (1) = VEC1 (2) * VEC2 (3) - VEC1 (3) * VEC2 (2)
      GX (2) = VEC1 (3) * VEC2 (1) - VEC1 (1) * VEC2 (3)
      GX (3) = VEC1 (1) * VEC2 (2) - VEC1 (2) * VEC2 (1)
      EX = DSQRT (GX (1)*GX (1) + GX (2)*GX (2) + GX (3)*GX (3))
      IF (DX.EQ.0.D0) THEN
          IERR = -1
      ELSE
          PPL = CX / DX
          PPT = EX / DX
      END IF
      RETURN
      END

      SUBROUTINE PXZERV (ISZE,VEC)
*.*********************************************************
*. ------
*. PXZERV
*. ------
*. SOURCE: J.W.Gary
*. Zero a vector of arbitrary length
*. Usage     :
*.
*.      INTEGER*4  NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  VEC (NDIM)
*.      INTEGER*4  ISIZ
*.
*.      ISIZ = 1.to.NDIM
*.      CALL PXZERV (ISZE,VEC)
*.
*. INPUT     : ISIZ    The length of the vector to be zeroed
*. INPUT     : VEC     the vector to be zeroed
*.
*.*********************************************************
      INTEGER*4  ISZE,IX
      REAL*8  VEC (*)
      DO 120  IX = 1,ISZE
          VEC (IX) = 0.D0
 120  CONTINUE
      RETURN
      END

      SUBROUTINE PXADDV (ISIZ,VEC1,VEC2,VECO)
*.*********************************************************
*. ------
*. PXADDV
*. ------
*. SOURCE:  J.W.Gary
*. Add two vectors of arbitrary length
*. Usage     :
*.
*.      INTEGER*4  ISIZ,NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  VEC1 (NDIM.or.more),
*.     +      VEC2 (NDIM.or.more),
*.     +      VECO (NDIM.or.more)
*.
*.      ISIZ = 1.to.NDIM
*.      CALL PXADDV (ISIZ,VEC1,VEC2,VECO)
*.
*. INPUT     : ISIZ    The length of the vectors
*. INPUT     : VEC1    The first vector
*. INPUT     : VEC2    The second vector
*. OUTPUT    : VECO    The vector sum of VEC1 and VEC2
*.                     (elements 1 to ISIZ)
*.
*.*********************************************************
      INTEGER*4  ISIZ,IX
      REAL*8  VEC1 (*),VEC2 (*),VECO (*)
      REAL*8  AX
      DO 120  IX = 1,ISIZ
          AX = VEC1 (IX) + VEC2 (IX)
          VECO (IX) = AX
 120  CONTINUE
      RETURN
      END

      SUBROUTINE PXCOPV (ISZE,VECI,VECO)
*.*********************************************************
*. ------
*. PXCOPV
*. ------
*. SOURCE:  J.W.Gary
*. Copy a vector of arbitrary length into another
*. Usage     :
*.
*.      INTEGER*4  NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  VEC1 (NDIM.or.more),
*.     +      VECO (NDIM.or.more)
*.      INTEGER*4  ISZE
*.
*.      ISZE = 1.to.NDIM
*.      CALL PXCOPV (ISZE,VECI,VECO)
*.
*. INPUT     : ISZE    The length of the vector to be copied
*. INPUT     : VECI    The input vector
*. OUTPUT    : VECO    The output vector
*.
*.*********************************************************
      INTEGER*4  ISZE,IX
      REAL*8  VECI (*),VECO (*)
      DO 120 IX = 1,ISZE
          VECO (IX) = VECI (IX)
 120  CONTINUE
      RETURN
      END

      SUBROUTINE PXCRO3 (VEC1,VEC2,VECO)
*.*********************************************************
*. ------
*. PXCRO3
*. ------
*. SOURCE: VECSUB (V. Blobel)
*. Calculate the cross product between two 3-vectors
*. Usage     :
*.
*.      REAL*8  VEC1 (3.or.more),
*.     +      VEC2 (3.or.more),
*.     +      VECO (3.or.more)
*.
*.      CALL PXCRO3 (VEC1,VEC2,VECO)
*.
*. INPUT     : VEC1    The first vector
*. INPUT     : VEC2    The second vector
*. OUTPUT    : VECO    The vector cross product:
*.                       --->   --->   --->
*.                       VECO = VEC1 x VEC2
*.
*.*********************************************************
      REAL*8  VEC1 (*),VEC2 (*),VECO (*)
      REAL*8  AX,BX
 
      AX = VEC1 (2) * VEC2 (3)
      BX = VEC1 (3) * VEC2 (2)
      VECO (1) = AX - BX
      AX = VEC1 (3) * VEC2 (1)
      BX = VEC1 (1) * VEC2 (3)
      VECO (2) = AX - BX
      AX = VEC1 (1) * VEC2 (2)
      BX = VEC1 (2) * VEC2 (1)
      VECO (3) = AX - BX
 
      RETURN
      END

      SUBROUTINE PXDOTV (ISIZ,VEC1,VEC2,DOTP)
*.*********************************************************
*. ------
*. PXDOTV
*. ------
*. SOURCE: J.W.Gary
*. Calculate the dot product between two vectors of
*. arbitrary length
*. Usage     :
*.
*.      INTEGER*4  NDIM
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  VEC1 (NDIM.or.more),
*.     +      VEC2 (NDIM.or.more)
*.      REAL*8  DOTP
*.      INTEGER*4  ISIZ
*.
*.      ISIZ = 1.to.NDIM
*.      CALL PXDOTV (ISIZ,VEC1,VEC2,DOTP)
*.
*. INPUT     : ISIZ    The length of the vector
*. INPUT     : VEC1    The first vector
*. INPUT     : VEC2    The second vector
*. OUTPUT    : DOTP    The vector dot product,
*.                            --->   --->
*.                     DOTP = VEC1 . VEC2
*.
*.*********************************************************
      INTEGER*4  ISIZ,IX
      REAL*8  VEC1 (*),VEC2 (*)
      REAL*8  DOTP
      REAL*8  AX
 
      AX = 0D0
      DO 120  IX = 1,ISIZ
          AX = AX + VEC1 (IX) * VEC2 (IX)
 120  CONTINUE
      DOTP = AX
 
      RETURN
      END

      SUBROUTINE PXMAGV (ISIZ,VEC,XMAG)
*.*********************************************************
*. ------
*. PXMAGV
*. ------
*. SOURCE: J.W.Gary
*. Calculate the magnitude of a vector of arbitrary length
*. Usage     :
*.
*.     INTEGER*4  NDIM
*.     PARAMETER  (NDIM=1.or.more)
*.     REAL*8  VEC (NDIM.or.more)
*.     REAL*8  XMAG
*.     INTEGER*4  ISIZ
*.
*.     ISIZ = 1.to.NDIM
*.     CALL PXMAGV (ISIZ,VEC,XMAG)
*.
*. INPUT     : ISIZ    The length of the vector
*. INPUT     : VEC     The vector
*. OUTPUT    : XMAG    The magnitude of the vector
*.
*.*********************************************************
      INTEGER*4  ISIZ,IX
      REAL*8  VEC (*)
      REAL*8  XMAG
      REAL*8  AX,BX
      AX = 0.0
      DO 120 IX = 1,ISIZ
          AX = AX + VEC (IX) * VEC (IX)
 120  CONTINUE
      BX = DSQRT (AX)
      XMAG = BX
      RETURN
      END

      SUBROUTINE PXSORV (ISZ,ARY,KIX,COPT)
*.*********************************************************
*. ------
*. PXSORV
*. ------
*. SOURCE: HERWIG (B.Webber,G.Marchesini)
*. Sort a REAL*8 array into assending order based on
*. the magnitude of its elements; provide an
*. INTEGER*4 "index array" which specifies the ordering
*. of the array.
*. Usage     :
*.
*.      PARAMETER  (NDIM=1.or.more)
*.      REAL*8  ARY (NDIM)
*.      INTEGER*4  KIX (NDIM)
*.      INTEGER*4  ISIZ
*.      CHARACTER*1  COPT
*.
*.      ISIZ = 1.to.NDIM
*.      COPT = 'I'
*.      CALL PXSORV (ISIZ,ARY,KIX,COPT)
*.
*. INPUT     : ISIZ  The dimension of the input array
*. INPUT     : ARY   The input array
*. OUTPUT    : KIX   The index array
*. CONTROL   : COPT  Control of output vector ARY
*.              = ' ' : return sorted ARY and index array KIX
*.              = 'I' : return index array KIX only, don't
*.                      modify ARY
*.
*.*********************************************************
      INTEGER*4  MXSZ
      PARAMETER  (MXSZ=500)
      INTEGER*4  ISZ,IX,JX
      INTEGER*4  KIX (*),IL (MXSZ),IR (MXSZ)
      REAL*8  ARY (*),BRY (MXSZ)
      CHARACTER*1  COPT
      IF (ISZ.GT.MXSZ) THEN
          WRITE (6,FMT='('' PXSORT: Error,'',
     +           '' Max array size ='',I6)') MXSZ
          KIX (1) = -1
          GO TO 990
      END IF
      IL (1) = 0
      IR (1) = 0
      DO 10 IX = 2,ISZ
          IL (IX) = 0
          IR (IX) = 0
          JX = 1
   2      IF (ARY (IX).GT.ARY (JX)) GO TO 5
   3      IF (IL (JX).EQ.0) GO TO 4
          JX = IL (JX)
          GO TO 2
   4      IR (IX) = -JX
          IL (JX) =  IX
          GO TO 10
   5      IF (IR (JX).LE.0) GO TO 6
          JX = IR (JX)
          GO TO 2
   6      IR (IX) = IR (JX)
          IR (JX) = IX
  10  CONTINUE
      IX = 1
      JX = 1
      GO TO 8
  20  JX = IL (JX)
   8  IF (IL (JX).GT.0) GO TO 20
   9  KIX (IX) = JX
      BRY (IX) = ARY (JX)
      IX = IX + 1
      IF (IR (JX)) 12,30,13
  13  JX = IR (JX)
      GO TO 8
  12  JX = -IR (JX)
      GO TO 9
  30  IF (COPT.EQ.'I') RETURN
      DO 31 IX = 1,ISZ
  31  ARY (IX) = BRY (IX)
 990  RETURN
      END

      SUBROUTINE PXAKO4 (NTRAK,ITKDM,MXTRK,PTRAK,NTPER,
     +                   AKOPL,AKOVEC,IERR)
*.*********************************************************
*. ------
*. PXAKO4
*. ------
*. Routine to calculate the acoplanarity distribution.
*. The algorithm is an approximate one based on a Jade
*. program by S.Bethke and E.Elsen.
*. The algorithm has the following characteristics:
*.   (1) The missing momentum is added as an extra particle
*.       in order to "complete" momentum conservation.
*.   (2) The acoplanarity plane is assumed to be defined by
*.       the momentum vectors of two of the particles in the
*.       event. The plane so-defined which results in the
*.       smallest acoplanarity value is taken as the "true"
*.       acoplanarity plane.
*.   (3) Should the number of particles in the event exceed
*.       the cutoff NTPER (see explanation of the argument list
*.       below) only the NTPER particles having the largest
*.       momentum are used to define the acoplanarity plane.
*.       The missing momentum vector used to complete momentum
*.       conservation is calculated using only these NTPER particles.
*.   (4) The final acoplanarity value is calculated using all
*.       particles, whether or not they were used in the search
*.       for the acoplanarity plane.  This acoplanarity calculation
*.       also includes a fictitious particle introduced to complete
*.       momentum conservation, should one be necessary.
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRK,NTPER
*.      PARAMETER  (ITKDM=4.or.more,MXTRK=1.or.more)
*.      PARAMETER  (NTPER=80.or.so)
*.      INTEGER*4  NTRAK,IERR
*.      REAL*8  PTRAK (ITKDM,MXTRK),AKOVEC (3.or.more)
*.      REAL*8  AKOPL
*.
*.      NTRAK = 1.to.MXTRK
*.      CALL PXAKO4 (NTRAK,ITKDM,MXTRK,PTRAK,
*.     +            NTPER,AKOPL,AKOVEC,IERR)
*.
*. INPUT     : NTRAK   Total number of particles
*. INPUT     : ITKDM   First dimension of PTRAK
*. INPUT     : MXTRK   Maximum number particles allowed in PTRAK
*. INPUT     : PTRAK   Particle 4-momentum (Px,Py,Pz,E)
*. INPUT     : NTPER   Maximum number of particles to use for the
*.                     Acoplanarity calculation (for speed purposes)
*. OUTPUT    : AKOPL   The acoplanarity value
*. OUTPUT    : AKOVEC  the normalized acoplanarity axis vector
*. OUTPUT    : IERR    = 0 if all is OK
*.
*. CALLS     : PXADDV,PXMAGV,PXCOPV,PXSORV,PXCRO3,PXDOTV
*. CALLED    : By User
*.
*. AUTHOR    :  M.Weber/J.W.Gary
*. CREATED   :  05-Mar-89
*. LAST MOD  :  05-Mar-89
*.
*. Modification Log.
*.
*.*********************************************************
      INTEGER*4 LMXTK
      PARAMETER  (LMXTK=250)
      INTEGER*4  KIX (LMXTK)
      INTEGER*4  NTRAK,ITKDM,MXTRK,IERR,NTPER,NPERM,IX,IP1,IP2,IP3,
     +         INX1,INX2,IOFF,IKCNT
      REAL*8  PTRAK (ITKDM,*),AKOVEC (*),PCROS (3),PMISS (4),
     +      PMAG (LMXTK)
      REAL*8  AKOPL,PTOT,AKTRY,DOTP,PSUM
      SAVE  IKCNT
      LOGICAL  SORT
      DATA  IKCNT / 0 /
 
      IERR = 0
      IF (NTRAK.LE.1.OR.NTRAK.GE.MXTRK) THEN
          WRITE (6,FMT='('' PXAKO4: Error, NTRAK+1 ='',I6,
     +           '' greater than MXTRK ='',I6)') NTRAK+1,MXTRK
          IERR = -1
          GO TO 990
      END IF
      IF (NTRAK.GE.LMXTK) THEN
          WRITE (6,FMT='('' PXAKO4: Error, NTRAK+1 ='',I6,
     +           '' greater than LMXTK ='',I6)') NTRAK+1,LMXTK
          IERR = -1
          GO TO 990
      END IF
*  sort momenta if number of tracks too large
*  ---- ------- -- ------ -- ------ --- -----
      NPERM = NTRAK
      IF (NPERM.GT.NTPER) THEN
          DO 110 IP1 = 1,NTRAK
              CALL PXMAGV (3,PTRAK (1,IP1),PMAG (IP1))
 110      CONTINUE
          CALL PXSORV (NTRAK,PMAG,KIX,'I')
          NPERM = NTPER
          SORT = .TRUE.
          IKCNT = IKCNT + 1
          IF (IKCNT.LE.5) WRITE (6,FMT='(
     +       '' PXKOP4: NTRAK,NPERM ='',2I10)') NTRAK,NPERM
      ELSE
          DO 115 IP1 = 1,NTRAK
              KIX (IP1) = IP1
 115      CONTINUE
          SORT = .FALSE.
      END IF
*  missing momentum for particles to be used to find plane
*  ------- -------- --- --------- -- -- ---- -- ---- -----
      CALL PXZERV (4,PMISS)
      DO 120 IP1 = 1,NPERM
C >>> This loop for Bethke/Elsen (almost)      DO 120 IP1 = 1,NTRAK
          INX1 = KIX ((NTRAK+1)-IP1)
          CALL PXADDV (4,PTRAK (1,INX1),PMISS,PMISS)
 120  CONTINUE
      CALL PXMAGV (3,PMISS,PTOT)
      IOFF = 1
      IF (PTOT.GT.(1.E-4)) THEN
          NPERM = NPERM + 1
          IOFF = 2
          DO 125 IX = 1,3
              PTRAK (IX,NTRAK+1) = - PMISS (IX)
 125      CONTINUE
          KIX (NTRAK+1) = NTRAK + 1
      END IF
*  loop over permutations, find acoplanarity axis
*  ---- ---- ------------  ---- ------------ ----
      AKOPL = 999999.
      DO 170 IP1 = 1,NPERM-1
          DO 160 IP2 = IP1+1,NPERM
*           trial acoplanarity plane
*           ----- ------------ -----
              INX1 = KIX ((NTRAK+IOFF)-IP1)
              INX2 = KIX ((NTRAK+IOFF)-IP2)
              CALL PXCRO3 (PTRAK (1,INX1),PTRAK (1,INX2),PCROS)
              CALL PXMAGV (3,PCROS,PTOT)
              IF (PTOT.EQ.0) THEN
                  WRITE (6,FMT='('' PXAKO4: PTOT ='',E12.4)') PTOT
                  GO TO 160
              END IF
*           trial acoplanarity value
*           ----- ------------ -----
              AKTRY = 0.
              DO 140 IP3 = 1,NPERM
                  INX1 = KIX ((NTRAK+IOFF)-IP3)
                  CALL PXDOTV (3,PCROS,PTRAK (1,INX1),DOTP)
                  AKTRY = AKTRY + ABS (DOTP)
 140          CONTINUE
              AKTRY = AKTRY / PTOT
*           store new acoplanarity value and axis
*           ----- --- ------------ ----- --- ----
              IF (AKTRY.LT.AKOPL) THEN
                  DO 150 IX = 1,3
                      AKOVEC (IX) = PCROS (IX) / PTOT
 150              CONTINUE
                  AKOPL = AKTRY
              END IF
 160      CONTINUE
 170  CONTINUE
*  missing momentum for all particles
*  ------- -------- --- --- ---------
      IF (SORT) CALL PXMIS4 (NTRAK,ITKDM,PTRAK,PTRAK (1,NTRAK+1))
      PSUM = 0.
      AKOPL = 0.
      DO 180 IP1 = 1,(NTRAK+IOFF)-1
          CALL PXDOTV (3,AKOVEC,PTRAK (1,IP1),DOTP)
          AKOPL = AKOPL + ABS (DOTP)
          CALL PXMAGV (3,PTRAK (1,IP1),PTOT)
          PSUM = PSUM + PTOT
 180  CONTINUE
      AKOPL = 4. * (AKOPL / PSUM)**2
 
 990  RETURN
      END

      SUBROUTINE PXMIS4 (NTRAK,ITKDM,PTRAK,PMISS)
*.*********************************************************
*. ------
*. PXMIS4
*. ------
*. SOURCE: J.W.Gary
*. Routine to calculate the missing momentum in an event
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRK
*.      PARAMETER  (ITKDM=4.or.more,MXTRK=1.or.more)
*.      INTEGER*4  NTRAK
*.      REAL*8  PTRAK (ITKDM,MXTRK),PMISS (4.or.more)
*.
*.      NTRAK = 1.to.MXTRK
*.      CALL PXMIS4 (NTRAK,ITKDM,PTRAK,PMISS)
*.
*. INPUT     : NTRAK   Total number of particles
*. INPUT     : ITKDM   First dimension of PTRAK
*. INPUT     : PTRAK   Particle 4-momentum (Px,Py,Pz,E)
*. OUTPUT    : PMISS   Missing momentum vector in first 3 components;
*.                     total visible energy in 4th component
*.
*.*********************************************************
      INTEGER*4  NTRAK,ITKDM,IP,IX
      REAL*8  PTRAK (ITKDM,*),PMISS (*)
 
      CALL PXZERV (4,PMISS)
      DO 160 IP = 1,NTRAK
          CALL PXADDV (4,PTRAK (1,IP),PMISS,PMISS)
 160  CONTINUE
      DO 110 IX = 1,3
          PMISS (IX) = - PMISS (IX)
 110  CONTINUE
      RETURN
      END

      SUBROUTINE PXEEC4 (NTRAK,ITKDM,PTRAK,ENORM,NBINS,
     +                   EEC,EECER,EECA,EECAER,CHAR,IERR)
*.*********************************************************
*. ------
*. PXEEC4
*. ------
*. Routine to calculate the Energy-Energy Correlation (EEC) and
*. Energy-Energy Correlation Asymmetry (EECA) distributions.
*. The EEC formula used is :
*.
*.                         NEVT  NTRAK
*. EEC(Thet) = (1/NEVT) *  SUM *  SUM { E(i)*E(j)/ENORM**2
*.                         k=1    i,j
*.                                         * (1/Dthet) * delta(i,j) }
*.
*. where Dthet is the angular bin size (determined automatically
*. from NBINS, see argument list below) and the delta function
*. delta(i,j) is defined by
*.
*. delta(i,j) = 1 if (Thet-DThet/2) < Angle(i,j) < (Thet+DThet/2)
*.            = 0 otherwise
*.
*. Please note that the errors are statistical only assuming no
*. correlation betweeen the entries in different bins.
*. Since this latter is not a valid presumption, the errors
*. returned are too small, probably by fifty percent or so.
*.
*. Usage     :
*.
*.      INTEGER*4  ITKDM,MXTRAK,NBINS
*.      PARAMETER  (ITKDM=4.or.more,MXTRAK=1.or.more)
*.      PARAMETER  (NBINS=50.or.60.or.something.like.that)
*.      INTEGER*4  NTRAK,IERR
*.      REAL*8  PTRAK (ITKDM,MXTRAK),EEC (NBINS),EECER (NBINS),
*.     +      EECA (NBINS/2),EECAER (NBINS/2)
*.      CHARACTER*1 CHAR
*.      REAL*8  ENORM
*.
*.      NTRAK = 1.to.MXTRAK
*.      CALL PXEEC4 (NTRAK,ITKDM,PTRAK,ENORM,NBINS,
*.     +             EEC,EECER,EECA,EECAER,CHAR,IERR)
*.
*. Example:  Normalize to Visible Energy, No self-correlations
*.
*.        DO 100  IEVT=1,NEVT
*.                  ...
*.         >>>  Generate.Event, calculate visible energy EVISIB
*.                  ...
*.*         Energy-Energy Correlation
*.*         ------ ------ -----------
*.           ENORM = EVISIB
*.           CALL PXEEC4 (NTRAK,ITKDM,PTRAK,ENORM,NBINS,
*.     +                  EEC,EECER,EECA,EECAER,' ',IERR)
*. 100    CONTINUE
*.*     Normalize EEC, Calculate EECA
*.*     --------- ---  --------- ----
*.        CALL PXEEC4 (NTRAK,ITKDM,PTRAK,ENORM,NBINS,
*.     +               EEC,EECER,EECA,EECAER,'N',IERR)
*.        END
*.
*. INPUT     : NTRAK   Total number of particles
*. INPUT     : ITKDM   First dimension of PTRAK
*. INPUT     : PTRAK   Particle 4-momentum (Px,Py,Pz,E)
*. INPUT     : ENORM   The energy with which to normalize
*.                     (either the visible energy or the C.M. energy)
*. INPUT     : NBINS   The number of bins for the EEC distribution
*.                     (typical=50; MUST BE AN EVEN VALUED INTEGER*4)
*. OUTPUT    : EEC     the Energy-Energy-Correlation distribution
*. OUTPUT    : EECER   the statistical errors for EEC
*. OUTPUT    : EECA    the EECA Asymmetry function distribution
*. OUTPUT    : EECAER  the statistical errors for EECA
*. CONTROL   : CHAR    control variable
*.                = ' ' (in event loop): no self-correlations
*.                = 'S' (in event loop): self-correlations included
*.                = 'N' (after event loop): normalize EEC and
*.                                          calculate EECA
*. OUTPUT    : IERR    = 0 if all is OK
*.
*. CALLS     : PXZERV,PXANG3
*. CALLED    : By User
*.
*. AUTHOR    :  M.Weber/J.W.Gary
*. CREATED   :  14-Feb-89
*. LAST MOD  :  05-Aug-90
*.
*. Modification Log.
*. 04-Mar-89   Calculation of statistical errors,  J.W.Gary
*. 05-Aug-90   double precision                    J.W.Gary
*.
*.*********************************************************
      INTEGER*4  NXEECB
      PARAMETER  (NXEECB=500)
      REAL*8  PIEEC
      PARAMETER  (PIEEC=3.141592653589793238d0)
      INTEGER*4  IERR,IECNT,NTRAK,ITKDM,NBINS,IB,IP1,IP2,IEND
      INTEGER*4  NEEC
      REAL*8  PTRAK (ITKDM,*),EEC (*),EECER (*),EECA (*),EECAER (*)
      REAL*8  ENORM,BINMID,COST,THET
      REAL*8  XBINSZ
      REAL*8  WEIGHT,WEIGH2,DEEC (NXEECB),DEECER (NXEECB),
     +                  DENOM,EECSUM,XFACT
      CHARACTER*1 CHAR
      LOGICAL  INIEEC
      SAVE  NEEC,XBINSZ,INIEEC,DEEC,DEECER,EECSUM
      DATA  INIEEC / .TRUE. /
      IERR = 0
      IF (CHAR.EQ.'N') GO TO 200
      IF (MOD (NBINS,2).NE.0) THEN
          WRITE (6,FMT='('' PXEEC4: Error, NBINS ='',I10,
     +      '' not an even number'')') NBINS
          IERR = -1
          GO TO 990
      END IF
      IF (NBINS.GT.NXEECB) THEN
          WRITE (6,FMT='('' PXEEC4: Error, NBINS ='',I10,
     +      '' must be smaller than NXEECB ='',I10)') NBINS,NXEECB
          IERR = -1
          GO TO 990
      END IF
      IF (ENORM.EQ.0) THEN
          WRITE (6,FMT='('' PXEEC4: Error, ENORM ='',E12.4)') ENORM
          IERR = -1
          GO TO 990
      END IF
*  initialize
*  ----------
      IF (INIEEC) THEN
          INIEEC = .FALSE.
          NEEC = 0
C          XBINSZ = (PIEEC / FLOAT (NBINS))
          XBINSZ = (2.d0 / FLOAT (NBINS))
          DO 105 IB = 1,NBINS
             DEEC (IB) = 0D0
             DEECER (IB) = 0D0
 105      CONTINUE
          EECSUM = 0D0
      END IF
      NEEC = NEEC + 1
*  set up self-correlation or no self-correlation
*  --- -- ---- ----------- -- -- ---- -----------
      IF (CHAR.EQ.'S') THEN
          IEND = NTRAK
          XFACT = 1D0
      ELSE
          XFACT = 2D0
      END IF
*  calculation of EEC
*  ----------- -- ---
      DO 180 IP1=1,NTRAK
          IF (XFACT.GT.1.5) IEND = IP1-1
          DO 160 IP2=1,IEND
              CALL PXANG3 (PTRAK (1,IP1),PTRAK (1,IP2),COST,THET,IERR)
              IF (IERR.EQ.-1) GO TO 990
C              IB = (THET/XBINSZ) + 1
              IB = ((COST+1.d0)/XBINSZ) + 1 
              IF (IB.LT.1.OR.IB.GT.NBINS) THEN
                 IB = NBINS
                 IF (IB.LT.1) IB = 1
              END IF
              WEIGHT = (PTRAK (4,IP2) * PTRAK (4,IP1)) / ENORM**2
              WEIGH2 = WEIGHT * WEIGHT
              DEEC (IB) = DEEC (IB) + XFACT * WEIGHT
              DEECER (IB) = DEECER (IB) + XFACT * WEIGH2
              EECSUM = EECSUM + XFACT * WEIGHT
 160      CONTINUE
 180  CONTINUE
      GO TO 990
*  normalization and calculation of asymmetry
*  ------------- --- ----------- -- ---------
C 200  WRITE (6,FMT='('' PXEEC4: Normalize EEC, EECA'')')
  200 CONTINUE
      IF (NEEC.EQ.0) THEN
          IERR = -1
          WRITE (6,FMT='('' PXEEC4: Error, NEEC ='',I10)') NEEC
          GO TO 990
      END IF
      DENOM = FLOAT (NEEC) * XBINSZ
C      WRITE (6,FMT='(''   EECSUM, EECSUM ='',2E12.4)')
C     +   EECSUM,EECSUM/FLOAT (NEEC)
      DO 220  IB=1,NBINS
          DEEC (IB) = DEEC (IB) / DENOM
          DEECER (IB) = DSQRT (DEECER (IB)) / DENOM
          EEC (IB) = DEEC (IB)
          EECER (IB) = DEECER (IB)
 220  CONTINUE
      DO 240 IB=1,NBINS/2
          EECA (IB) = DEEC (NBINS+1-IB) - DEEC (IB)
          EECAER (IB) = DSQRT (DEECER (NBINS+1-IB)**2 + DEECER (IB)**2)
 240  CONTINUE
      INIEEC = .TRUE.
      NEEC = 0
 
 990  RETURN
      END
c
c     -----------------------------------------------------------------------
c
      SUBROUTINE PXPTC4 (NTRAK,ITKDM,PTRAK,ENORM,NBINS,CHAR,
     +                   THETP,BETA,PTC,PTCI,PTCER,PTCIER,IERR)
*.*********************************************************
*. ------
*. PXPTC4
*. ------
*. Calculates planar triple energy correlations (PTC)
*.
*.                 d(j,k) * d(i,k)
*. PTC(ibi,ibj) =  --------------
*.                 NEVT * Dthet**2
*.
*.                      NEVT  NTRAK   Ei * Ej * Ek
*.                  *   SUM  * SUM  { ------------ }
*.                      n=1   i,j,k     ENORM**3
*.
*. where, ibi is the bin number corresponding to
*.            angle(j,k) = (ibi-0.5)*Dthet
*.        NEVT is the number of events
*.        Dthet is the width of the angular bin
*.        d(i,j) = 1 if:  ibi -1  <  Angle(i,j)/Dthet  < ibi
*.                 0 otherwise
*.
*. INPUT     : NTRAK    Total number of particles
*. INPUT     : ITKDM    First dimension of PTRAK
*. INPUT     : PTRAK    Particle 4-momentum (Px,Py,Pz,E)
*. INPUT     : ENORM    The energy with which to normalize
*. INPUT     : NBINS    The number of bins in each dimension
*. CONTROL   : CHAR     control variable
*.                      = ' ' (in event loop)
*.                      = 'N' (after event loop): normalize PTC
*. INPUT     : THETP    Maximum angle for planarity condition (rad.)
*. INPUT     : BETA     Angle for range of integration (rad.)
*. OUTPUT    : PTC      The Planar TEC
*. OUTPUT    : PTCI     Integral of PTC
*. OUTPUT    : PTCER    The statistical error for PTC
*. OUTPUT    : PTCIER   The statistical error for PTCI
*. OUTPUT    : IERR     = 0 if all is OK
*.
*. CALLS     : PXANG3
*. CALLED    : By User
*.
*. AUTHOR    : G. Azuelos
*. CREATED   :
*. LAST MOD  :
*.
*. Modification Log.
*. 24-Jul-90   Integration into PX library    J.W.Gary
*.
*.*********************************************************
      INTEGER*4  NXPTCB
      PARAMETER  (NXPTCB=50)
      REAL*8  PIPTC
      PARAMETER  (PIPTC=3.141593)
      INTEGER*4  NTRAK,ITKDM,NBINS,IFLAG,IERR,IBI,IBJ,IER,I,J,K
      INTEGER*4  NPTC
      REAL*8  PTC (NBINS,NBINS),PTCER (NBINS,NBINS),PTRAK(ITKDM,*)
      REAL*8  ENORM,BETA,PTCI,PTCIER,THETP,COST,XI,XJ,XK,PTCIET
      REAL*8  BPTSIZ,BPTSZ2
      REAL*8  WEIGHT,DPTC (NXPTCB,NXPTCB),DPTCI,DPTCIE,
     +                  DPTCER (NXPTCB,NXPTCB),DPTCIT,DENOM
      CHARACTER*1 CHAR
      LOGICAL  INIPTC
      SAVE BPTSIZ,BPTSZ2,NPTC,INIPTC
      DATA  INIPTC / .TRUE. /
 
      IERR = 0
      IF (CHAR.EQ.'N') GO TO 200
      IF (ENORM.EQ.0) THEN
         WRITE (6,FMT='('' PXPTC4: Error, ENORM ='',E12.4)') ENORM
         IERR = -1
         GO TO 990
      END IF
      IF (NBINS.GT.NXPTCB) THEN
          WRITE (6,FMT='('' PXPTC4: Error, NBINS ='',I10,
     +      '' must be smaller than NXPTCB ='',I10)') NBINS,NXPTCB
          IERR = -1
          GO TO 990
      END IF
*  initialize
*  ----------
      IF (INIPTC) THEN
         INIPTC = .FALSE.
         NPTC = 0
         BPTSIZ = PIPTC / FLOAT (NBINS)
         BPTSZ2 = BPTSIZ**2
         DO 120 I=1,NBINS
            DO 110 J=1,NBINS
               DPTC (I,J) = 0D0
               DPTCER (I,J) = 0D0
 110        CONTINUE
 120     CONTINUE
         DPTCI = 0D0
         DPTCIE = 0D0
         DPTCIT = 0D0
      END IF
      NPTC = NPTC + 1
      DO 180 I=1,NTRAK
         DO 170 J=1,NTRAK
            DO 160 K=1,NTRAK
               CALL PXANG3 (PTRAK(1,J),PTRAK(1,K),COST,XI,IERR)
               IF (IERR.NE.0) GO TO 990
               CALL PXANG3 (PTRAK(1,I),PTRAK(1,K),COST,XJ,IERR)
               IF (IERR.NE.0) GO TO 990
               CALL PXANG3 (PTRAK(1,I),PTRAK(1,J),COST,XK,IERR)
               IF (IERR.NE.0) GO TO 990
***  temporary to check normalization
Comment               WEIGHT = (PTRAK (4,I)/ENORM) * (PTRAK (4,J)/ENORM)
Comment     +                * (PTRAK (4,K)/ENORM)
Comment               DPTCIT = DPTCIT + WEIGHT
*************************************
*            select planar triplets
*            ------ ------ --------
               IF (ABS ((2*PIPTC)-XI-XJ-XK).LT.THETP) THEN
                  IBI = (XI/BPTSIZ) + 1
                  IBJ = (XJ/BPTSIZ) + 1
                  IF (IBI.LT.1) IBI = 1
                  IF (IBI.GT.NBINS) IBI = NBINS
                  IF (IBJ.LT.1) IBJ = 1
                  IF (IBJ.GT.NBINS) IBJ = NBINS
                  WEIGHT = (PTRAK (4,I)/ENORM) * (PTRAK (4,J)/ENORM)
     +                   * (PTRAK (4,K)/ENORM)
                  DPTC (IBI,IBJ) = DPTC (IBI,IBJ) + WEIGHT
                  DPTCER (IBI,IBJ) = DPTCER (IBI,IBJ) + WEIGHT * WEIGHT
*               integral of distribution up to BETA
*               -------- -- ------------ -- -- ----
                  IF ((XI.LT.PIPTC-BETA).AND.(XJ.LT.PIPTC-BETA).AND.
     +                (XI+XJ.GT.PIPTC+BETA-THETP)) THEN
Comment*                   MarkJ definition
Comment*                   ----- ----------
Comment     +                (XI+XJ.GT.PIPTC+BETA)) THEN
                     DPTCI = DPTCI + WEIGHT
                     DPTCIE = DPTCIE + WEIGHT * WEIGHT
                  END IF
               END IF
 160        CONTINUE
 170     CONTINUE
 180  CONTINUE
      GO TO 990
*  normalization
*  -------------
 200  CONTINUE
      IF (NPTC.EQ.0) THEN
         PTCI = 0.
         PTCIER = 0.
      ELSE
         DENOM = FLOAT (NPTC) * BPTSZ2
         DO 280 I=1,NBINS
            DO 260 J=1,NBINS
               PTC (I,J) = DPTC (I,J) / DENOM
               PTCER (I,J) = DSQRT (DPTCER (I,J)) / DENOM
 260        CONTINUE
 280     CONTINUE
         PTCI = DPTCI / FLOAT (NPTC)
         PTCIER = DSQRT (DPTCIE) / FLOAT (NPTC)
Comment         PTCIET = DPTCIT / FLOAT (NPTC)
Comment         WRITE (6,FMT='('' PXPTC4: PTCIET ='',E12.4)') PTCIET
      END IF
 
 990  RETURN
      END
c
c     -------------------------------------------------------------------------
c
      SUBROUTINE PXANG3 (VEC1,VEC2,COST,THET,IERR)
*.*********************************************************
*. ------
*. PXANG3
*. ------
*. SOURCE: VECSUB (V. Blobel)
*. Calculate the angle beteen two 3-vectors
*. Usage     :
*.
*.      INTEGER*4  IERR
*.      REAL*8  VEC1 (3.or.more),
*.     +      VEC2 (3.or.more)
*.      REAL*8  COST,THET
*.
*.      CALL PXANG3 (VEC1,VEC2,COST,THET,IERR)
*.
*. INPUT     : VEC1    The first vector
*. INPUT     : VEC2    The second vector
*. OUTPUT    : COST    Cosine of the angle between the vectors
*. OUTPUT    : THET    The angle between the vectors (radians)
*. OUTPUT    : IERR    = 0 if all is OK ;   = -1 otherwise
*.
*.*********************************************************
      REAL*8 AX,BX,CX,DX
      REAL*8  VEC1 (*),VEC2 (*)
      REAL*8  COST,THET
      INTEGER*4  IX,IERR
      IERR = 0
      AX = 0D0
      BX = 0D0
      CX = 0D0
      DO 120  IX = 1,3
          AX = AX + VEC1 (IX) * VEC1 (IX)
          BX = BX + VEC2 (IX) * VEC2 (IX)
          CX = CX + VEC1 (IX) * VEC2 (IX)
 120  CONTINUE
      DX = DSQRT (AX * BX)
      IF (DX.NE.0.0) THEN
          DX = CX / DX
      ELSE
          WRITE (6,FMT='('' PXANG3: Error, DX='',E12.4)') DX
          IERR = -1
          RETURN
      END IF
      IF (DABS (DX).GT.1.D0) DX = DSIGN (1.D0,DX)
      COST = DX
      THET = DACOS (DX)
      RETURN
      END
C
      program main
c s = sum(p**2)
          real*8 :: s
          real*8, dimension(4) :: p3,p4,p5,p6
          real*8, dimension(4, 4) :: EE,THET
          real*8 :: thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan
          real*8 :: acopla,rmin,rmax,Dpar,sve
          character(100) :: arg_in
          integer :: inp_arg_reached, num_inp_args
c check we got a valid number of arguments
          num_inp_args = command_argument_count()
          if(num_inp_args.gt.17) then
              print*, "Too many arguments"
              stop
          endif
          if(num_inp_args.lt.9) then
              print*, "Too few arguments"
              stop
          endif
c process the arguments
c when all the inputs have been used call the apropreate function
          call get_command_argument(1, arg_in)
          read(arg_in, *) s
          inp_arg_reached = 1
          do i=1,4
            call get_command_argument(i+inp_arg_reached, arg_in)
            read(arg_in, *) p3(i)
          end do
          inp_arg_reached = 5
          do i=1,4
            call get_command_argument(i+inp_arg_reached, arg_in)
            read(arg_in, *) p4(i)
          end do
          inp_arg_reached = 9
          if (num_inp_args.eq.inp_arg_reached) then
              call shape2(s,p3,p4,
     &                    thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan,
     &                    acopla,EE,THET,rmin,rmax,Dpar,sve)
              goto 130
          end if
          do i=1,4
            call get_command_argument(i+inp_arg_reached, arg_in)
            read(arg_in, *) p5(i)
          end do
          inp_arg_reached = 13
          if (num_inp_args.eq.inp_arg_reached) then
              call shape3(s,p3,p4,p5,
     &                    thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan,
     &                    acopla,EE,THET,rmin,rmax,Dpar,sve)
              goto 130
          end if
          do i=1,4
            call get_command_argument(i+inp_arg_reached, arg_in)
            read(arg_in, *) p6(i)
          end do
c if we get here it must be shape4
          call shape4(s,p3,p4,p5,p6,
     &                thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan,
     &                acopla,EE,THET,rmin,rmax,Dpar,sve)
c print the results
 130      continue
          print*, "thrust ", thr
          print*, "oblateness ", obl
          print*, "sphericity ", sfe
          print*, "heavy_jet_mass2 ", hjm
          print*, "light_jet_mass2 ", ljm
          print*, "difference_jet_mass2 ", djm
          print*, "alpanarity ", apla
          print*, "planarity ", plan
          print*, "acoplanarity ", acopla
          print*, "minor ", rmin
          print*, "major ", rmax
          print*, "D parameter ", Dpar
          print*, "C parameter ", Cpar
          print*, "spherocity ", sve
c          print*, "EE ", EE
c          print*, "THET ", THET
      end program main
