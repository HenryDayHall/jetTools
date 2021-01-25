c Contents;
c  - shape3
c  - order3_p (used in shape3)
c To make python hooks; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c      $ f2py -c shape3.f shape4.f -m shape3
c To use python hooks;
c      $ ipython3
c In [0]: import shape3
c In [1]: shape3.shape3(0., [1,2,3,4], [5,6,7,8], [9, 10, 11, 12])
c Out[2]: stuff......
c ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      subroutine shape3(s,p3,p4,p5,
     &                  thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan,
     &                  acopla,EE,THET,rmin,rmax,Dpar,sve)
      implicit real*8 (a-h,o-z)
      real*8 ljm2,ljm
      common/taxis3/tv3(3)
      real, intent(in) :: s
      real*8, dimension(4), intent(in) :: p3,p4,p5
      dimension ptrak(3,3)
      dimension EVAL(3),EVEC(3,3) 
      dimension pm1(4),pm2(4),pm3(4)
      dimension pn1(3),pn2(3),pn3(3)
      dimension T(2,2)
      real*8, dimension(4, 4), intent(out) :: EE,THET
      parameter (nmax=8,ia=nmax,iv=nmax)
      dimension a(ia,nmax),e(nmax),rr(nmax),v(iv,nmax)
      parameter (pi=3.1415926536d0) 
      intent(out) :: thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan
      intent(out) :: acopla,rmin,rmax,Dpar,sve
c ordino impulsi secondo il modulo (pmod1>pmod2>pmod3).
      call order3_p(p3,p4,p5,pm1,pm2,pm3,pmod1,pmod2,pmod3)
c serve per il C- and D- parameter.
      do i=1,3
        ptrak(i,1)=p3(i)
        ptrak(i,2)=p4(i)
        ptrak(i,3)=p5(i)
      end do
c calcolo thrust (asse di thrust lungo pm1).
      do i=1,3
        tv3(i)=pm1(i)/pmod1
      end do
      thr=2.d0*pmod1/(pmod1+pmod2+pmod3)      
c calcolo oblateness.
      ct1=1.d0
      ct2=(pm1(1)*pm2(1)+pm1(2)*pm2(2)+pm1(3)*pm2(3))/pmod1/pmod2
      ct3=(pm1(1)*pm3(1)+pm1(2)*pm3(2)+pm1(3)*pm3(3))/pmod1/pmod3
      st1=0.d0
      st2=sqrt(abs(1.d0-ct2*ct2))
      st3=sqrt(abs(1.d0-ct3*ct3))	
      ptr1=0.d0
      ptr2=abs(pmod2*st2)
      ptr3=abs(pmod3*st3)
      obl=(ptr1+ptr2+ptr3)/(pmod1+pmod2+pmod3)
      rmax=obl
      rmin=0.d0
c calcolo heavy jet mass squared.
      hjm2=(pm2(4)+pm3(4))**2
      do i=1,3
        hjm2=hjm2-(pm2(i)+pm3(i))**2
      end do
c      hjm=sqrt(abs(hjm2/s))
      hjm=abs(hjm2/s)
c calcolo light jet mass squared.
      ljm2=pm1(4)**2
      do i=1,3
        ljm2=ljm2-pm1(i)**2
      end do
c      ljm=sqrt(abs(ljm2/s))
      ljm=abs(ljm2/s)
      ljm=0.d0
c calcolo difference jet mass squared.
c      djm=sqrt(abs((hjm2-ljm2)/s))
      djm=abs((hjm2-ljm2)/s)
c serve per C-parameter (T) and for sphericity, aplanarity and planarity (a).
      pn1(1)=pmod1
      do i=2,3
        pn1(i)=0.d0
      end do      
      pn2(1)=pmod2*ct2
      pn3(1)=pmod3*ct3
      pn2(2)=pmod2*st2
      pn3(2)=-pmod3*st3
      do i=3,3
        pn2(i)=0.d0
        pn3(i)=0.d0
      end do
      denpa=pmod1+pmod2+pmod3
      denpb=pmod1**2+pmod2**2+pmod3**2
      do i=1,2
        do j=1,2
          T(i,j)=pn1(i)*pn1(j)/pmod1
     &          +pn2(i)*pn2(j)/pmod2  
     &	        +pn3(i)*pn3(j)/pmod3  
          a(i,j)=pn1(i)*pn1(j)
     &          +pn2(i)*pn2(j)
     &		+pn3(i)*pn3(j)
          T(i,j)=T(i,j)/denpa
          a(i,j)=a(i,j)/denpb
        end do
      end do
c calcolo C- and D-parameter.
      deter=T(1,1)*T(2,2)-T(1,2)*T(2,1)
      Cpar=3.d0*deter
      Dpar=0.d0
      n=3
      ifail=1
c serve per sphericity, aplanarity e planarity.
      call PXJSP3(3,3,ptrak,EVAL,EVEC,IERR)
c calcolo sphericity.
      sfe=3.d0*(eval(1)+eval(2))/2.d0
c calcolo aplanarity.
      apla=3.d0/2.d0*eval(1)
c calcolo planarity.
      plan=eval(2)-eval(1)
c calcolo acoplanarity.
      acopla=0.d0
c calcolo spherocity.
      sve1=64.d0/pi/pi*(1.d0-2.d0*pmod1/denpa)
     &                *(1.d0-2.d0*pmod2/denpa)
     &                *(1.d0-2.d0*pmod3/denpa)
     &                /thr/thr
      sve2=16.d0/pi/pi/denpa/denpa
     &   *(abs(pmod1*st1)+abs(pmod2*st2)+abs(pmod3*st3))**2
      sve=sve1
c serve energy-energy correlation and its asymmetry.
      tmod3=sqrt(abs(p3(1)**2+p3(2)**2+p3(3)**2))
      tmod4=sqrt(abs(p4(1)**2+p4(2)**2+p4(3)**2))
      tmod5=sqrt(abs(p5(1)**2+p5(2)**2+p5(3)**2))
      EE(1,2)=p3(4)*p4(4)/s
      EE(1,3)=p3(4)*p5(4)/s
      EE(2,3)=p4(4)*p5(4)/s
      EE(2,1)=EE(1,2)
      EE(3,1)=EE(1,3)
      EE(3,2)=EE(2,3)
      EE(1,1)=p3(4)*p3(4)/s
      EE(2,2)=p4(4)*p4(4)/s
      EE(3,3)=p5(4)*p5(4)/s
      THET(1,2)=(p3(1)*p4(1)+p3(2)*p4(2)+p3(3)*p4(3))/tmod3/tmod4
      THET(1,3)=(p3(1)*p5(1)+p3(2)*p5(2)+p3(3)*p5(3))/tmod3/tmod5
      THET(2,3)=(p4(1)*p5(1)+p4(2)*p5(2)+p4(3)*p5(3))/tmod4/tmod5
      do i=1,2
        do j=i+1,3
          if(THET(i,j).gt.1.d0)THET(i,j)=1.d0
          if(THET(i,j).lt.-1.d0)THET(i,j)=-1.d0
        end do
      end do
      THET(1,2)=dacos(THET(1,2))
      THET(1,3)=dacos(THET(1,3))
      THET(2,3)=dacos(THET(2,3))
      THET(2,1)=THET(1,2)
      THET(3,1)=THET(1,3)
      THET(3,2)=THET(2,3)
      do i=1,3
        THET(i,i)=0.d0
      end do
      return
      end
c
c     ----------------------------------------------------------------------
c
      subroutine order3_p(a,b,c,aa,bb,cc,pmodaa,pmodbb,pmodcc)
      implicit real*8 (a-h,o-z)
      dimension a(4),b(4),c(4)
      dimension aa1(4),bb1(4),cc1(4)
      dimension aa2(4),bb2(4),cc2(4)
      dimension aa3(4),bb3(4),cc3(4)
      dimension aa(4),bb(4),cc(4)
      pmoda=sqrt(abs(a(1)*a(1)+a(2)*a(2)+a(3)*a(3)))
      pmodb=sqrt(abs(b(1)*b(1)+b(2)*b(2)+b(3)*b(3)))
      pmodc=sqrt(abs(c(1)*c(1)+c(2)*c(2)+c(3)*c(3)))
      pmodaa1=pmoda
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
      pmodaa=pmodaa3
      pmodbb=pmodbb3
      pmodcc=pmodcc3
      do i=1,4
        aa(i)=aa3(i)
        bb(i)=bb3(i)
        cc(i)=cc3(i)
      end do
      return
      end
