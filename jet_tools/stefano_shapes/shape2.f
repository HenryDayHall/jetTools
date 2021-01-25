c To make python hooks; ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c      $ f2py -c shape2.f -m shape2
c To use python hooks;
c      $ ipython3
c In [0]: import shape2
c In [1]: shape2.shape2(0., [1,2,3,4], [5,6,7,8])
c Out[2]: stuff......
c ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      subroutine shape2(s,p3,p4,
     &                  thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan,
     &                  acopla,EE,THET,rmin,rmax,Dpar,sve)
      implicit real*8 (a-h,o-z)
      real*8 ljm2
      common/taxis2/tv2(3)
      real, intent(in) :: s
      real*8, dimension(4), intent(in) :: p3,p4
      real*8, dimension(4, 4), intent(out) :: EE,THET
      intent(out) :: thr,obl,hjm,ljm,djm,Cpar,sfe,apla,plan
      intent(out) :: acopla,rmin,rmax,Dpar,sve

c calcolo thrust (asse di thrust lungo p3).
      do i=1,3
        tv2(i)=p3(i)/dsqrt(p3(1)**2+p3(2)**2+p3(3)**2)
      end do
      thr=1.d0
c calcolo oblateness.
      obl=0.d0
c calcolo heavy jet mass squared.
      hjm=0.d0
c calcolo light jet mass squared.
      ljm=0.d0
c calcolo difference jet mass squared.
      djm=0.d0
c calcolo C- and D-parameter.
      Cpar=0.d0
      Dpar=0.d0
c calcolo sphericity.
      sfe=0.d0
c calcolo aplanarity.
      apla=0.d0
c calcolo planarity.
      plan=0.d0
c calcolo acoplanarity.
      acopla=0.d0
c calcolo spherocity.
      sve=0.d0
c calcolo minor and major.
      rmin=0.d0
      rmax=0.d0
c serve energy-energy correlation and its asymmetry.
      EE(1,2)=1.d0/4.d0
      EE(2,1)=EE(1,2)
      do i=1,2
        EE(i,i)=1.d0/4.d0
      end do
      THET(1,2)=dacos(-1.d0)
      THET(2,1)=THET(1,2)
      do i=1,2
        THET(i,i)=0.d0
      end do
      return
      end
