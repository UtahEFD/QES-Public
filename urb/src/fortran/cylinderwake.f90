!                                 Notice
!  This program was prepared by the University of California (University)
!  under Contract W-7405-ENG-36 with the U.S. Department of Energy (DOE).
!  All rights in the program are reserved by DOE on behalf of the Government
!  and the University pursuant to the contract. You are authorized to use
!  this program for Government purposes but it is not to be released or
!  distributed to the public.
!  NEITHER THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY,
!  NOR THE UNIVERSITY OF CALIFORNIA, NOR ANY OF THEIR EMPLOYEES,
!  MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY
!  OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS, OF
!  ANY INFORMATION, APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS
!  THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
!
      subroutine cylinderwake
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!  
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer circle_flag,in_subdomain
         real uo_h,vo_h,upwind_dir,xco,yco,tol,zb
         real farwake_exponent,farwake_factor,farwake_velocity
         real thetamax,thetamin,thetai,LoverH,WoverH
         real ynorm,radius,xnorm_bisect,cav_fac,wake_fac,eff_height
         real dxy,canyon_factor,xc,yc,dN,xwall,yw1,yw3,y1,y2
         integer x_idx,y_idx,x_idx_min,iu,ju,iv,jv,kk
         integer, allocatable :: ufarwake(:,:),vfarwake(:,:)
         integer ktop,kbottom
         
         if(bldtype(ibuild) .eq. 5 .and. atten(ibuild) .lt. 0.)then
            eff_height=0.8*(Ht(ibuild)-zfo_actual(ibuild))+zfo_actual(ibuild)
         else
            eff_height=Ht(ibuild)
         endif
         allocate(ufarwake(nx,ny),vfarwake(nx,ny))
         farwake_exponent=1.5
         farwake_factor=3
         xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))!CENTER of building in QUIC domain coordinates
         yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
         uo_h=uo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         vo_h=vo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
! find upwind direction and determine the type of flow regime
         upwind_dir=atan2(vo_h,uo_h)
         dxy=min(dx,dy)
         tol=0.01*dxy
         if(abs(aa(ibuild)-bb(ibuild)) .lt. tol)then
            circle_flag=1
         else
            circle_flag=0
         endif
         if(circle_flag .eq. 1)then
            thetamin=-0.5*pi
            thetamax=0.5*pi
            yw1=Lt(ibuild)
            yw3=-Lt(ibuild)
            Weff(ibuild)=Lti(ibuild)
            Leff(ibuild)=Lti(ibuild)
         else
            y1=0.
            y2=0.
            do i=1,180
               thetai=real(180-i)*pi/180.
               y1=radius(aa(ibuild),bb(ibuild),thetai,&
                   gamma(ibuild)-upwind_dir)*sin(thetai)
               if(y1 .lt. y2)then
                  exit
               endif
               y2=y1
            enddo
            thetamax=thetai+pi/180.
            thetamin=thetamax-pi
            yw1=y2
            yw3=-y2
            Weff(ibuild)=2*yw1
            Leff(ibuild)=2*radius(aa(ibuild),bb(ibuild),upwind_dir,gamma(ibuild))
         endif
         if(wakeflag .eq. 2)then
            cav_fac=1.1
            wake_fac=0.1
         else
            cav_fac=1.
            wake_fac=0.
         endif
         LoverH=Leff(ibuild)/eff_height
         WoverH=Weff(ibuild)/eff_height
         if(LoverH .gt. 3.)LoverH=3.
         if(LoverH .lt. 0.3)LoverH=0.3
         if(WoverH .gt. 10.)WoverH=10.
         Lr(ibuild)=0.9*eff_height*WoverH/((LoverH**(0.3))*(1+0.24*WoverH))
         in_subdomain=0
         if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
               xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
               yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
               yfo(ibuild)-dy.le.y_subdomain_end) in_subdomain=1
         ynorm=yw1
         if(in_subdomain .eq. 1)then
            ! MAN 07/25/2008 stretched vertical grid
            do k=2,kstart(ibuild)
               kbottom=k
               if(zfo(ibuild) .le. zm(k))exit
            enddo
            do k=kstart(ibuild),nz-1
               ktop=k
               if(eff_height .lt. zm(k+1))exit
            enddo
            do k=kstart(ibuild),kend(ibuild)
               kk=k
               if(0.75*(Ht(ibuild)-zfo_actual(ibuild))+zfo_actual(ibuild) .le. zm(k))exit
            enddo
lp003:      do k=ktop,kbottom,-1
               zb=zm(k)-zfo(ibuild)
               ufarwake(:,:)=0
               vfarwake(:,:)=0
lp002:         do y_idx=1,2*int((yw1-yw3)/dxy)-1
                  yc=0.5*real(y_idx)*dxy+yw3
                  if(circle_flag .eq. 1)then
                     xwall=sqrt((Lt(ibuild)**2.)-(yc**2.))
                  else
                     xwall=xnorm_bisect(aa(ibuild),bb(ibuild),&
                           gamma(ibuild)-upwind_dir,yc,thetamin,thetamax,dxy)
                  endif
                  !check for building that will disrupt the wake
                  canyon_factor=1.
                  x_idx_min=-1
                  do x_idx=1,int(Lr(ibuild)/dxy)+1
                     xc=real(x_idx)*dxy
                     i=int(((xc+xwall)*cos(upwind_dir)-yc*sin(upwind_dir)+xco)/dx)+1
                     j=int(((xc+xwall)*sin(upwind_dir)+yc*cos(upwind_dir)+yco)/dy)+1
                     if(i .ge. nx-1 .or. i .le. 1 .or. j .ge. ny-1 .or. j .le. 1)then
                        exit
                     endif
                     if(icellflag(i,j,kk) .ne. 0 .and. x_idx_min .lt. 0)then
                        x_idx_min=x_idx
                     endif
                     if(icellflag(i,j,kk) .eq. 0 .and. ibldflag(i,j,kk) .ne. ibuild .and. x_idx_min .gt. 0)then
                        canyon_factor=xc/Lr(ibuild)
                        exit
                     endif
                  enddo
                  dN=sqrt((1.-(yc/ynorm)**2.)*(1.-((zb)/eff_height)**2.)*(canyon_factor*Lr(ibuild))**2)
                  x_idx_min=-1
lp001:            do x_idx=0,2*int(farwake_factor*dN/dxy)+1
                     xc=0.5*real(x_idx)*dxy
                     i=int(((xc+xwall)*cos(upwind_dir)-yc*sin(upwind_dir)+xco)/dx)+1
                     j=int(((xc+xwall)*sin(upwind_dir)+yc*cos(upwind_dir)+yco)/dy)+1
                     if(i .ge. nx-1 .or. i .le. 1 .or. j .ge. ny-1 .or. j .le. 1)then
                        exit
                     endif
                     if(icellflag(i,j,k) .ne. 0 .and. x_idx_min .lt. 0)then
                        x_idx_min=x_idx
                     endif
                     if(icellflag(i,j,k) .eq. 0)then
                        if(x_idx_min .ge. 0)then
                           if(ibldflag(i,j,k) .eq. ibuild)then
                              x_idx_min=-1
                           elseif(canyon_factor .lt. 1.)then
                              exit
                           endif
                        endif
                     endif
! u values
                     iu=nint(((xc+xwall)*cos(upwind_dir)-yc*sin(upwind_dir)+xco)/dx)+1
                     ju=int(((xc+xwall)*sin(upwind_dir)+yc*cos(upwind_dir)+yco)/dy)+1
! Far wake
                     if(icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i-1,j,k) .ne. 0
                        if(xc .gt. dN)then
                           farwake_velocity=uo(iu,ju,k)*(1.-(dN/(xc+wake_fac*dN))**(farwake_exponent))
                           if(abs(farwake_velocity) .lt. abs(uo(iu,ju,k)) .and. &
                                 icellflag(i,j,k) .ne. 4 .and. ufarwake(iu,ju) .eq. 0 .and. &
                                 canyon_factor .eq. 1.)then
                              uo(iu,ju,k)=farwake_velocity
                              ufarwake(iu,ju)=1
                              wo(i,j,k)=0.
                              if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=5
                           endif
! Cavity             
                        else
                           uo(iu,ju,k)=-uo_h*min((1.-xc/(cav_fac*dN))**2.,1.)
                           if(abs(uo(iu,ju,k)) .gt. max_velmag)then
                              print*,'Parameterized U exceeds max in rectangle wake',&
                                 uo(iu,ju,k),max_velmag,iu,ju,k
                           endif
                           wo(iu,ju,k)=0.
                           if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=4
                        endif
                     endif
! v values
                     iv=int(((xc+xwall)*cos(upwind_dir)-yc*sin(upwind_dir)+xco)/dx)+1
                     jv=nint(((xc+xwall)*sin(upwind_dir)+yc*cos(upwind_dir)+yco)/dy)+1
! Far wake
                     if(icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i,j-1,k) .ne. 0
                        if(xc .gt. dN)then
                           farwake_velocity=vo(iv,jv,k)*(1.-(dN/(xc+wake_fac*dN))**(farwake_exponent))
                           if(abs(farwake_velocity) .lt. abs(vo(iv,jv,k)) .and. &
                                 icellflag(i,j,k) .ne. 4 .and. vfarwake(iv,jv) .eq. 0 .and. &
                                 canyon_factor .eq. 1.)then
                              vo(iv,jv,k)=farwake_velocity
                              vfarwake(iv,jv)=1
                              wo(iv,jv,k)=0.
                              if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=5
                           endif
! Cavity
                        else
                           vo(iv,jv,k)=-vo_h*min((1.-xc/(cav_fac*dN))**2.,1.)
                           if(abs(vo(iv,jv,k)) .gt. max_velmag)then
                              print*,'Parameterized V exceeds max in rectangle wake',&
                                 vo(iv,jv,k),max_velmag,iv,jv,k
                           endif
                           wo(iv,jv,k)=0.
                           if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=4
                        endif
                     endif
                  enddo   lp001      
               enddo   lp002      
            enddo   lp003
         endif
         deallocate(ufarwake,vfarwake)
         return
      end
      
      
      real function radius(a,b,gamma,theta)
         implicit none
         real a,b,gamma,theta
         radius=a*b/sqrt( (a*sin(theta-gamma))**2. + (b*cos(theta-gamma))**2. )
         return
      end
      
      
      real function xnorm_bisect(a,b,gamma,y,thetamin,thetamax,dxy)
         implicit none
         integer i
         real, intent(in) :: a,b,gamma,y,dxy,thetamin,thetamax
         real yguess,yguess_low,eps,prod,theta,rad,radius,thetalow,thetahigh
         i=0
         eps=a+b
         thetalow=thetamin
         thetahigh=thetamax
         do while (i .lt. 100 .and. eps .gt. 0.1*dxy)
            i=i+1
            theta=0.5*(thetalow+thetahigh)
            rad=radius(a,b,gamma,theta)
            yguess=rad*sin(theta)-y
            yguess_low=radius(a,b,gamma,thetalow)*sin(thetalow)-y
            eps=abs(yguess)
            prod=yguess*yguess_low
            if(prod .lt. 0)then
               thetahigh=theta
            elseif(prod .gt. 0)then
               thetalow=theta
            else
               eps=0.
            endif
         enddo
         xnorm_bisect=rad*cos(theta)
         return
      end