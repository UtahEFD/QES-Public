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
      subroutine bridgewake
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer perpendicular_flag,in_subdomain
         real uo_h,vo_h,upwind_dir,upwind_rel,xco,yco
         real x1,y1,x2,y2,x3,y3,x4,y4
         real xw1,yw1,xw2,yw2,xw3,yw3,tol,zb,ynorm
         real farwake_exponent,farwake_factor,farwake_velocity
         real cav_fac,wake_fac,beta,LoverH,upwind_rel_norm
         real bridge_thickness,dxy,xc,yc,dN,xwall
         integer x_idx,y_idx,x_idx_min,iu,ju,iv,jv
         integer, allocatable :: ufarwake(:,:),vfarwake(:,:)
         integer ktop,kbottom
         
         allocate(ufarwake(nx,ny),vfarwake(nx,ny))
         dxy=min(dx,dy)
         xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))!CENTER of building in QUIC domain coordinates
         yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
         ! find upwind direction and determine the type of flow regime
         uo_h=uo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         vo_h=vo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         upwind_dir=atan2(vo_h,uo_h)
         upwind_rel=upwind_dir-gamma(ibuild)
         if(upwind_rel.gt.pi)upwind_rel=upwind_rel-2*pi
         if(upwind_rel.le.-pi)upwind_rel=upwind_rel+2*pi
         upwind_rel_norm=upwind_rel+0.5*pi
         if(upwind_rel_norm .gt. pi)upwind_rel_norm=upwind_rel_norm-2*pi
         beta=abs(atan2(Lti(ibuild),Wti(ibuild)))
         if(abs(upwind_rel) .gt. 0.5*pi-beta .and. &
               abs(upwind_rel) .lt. 0.5*pi+beta)then
            Leff(ibuild)=abs(Wti(ibuild)/sin(upwind_rel))
         else
            Leff(ibuild)=abs(Lti(ibuild)/cos(upwind_rel))
         endif
         if(wakeflag .eq. 2)then
            cav_fac=1.1
            wake_fac=0.1
            if(abs(upwind_rel_norm) .gt. 0.5*pi-beta .and. &
                  abs(upwind_rel_norm) .lt. 0.5*pi+beta)then
               Weff(ibuild)=abs(Wti(ibuild)/sin(upwind_rel_norm))
            else
               Weff(ibuild)=abs(Lti(ibuild)/cos(upwind_rel_norm))
            endif
         else
            cav_fac=1.
            wake_fac=0.
            Weff(ibuild)=Lti(ibuild)*sin(abs(phiprime(ibuild)&
                  -gamma(ibuild)))+Wti(ibuild)*cos(abs(phiprime(ibuild)-gamma(ibuild)))
         endif
         bridge_thickness=0.5*(Ht(ibuild)-zfo(ibuild))
         LoverH=Leff(ibuild)/bridge_thickness
         if(LoverH.gt.3.)LoverH=3.
         if(LoverH.lt.0.3)LoverH=0.3
         Lr(ibuild)=0.9*Weff(ibuild)/((LoverH**(0.3))*   &
                        (1+0.24*Weff(ibuild)/bridge_thickness))
         tol=0.01*pi/180.
         farwake_exponent=1.5
         farwake_factor=3
         !Location of corners relative to the center of the building
         x1=xfo(ibuild)+Wt(ibuild)*sin(gamma(ibuild))-xco
         y1=yfo(ibuild)-Wt(ibuild)*cos(gamma(ibuild))-yco
         x2=x1+Lti(ibuild)*cos(gamma(ibuild))
         y2=y1+Lti(ibuild)*sin(gamma(ibuild))
         x4=xfo(ibuild)-Wt(ibuild)*sin(gamma(ibuild))-xco
         y4=yfo(ibuild)+Wt(ibuild)*cos(gamma(ibuild))-yco
         x3=x4+Lti(ibuild)*cos(gamma(ibuild))
         y3=y4+Lti(ibuild)*sin(gamma(ibuild))
         if(upwind_rel .gt. 0.5*pi+tol .and. upwind_rel .lt. pi-tol)then
            xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            xw2=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yw2=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            perpendicular_flag=0
         elseif(upwind_rel .ge. 0.5*pi-tol .and. upwind_rel .le. 0.5*pi+tol)then
            xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            perpendicular_flag=1
         elseif(upwind_rel .gt. tol .and. upwind_rel .lt. 0.5*pi-tol)then
            xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            xw2=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yw2=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            perpendicular_flag=0
         elseif(abs(upwind_rel) .le. tol)then
            xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            perpendicular_flag=1
         elseif(upwind_rel .lt. -tol .and. upwind_rel .gt. -0.5*pi+tol)then
            xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            xw2=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yw2=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            perpendicular_flag=0
         elseif(upwind_rel .lt. -0.5*pi+tol .and. upwind_rel .gt. -0.5*pi-tol)then
            xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            perpendicular_flag=1
         elseif(upwind_rel .lt. -0.5*pi-tol .and. upwind_rel .gt. -pi+tol)then
            xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            xw2=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yw2=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            perpendicular_flag=0
         else
            xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            perpendicular_flag=1
         endif
         in_subdomain=0
         if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
               xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
               yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
               yfo(ibuild)-dy.le.y_subdomain_end) in_subdomain=1
         if(in_subdomain .eq. 1)then
            do k=2,kstart(ibuild)
               kbottom=k
               if(zfo(ibuild) .le. zm(k))exit
            enddo
            do k=kstart(ibuild),nz-1
               ktop=k
               if(Ht(ibuild) .lt. zm(k+1))exit
            enddo
lp003:      do k=ktop,kbottom,-1
               zb=zm(k)-(bridge_thickness+zfo(ibuild))
               ufarwake(:,:)=0
               vfarwake(:,:)=0
lp002:         do y_idx=1,2*int((yw1-yw3)/dxy)-1
                  yc=0.5*real(y_idx)*dxy+yw3
                  if(perpendicular_flag .gt. 0)then
                     xwall=xw1
                  elseif(yc.ge.yw2)then
                     xwall=((xw2-xw1)/(yw2-yw1))*(yc-yw1)+xw1
                  else
                     xwall=((xw3-xw2)/(yw3-yw2))*(yc-yw2)+xw2
                  endif
                  if(yc.ge.0.)then
                     ynorm=yw1
                  else
                     ynorm=yw3
                  endif    
                  dN=sqrt((1.-(yc/ynorm)**2.)*(1.-((zb)/bridge_thickness)**2.)*(Lr(ibuild))**2)
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
                           else
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
                                 icellflag(i,j,k) .ne. 4 .and. ufarwake(iu,ju) .eq. 0)then
                              uo(iu,ju,k)=farwake_velocity
                              ufarwake(iu,ju)=1
                              wo(i,j,k)=0.
!                              if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=5
                           endif
! Cavity             
                        else
                           uo(iu,ju,k)=-uo_h*min((1.-xc/(cav_fac*dN))**2.,1.)
                           if(abs(uo(iu,ju,k)) .gt. max_velmag)then
                              print*,'Parameterized U exceeds max in rectangle wake',&
                                 uo(iu,ju,k),max_velmag,iu,ju,k
                           endif
                           wo(iu,ju,k)=0.
!                           if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=4
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
                                 icellflag(i,j,k) .ne. 4 .and. vfarwake(iv,jv) .eq. 0)then
                              vo(iv,jv,k)=farwake_velocity
                              vfarwake(iv,jv)=1
                              wo(iv,jv,k)=0.
!                              if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=5
                           endif
! Cavity
                        else
                           vo(iv,jv,k)=-vo_h*min((1.-xc/(cav_fac*dN))**2.,1.)
                           if(abs(vo(iv,jv,k)) .gt. max_velmag)then
                              print*,'Parameterized V exceeds max in rectangle wake',&
                                 vo(iv,jv,k),max_velmag,iv,jv,k
                           endif
                           wo(iv,jv,k)=0.
!                           if(icellflag(i,j,k) .ne. 0)icellflag(i,j,k)=4
                        endif
                     endif
                     if(icellflag(i,j,k) .ne. 0)then
                        if(xc .gt. dN)then
                           if(icellflag(i,j,k) .ne. 4)icellflag(i,j,k)=5
                        else
                           icellflag(i,j,k)=4
                        endif
                     endif
                  enddo   lp001      
               enddo   lp002      
            enddo   lp003
         endif
         deallocate(ufarwake,vfarwake)
         return
      end
