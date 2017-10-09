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
      subroutine rooftop
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer perpendicular_flag,in_subdomain,ns_flag
         integer roofflag_temp,uflag,vflag,wflag,k_ref,kendv
         integer k_shell,kk
         real uo_h,vo_h,upwind_dir,upwind_rel,xco,yco
         real x1,y1,x2,y2,x3,y3,x4,y4
         real tol,xfront,yfront
         real zr,x_u,y_u,x_v,y_v,x_w,y_w
         real vd,hd,Bs,BL,roofangle,hx,hy
         real xnorm,ynorm,vel_mag,zref2
         real shell_height
         
         xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))!CENTER of building in QUIC domain coordinates
         yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
         ! find upwind direction and determine the type of flow regime
         uo_h=uo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         vo_h=vo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         upwind_dir=atan2(vo_h,uo_h)
         upwind_rel=upwind_dir-gamma(ibuild)
         tol=30*pi/180.
         if(upwind_rel.gt.pi)upwind_rel=upwind_rel-2*pi
         if(upwind_rel.le.-pi)upwind_rel=upwind_rel+2*pi
         in_subdomain=0
         if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
               xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
               yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
               yfo(ibuild)-dy.le.y_subdomain_end .and. &
               upwindflag .gt. 0) in_subdomain=1
         !Location of corners relative to the center of the building
         x1=xfo(ibuild)+Wt(ibuild)*sin(gamma(ibuild))-xco
         y1=yfo(ibuild)-Wt(ibuild)*cos(gamma(ibuild))-yco
         x2=x1+Lti(ibuild)*cos(gamma(ibuild))
         y2=y1+Lti(ibuild)*sin(gamma(ibuild))
         x4=xfo(ibuild)-Wt(ibuild)*sin(gamma(ibuild))-xco
         y4=yfo(ibuild)+Wt(ibuild)*cos(gamma(ibuild))-yco
         x3=x4+Lti(ibuild)*cos(gamma(ibuild))
         y3=y4+Lti(ibuild)*sin(gamma(ibuild))
         perpendicular_flag=0
         if(upwind_rel .gt. 0.5*pi+tol .and. upwind_rel .lt. pi-tol)then
            xfront=Lt(ibuild)
            yfront=-Wt(ibuild)
            perpendicular_flag=0
            roofangle=0.0513*exp(1.7017*(abs(0.5*pi-upwind_rel)-2*abs(0.75*pi-upwind_rel)))
            xnorm=gamma(ibuild)!+roofangle
            ynorm=gamma(ibuild)-0.5*pi!-roofangle
         elseif(upwind_rel .ge. 0.5*pi-tol .and. upwind_rel .le. 0.5*pi+tol)then
            xfront=Lt(ibuild)
            yfront=-Wt(ibuild)
            perpendicular_flag=1
            ns_flag=1
         elseif(upwind_rel .gt. tol .and. upwind_rel .lt. 0.5*pi-tol)then
            xfront=-Lt(ibuild)
            yfront=-Wt(ibuild)
            perpendicular_flag=0
            roofangle=0.0513*exp(1.7017*(abs(upwind_rel)-2*abs(0.25*pi-upwind_rel)))
            xnorm=gamma(ibuild)+pi!-roofangle
            ynorm=gamma(ibuild)-0.5*pi!+roofangle
         elseif(abs(upwind_rel) .le. tol)then
            xfront=-Lt(ibuild)
            yfront=-Wt(ibuild)
            perpendicular_flag=1
            ns_flag=0
         elseif(upwind_rel .lt. -tol .and. upwind_rel .gt. -0.5*pi+tol)then
            xfront=-Lt(ibuild)
            yfront=Wt(ibuild)
            perpendicular_flag=0
            roofangle=0.0513*exp(1.7017*(abs(upwind_rel)-2.0*abs(-0.25*pi-upwind_rel)))
            xnorm=gamma(ibuild)-pi!+roofangle
            ynorm=gamma(ibuild)+0.5*pi!-roofangle
         elseif(upwind_rel .lt. -0.5*pi+tol .and. upwind_rel .gt. -0.5*pi-tol)then
            xfront=-Lt(ibuild)
            yfront=Wt(ibuild)
            perpendicular_flag=1
            ns_flag=1
         elseif(upwind_rel .lt. -0.5*pi-tol .and. upwind_rel .gt. -pi+tol)then
            xfront=Lt(ibuild)
            yfront=Wt(ibuild)
            perpendicular_flag=0
            roofangle=0.0513*exp(1.7017*(abs(-0.5*pi-upwind_rel)-2.0*abs(-0.75*pi-upwind_rel)))
            xnorm=gamma(ibuild)!-roofangle
            ynorm=gamma(ibuild)+0.5*pi!+roofangle
         else
            xfront=Lt(ibuild)
            yfront=Wt(ibuild)
            perpendicular_flag=1
            ns_flag=0
         endif
         ! MAN 07/25/2008 stretched vertical grid
         do k=kend(ibuild)+1,nz-1
            k_ref=k
            if(1.5*Ht(ibuild) .lt. z(k))exit
         enddo
         if(k_ref .gt. nz)then
            in_subdomain=0
         endif
         if(in_subdomain .eq. 1)then
            Bs=min(Weff(ibuild),Ht(ibuild))
            BL=max(Weff(ibuild),Ht(ibuild))
            Rscale(ibuild) = ((Bs**(2./3.))*(BL**(1./3.)))
            Rcx(ibuild)=(0.9*Rscale(ibuild))
            vd= 0.5*0.22*Rscale(ibuild)
            ! MAN 07/25/2008 stretched vertical grid
            do k=kend(ibuild)+1,k_ref
               kendv=k
               if(Ht(ibuild)+vd .lt. zm(k))exit
            enddo
            uo_roof=uo
            vo_roof=vo
            if(roofflag .eq. 2 .and. (bldtype(ibuild) .eq. 1 .or. bldtype(ibuild) .eq. 10) &
                  .and. rooftop_flag(ibuild) .eq. 1)then
               roofflag_temp=2
            elseif(roofflag .eq. 0)then
               roofflag_temp=0
               rooftop_flag(ibuild)=0
            else
               roofflag_temp=1
               rooftop_flag(ibuild)=0
            endif
            !print*,"rooftype      = ",roofflag
						!print*,"doRooftop     = ",rooftop_flag(ibuild)
            !print*,'roofflag_temp = ',roofflag_temp
            
lp003:      do k=kend(ibuild)+1,k_ref
               ! MAN 07/25/2008 stretched vertical grid
               zr=zm(k)-Ht(ibuild)
               vel_mag=sqrt((uo_roof(nint(xco/dx),nint(yco/dy),k)**2.)+&
                       (vo_roof(nint(xco/dx),nint(yco/dy),k)**2.))
lp002:         do j=jstart(ibuild),jend(ibuild)
lp001:            do i=istart(ibuild),iend(ibuild)
                     uflag=0
                     vflag=0
                     wflag=0
                     !check to see if velocity vector is above the building or in a street canyon cell
                     if(icellflag(i,j,kend(ibuild)) .eq. 0)then
                        uflag=1
                        vflag=1
                        wflag=1
                     else
                        if(icellflag(i-1,j,kend(ibuild)) .eq. 0)uflag=1
                        if(icellflag(i,j-1,kend(ibuild)) .eq. 0)vflag=1
                     endif
                     if(icellflag(i,j,k) .eq. 6 .or. icellflag(i,j,k) .eq. 4)then
                        uflag=0
                        vflag=0
                        wflag=0
                     endif
                     x_u=((real(i)-1)*dx-xco)*cos(gamma(ibuild))+ &
                           ((real(j)-0.5)*dy-yco)*sin(gamma(ibuild))
                     y_u=-((real(i)-1)*dx-xco)*sin(gamma(ibuild))+ &
                           ((real(j)-0.5)*dy-yco)*cos(gamma(ibuild))
                     x_v=((real(i)-0.5)*dx-xco)*cos(gamma(ibuild))+ &
                           ((real(j)-1)*dy-yco)*sin(gamma(ibuild))
                     y_v=-((real(i)-0.5)*dx-xco)*sin(gamma(ibuild))+	&
                           ((real(j)-1)*dy-yco)*cos(gamma(ibuild))
                     x_w=((real(i)-0.5)*dx-xco)*cos(gamma(ibuild))+ &
                           ((real(j)-0.5)*dy-yco)*sin(gamma(ibuild))
                     y_w=-((real(i)-0.5)*dx-xco)*sin(gamma(ibuild))+	&
                           ((real(j)-0.5)*dy-yco)*cos(gamma(ibuild))
                     select case(roofflag_temp)
                        case(1)
                           if(uflag .eq. 1)then
                              if(perpendicular_flag .gt. 0)then
                                 if(ns_flag .eq. 1)then
                                    hd=abs(y_u-yfront)
                                 else
                                    hd=abs(x_u-xfront)
                                 endif
                              else
                                 hd=min(abs(x_u-xfront),abs(y_u-yfront))
                              endif
                              zref2=(vd/sqrt(0.5*Rcx(ibuild)))*sqrt(hd)
                              ! MAN 07/25/2008 stretched vertical grid
                              do kk=kend(ibuild),k_ref
                                 k_shell=kk
                                 if(zref2+Ht(ibuild) .lt. zm(kk+1))exit
                              enddo
                              if(zr .le. zref2 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                                 uo(i,j,k)=uo(i,j,k_shell)*log(zr/zo)/log((zm(k_shell)-Ht(ibuild))/zo)
                                 if(abs(uo(i,j,k)) .gt. max_velmag)then
                                    print*,'Parameterized U exceeds max in rooftop',&
                                       uo(i,j,k),max_velmag,i,j,k
                                 endif
                                 if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                    if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                       icellflag(i,j,k)=3
                                    endif
                                 endif
                              endif
                           endif
                           if(vflag .eq. 1)then
                              if(perpendicular_flag .gt. 0)then
                                 if(ns_flag .eq. 1)then
                                    hd=abs(y_v-yfront)
                                 else
                                    hd=abs(x_v-xfront)
                                 endif
                              else
                                 hd=min(abs(x_v-xfront),abs(y_v-yfront))
                              endif
                              zref2=(vd/sqrt(0.5*Rcx(ibuild)))*sqrt(hd)
                              ! MAN 07/25/2008 stretched vertical grid
                              do kk=kend(ibuild),k_ref
                                 k_shell=kk
                                 if(zref2+Ht(ibuild) .lt. zm(kk+1))exit
                              enddo
                              if(zr .le. zref2 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j-1,k) .ne. 0)then
                                 vo(i,j,k)=vo(i,j,k_shell)*log(zr/zo)/log((zm(k_shell)-Ht(ibuild))/zo)
                                 if(abs(vo(i,j,k)) .gt. max_velmag)then
                                    print*,'Parameterized V exceeds max in rooftop',&
                                       vo(i,j,k),max_velmag,i,j,k
                                 endif
                                 if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                    if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                       icellflag(i,j,k)=3
                                    endif
                                 endif
                              endif
                           endif
                        case(2)
                           if(perpendicular_flag == 1)then
                              if(k .le. kendv)then
                                 if(uflag .eq. 1)then
                                    if(ns_flag .eq. 1)then
                                       hd=abs(y_u-yfront)
                                    else
                                       hd=abs(x_u-xfront)
                                    endif
                                    zref2=(vd/sqrt(0.5*Rcx(ibuild)))*sqrt(hd)
                                    ! MAN 07/25/2008 stretched vertical grid
                                    do kk=kend(ibuild),k_ref
                                       k_shell=kk
                                       if(zref2+Ht(ibuild) .lt. zm(kk+1))exit
                                    enddo
                                    if(zr .le. zref2 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                                       uo(i,j,k)=uo(i,j,k_shell)*log(zr/zo)/log((zm(k_shell)-Ht(ibuild))/zo)
                                       if(abs(uo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized U exceeds max in rooftop',&
                                             uo(i,j,k),max_velmag,i,j,k
                                       endif
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                             icellflag(i,j,k)=3
                                          endif
                                       endif
                                    endif
                                    shell_height=vd*sqrt(1-((0.5*Rcx(ibuild)-hd)/(0.5*Rcx(ibuild)))**2.)
                                    if(hd .lt. Rcx(ibuild) .and. zr .le. shell_height &
                                          .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                                       uo(i,j,k)=-uo_roof(i,j,k)*abs((shell_height-zr)/vd)
                                       if(abs(uo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized U exceeds max in rooftop',&
                                             uo(i,j,k),max_velmag,i,j,k
                                       endif
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                             icellflag(i,j,k)=3
                                          endif
                                       endif
                                    endif
                                 endif
                                 if(vflag .eq. 1)then
                                    if(ns_flag .eq. 1)then
                                       hd=abs(y_v-yfront)
                                    else
                                       hd=abs(x_v-xfront)
                                    endif
                                    zref2=(vd/sqrt(0.5*Rcx(ibuild)))*sqrt(hd)
                                    ! MAN 07/25/2008 stretched vertical grid
                                    do kk=kend(ibuild),k_ref
                                       k_shell=kk
                                       if(zref2+Ht(ibuild) .lt. zm(kk+1))exit
                                    enddo
                                    if(zr .le. zref2 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j-1,k) .ne. 0)then
                                       vo(i,j,k)=vo(i,j,k_shell)*log(zr/zo)/log((zm(k_shell)-Ht(ibuild))/zo)
                                       if(abs(vo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized V exceeds max in rooftop',&
                                             vo(i,j,k),max_velmag,i,j,k
                                       endif
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                             icellflag(i,j,k)=3
                                          endif
                                       endif
                                    endif
                                    shell_height=vd*sqrt(1-((0.5*Rcx(ibuild)-hd)/(0.5*Rcx(ibuild)))**2.)
                                    if(hd .lt. Rcx(ibuild) .and. zr .le. shell_height &
                                          .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                                       vo(i,j,k)=-vo_roof(i,j,k)*abs((shell_height-zr)/vd)
                                       if(abs(vo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized V exceeds max in rooftop',&
                                             vo(i,j,k),max_velmag,i,j,k
                                       endif
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                             icellflag(i,j,k)=3
                                          endif
                                       endif
                                    endif
                                 endif
                                 if(wflag .eq. 1)then
                                    if(ns_flag .eq. 1)then
                                       hd=abs(y_w-yfront)
                                    else
                                       hd=abs(x_w-xfront)
                                    endif
                                    shell_height=vd*sqrt(1-((0.5*Rcx(ibuild)-hd)/(0.5*Rcx(ibuild)))**2.)
                                    if(hd .lt. Rcx(ibuild) .and. zr .le. shell_height .and. &
                                          icellflag(i,j,k) .ne. 0)then
                                       icellflag(i,j,k)=3
                                    endif
                                 endif
                              endif
                           else
                              if(uflag .eq. 1)then
!                                 uo(i,j,k)=uo_roof(i,j,k_ref)*log(zr/zo)/log(zref2/zo)
                                 hx=abs(x_u-xfront)
                                 hy=abs(y_u-yfront)
                                 if(hx .le. min(Rcx(ibuild),2*hy*tan(roofangle)) &
                                       .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                                    if(zr .le. min(Rcx(ibuild),hy*tan(roofangle)))then
                                       uo(i,j,k)=vel_mag*cos(xnorm)
                                       if(abs(uo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized U exceeds max in rooftop',&
                                             uo(i,j,k),max_velmag,i,j,k
                                       endif
                                    elseif(zr .le. min(Rcx(ibuild),2*hy*tan(roofangle)))then
                                       uo(i,j,k)=vel_mag*cos(xnorm+pi)
                                       if(abs(uo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized U exceeds max in rooftop',&
                                             uo(i,j,k),max_velmag,i,j,k
                                       endif
                                    endif
                                 endif
                                 if(hy .le. min(Rcx(ibuild),2*hx*tan(roofangle)) &
                                       .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                                    if(zr .le. min(Rcx(ibuild),hx*tan(roofangle)))then
                                       uo(i,j,k)=vel_mag*cos(ynorm)
                                       if(abs(uo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized U exceeds max in rooftop',&
                                             uo(i,j,k),max_velmag,i,j,k
                                       endif
                                    elseif(zr .le. min(Rcx(ibuild),2*hx*tan(roofangle)))then
                                       uo(i,j,k)=vel_mag*cos(ynorm+pi)
                                       if(abs(uo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized U exceeds max in rooftop',&
                                             uo(i,j,k),max_velmag,i,j,k
                                       endif
                                    endif
                                 endif
                              endif
                              if(vflag .eq. 1)then
!                                 vo(i,j,k)=vo_roof(i,j,k_ref)*log(zr/zo)/log(zref2/zo)
                                 hx=abs(x_u-xfront)
                                 hy=abs(y_u-yfront)
                                 if(hx .le. min(Rcx(ibuild),2*hy*tan(roofangle)) &
                                       .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j-1,k) .ne. 0)then
                                    if(zr .le. min(Rcx(ibuild),hy*tan(roofangle)))then
                                       vo(i,j,k)=vel_mag*sin(xnorm)
                                       if(abs(vo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized V exceeds max in rooftop',&
                                             vo(i,j,k),max_velmag,i,j,k
                                       endif
                                    elseif(zr .le. min(Rcx(ibuild),2*hy*tan(roofangle)))then
                                       vo(i,j,k)=vel_mag*sin(xnorm+pi)
                                       if(abs(vo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized V exceeds max in rooftop',&
                                             vo(i,j,k),max_velmag,i,j,k
                                       endif
                                    endif
                                 endif
                                 if(hy .le. min(Rcx(ibuild),2*hx*tan(roofangle)) &
                                       .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j-1,k) .ne. 0)then
                                    if(zr .le. min(Rcx(ibuild),hx*tan(roofangle)))then
                                       vo(i,j,k)=vel_mag*sin(ynorm)
                                       if(abs(vo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized V exceeds max in rooftop',&
                                             vo(i,j,k),max_velmag,i,j,k
                                       endif
                                    elseif(zr .le. min(Rcx(ibuild),2*hx*tan(roofangle)))then
                                       vo(i,j,k)=vel_mag*sin(ynorm+pi)
                                       if(abs(vo(i,j,k)) .gt. max_velmag)then
                                          print*,'Parameterized V exceeds max in rooftop',&
                                             vo(i,j,k),max_velmag,i,j,k
                                       endif
                                    endif
                                 endif
                              endif
                              if(wflag .eq. 1)then
                                 hx=abs(x_w-xfront)
                                 hy=abs(y_w-yfront)
                                 hd=hy*tan(roofangle)
                                 if(hx .le. min(Rcx(ibuild),2*hd) .and. zr .le. min(Rcx(ibuild),2*hd) &
                                          .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j,k-1) .ne. 0)then
                                    wo(i,j,k)=0.1*vel_mag*((hd-hx)/hd)*(1-abs((zr-hd)/hd))
                                    if(abs(wo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized W exceeds max in rooftop',&
                                          wo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                          icellflag(i,j,k)=3
                                       endif
                                    endif
                                 endif
                                 hd=hx*tan(roofangle)
                                 if(hy .le. min(Rcx(ibuild),2*hd) .and. zr .le. min(Rcx(ibuild),2*hd) &
                                          .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j,k-1) .ne. 0)then
                                    wo(i,j,k)=0.1*vel_mag*((hd-hy)/hd)*(1-abs((zr-hd)/hd))
                                    if(abs(wo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized W exceeds max in rooftop',&
                                          wo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k).ne.0 .and. wflag .eq. 1)then
                                          icellflag(i,j,k)=3
                                       endif
                                    endif
                                 endif
                              endif
                           endif
                     endselect
                  enddo   lp001      
               enddo   lp002      
            enddo   lp003
         endif
         return
      end
