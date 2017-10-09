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
      subroutine upwind
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! 
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer perpendicular_flag,in_subdomain,ns_flag
         integer upIstart,upIstop,upJstart,upJstop
         real uo_h,vo_h,upwind_dir,upwind_rel,xco,yco
         real x1,y1,x2,y2,x3,y3,x4,y4
         real xf1,yf1,xf2,yf2,tol,ynorm,lfcoeff
         real zf,x_u,y_u,x_v,y_v,x_w,y_w
         real xs_u,xs_v,xs_w,xv_u,xv_v,xv_w,xrz_u,xrz_v
         real urot,vrot,uhrot,vhrot,vel_mag
         real vortex_height,build_width,retarding_factor
         real length_factor,height_factor,rz_end,retarding_height,eff_height
         integer ktop,kbottom
         
         if(bldtype(ibuild) .eq. 4)then
            eff_height=0.8*(Ht(ibuild)-zfo_actual(ibuild))+zfo_actual(ibuild)
         else
            eff_height=Ht(ibuild)
         endif
         xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))!CENTER of building in QUIC domain coordinates
         yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
         ! find upwind direction and determine the type of flow regime
         uo_h=uo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         vo_h=vo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         upwind_dir=atan2(vo_h,uo_h)
         upwind_rel=upwind_dir-gamma(ibuild)
         uhrot=uo_h*cos(gamma(ibuild))+vo_h*sin(gamma(ibuild))
         vhrot=-uo_h*sin(gamma(ibuild))+vo_h*cos(gamma(ibuild))
         vel_mag=sqrt((uo_h**2.)+(vo_h**2.))
         tol=10*pi/180.
         if(upwindflag .eq. 1)then
            lfcoeff=2
         else
            lfcoeff=1.5
         endif
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
         if(upwind_rel .gt. 0.5*pi-tol .and. upwind_rel .lt. 0.5*pi+tol)then
            xf1=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yf1=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            xf2=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yf2=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            perpendicular_flag=1
            ns_flag=1
            Lf(ibuild)=abs(lfcoeff*Lti(ibuild)*sin(upwind_rel)/(1+0.8*Lti(ibuild)/eff_height))
            build_width=Lti(ibuild)
         elseif(upwind_rel .gt. -tol .and. upwind_rel .lt. tol)then
            xf1=x1*cos(upwind_dir)+y1*sin(upwind_dir)
            yf1=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
            xf2=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yf2=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            perpendicular_flag=1
            ns_flag=0
            Lf(ibuild)=abs(lfcoeff*Wti(ibuild)*cos(upwind_rel)/(1+0.8*Wti(ibuild)/eff_height))
            build_width=Wti(ibuild)
         elseif(upwind_rel .gt. -0.5*pi-tol .and. upwind_rel .lt. -0.5*pi+tol)then
            xf1=x4*cos(upwind_dir)+y4*sin(upwind_dir)
            yf1=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
            xf2=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yf2=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            perpendicular_flag=1
            ns_flag=1
            Lf(ibuild)=abs(lfcoeff*Lti(ibuild)*sin(upwind_rel)/(1+0.8*Lti(ibuild)/eff_height))
            build_width=Lti(ibuild)
         elseif(upwind_rel .gt. pi-tol .or. upwind_rel .lt. -pi+tol)then
            xf1=x3*cos(upwind_dir)+y3*sin(upwind_dir)
            yf1=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
            xf2=x2*cos(upwind_dir)+y2*sin(upwind_dir)
            yf2=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
            perpendicular_flag=1
            ns_flag=0
            Lf(ibuild)=abs(lfcoeff*Wti(ibuild)*cos(upwind_rel)/(1+0.8*Wti(ibuild)/eff_height))
            build_width=Wti(ibuild)
         endif
!         if(ns_flag .eq. 1)then
!            Lfy(ibuild)=Lf(ibuild)
!            Lfx(ibuild)=0.
!         else
!            Lfx(ibuild)=Lf(ibuild)
!            Lfy(ibuild)=0.
!         endif
         ynorm=abs(yf1)
         retarding_factor=0.4
         length_factor=0.4
         height_factor=0.6
         if(in_subdomain .eq. 1 .and. perpendicular_flag .eq. 1)then
            if(upwindflag .eq. 3)then
               vortex_height=min(build_width,eff_height)
               retarding_height=eff_height
            else
               vortex_height=eff_height
               retarding_height=eff_height
            endif
            ! MAN 07/25/2008 stretched vertical grid
            do k=2,kstart(ibuild)
               kbottom=k
               if(zfo(ibuild) .le. zm(k))exit
            enddo
            do k=kstart(ibuild),nz-1
               ktop=k
               if(height_factor*retarding_height+zfo_actual(ibuild) .le. z(k))exit
            enddo
            upIstart=max(istart(ibuild)-nint(1.5*Lf(ibuild)/dx),2)
            upIstop=min(iend(ibuild)+nint(1.5*Lf(ibuild)/dx),nx-1)
            upJstart=max(jstart(ibuild)-nint(1.5*Lf(ibuild)/dy),2)
            upJstop=min(jend(ibuild)+nint(1.5*Lf(ibuild)/dy),ny-1)
lp003:      do k=kbottom,ktop
               zf=zm(k)-zfo(ibuild)
lp002:         do j=upJstart,upJstop
lp001:            do i=upIstart,upIstop
                     x_u=((real(i)-1)*dx-xco)*cos(upwind_dir)+ &
                                  ((real(j)-0.5)*dy-yco)*sin(upwind_dir)
                     y_u=-((real(i)-1)*dx-xco)*sin(upwind_dir)+ &
                                  ((real(j)-0.5)*dy-yco)*cos(upwind_dir)
                     x_v=((real(i)-0.5)*dx-xco)*cos(upwind_dir)+ &
                                  ((real(j)-1)*dy-yco)*sin(upwind_dir)
                     y_v=-((real(i)-0.5)*dx-xco)*sin(upwind_dir)+	&
                                  ((real(j)-1)*dy-yco)*cos(upwind_dir)
                     x_w=((real(i)-0.5)*dx-xco)*cos(upwind_dir)+ &
                                  ((real(j)-0.5)*dy-yco)*sin(upwind_dir)
                     y_w=-((real(i)-0.5)*dx-xco)*sin(upwind_dir)+	&
                                  ((real(j)-0.5)*dy-yco)*cos(upwind_dir)
! u values
                     if(y_u .ge. -ynorm .and. y_u .le. ynorm)then
                        xs_u=((xf2-xf1)/(yf2-yf1))*(y_u-yf1)+xf1
                        xv_u=-Lf(ibuild)*sqrt((1-((y_u/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
                        xrz_u=-Lf(ibuild)*sqrt((1-((y_u/ynorm)**2.))*(1-((zf/(height_factor*retarding_height))**2.)))
                        if(zf .gt. height_factor*vortex_height)then
                           rz_end=0.
                        else
                           rz_end=length_factor*xv_u
                        endif
                        if(upwindflag .eq. 1)then
                           if(x_u-xs_u .ge. xv_u .and. x_u-xs_u .lt. 0.)then
                              uo(i,j,k)=0.
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                        else
                           if(x_u-xs_u .ge. xrz_u .and. x_u-xs_u .lt. rz_end &
                                 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                              if(upwindflag .eq. 3)then
                                 uo(i,j,k)=((x_u-xs_u-xrz_u)*(retarding_factor-1.)/(rz_end-xrz_u)+1.)*uo(i,j,k)
                              else
                                 uo(i,j,k)=retarding_factor*uo(i,j,k)
                              endif
                              if(abs(uo(i,j,k)) .gt. max_velmag)then
                                 print*,'Parameterized U exceeds max in upwind',&
                                    uo(i,j,k),max_velmag,i,j,k
                              endif
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                           if(x_u-xs_u .ge. length_factor*xv_u .and. x_u-xs_u .lt. 0. &
                                 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0)then
                              urot=uo(i,j,k)*cos(gamma(ibuild))
                              vrot=-uo(i,j,k)*sin(gamma(ibuild))
                              if(ns_flag .eq. 1)then
                                 vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
                                      *(-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*Lf(ibuild)))+0))
                              else
                                 urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
                                      *(-height_factor*sin(((pi*abs(x_u-xs_u))/(length_factor*Lf(ibuild)))+0))
                              endif
                              uo(i,j,k)=urot*cos(-gamma(ibuild))+vrot*sin(-gamma(ibuild))
                              if(abs(uo(i,j,k)) .gt. max_velmag)then
                                 print*,'Parameterized U exceeds max in upwind',&
                                    uo(i,j,k),max_velmag,i,j,k
                              endif
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                        endif
                     endif
!v values
                     if(y_v .ge. -ynorm .and. y_v .le. ynorm)then
                        xs_v=((xf2-xf1)/(yf2-yf1))*(y_v-yf1)+xf1
                        xv_v=-Lf(ibuild)*sqrt((1-((y_v/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
                        xrz_v=-Lf(ibuild)*sqrt((1-((y_v/ynorm)**2.))*(1-((zf/(height_factor*retarding_height))**2.)))
                        if(zf .ge. height_factor*vortex_height)then
                           rz_end=0.
                        else
                           rz_end=length_factor*xv_v
                        endif
                        if(upwindflag .eq. 1)then
                           if(x_v-xs_v .ge. xv_v .and. x_v-xs_v .lt. 0.)then
                              vo(i,j,k)=0.
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                        else
                           if(x_v-xs_v .ge. xrz_v .and. x_v-xs_v .lt. rz_end &
                                 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j-1,k) .ne. 0)then
                              if(upwindflag .eq. 3)then
                                 vo(i,j,k)=((x_v-xs_v-xrz_v)*(retarding_factor-1.)/(rz_end-xrz_v)+1.)*vo(i,j,k)
                              else
                                 vo(i,j,k)=retarding_factor*vo(i,j,k)
                              endif
                              if(abs(vo(i,j,k)) .gt. max_velmag)then
                                 print*,'Parameterized V exceeds max in upwind',&
                                    vo(i,j,k),max_velmag,i,j,k
                              endif
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                           if(x_v-xs_v .ge. length_factor*xv_v .and. x_v-xs_v .lt. 0. &
                                 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j-1,k) .ne. 0)then
                              urot=vo(i,j,k)*sin(gamma(ibuild))
                              vrot=vo(i,j,k)*cos(gamma(ibuild))
                              if(ns_flag .eq. 1)then
                                 vrot=-vhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
                                      *(-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*Lf(ibuild)))+0))
                              else
                                 urot=-uhrot*(-height_factor*cos(((pi*zf)/(0.5*vortex_height)))+0.05)   &
                                      *(-height_factor*sin(((pi*abs(x_v-xs_v))/(length_factor*Lf(ibuild)))+0))
                              endif
                              vo(i,j,k)=-urot*sin(-gamma(ibuild))+vrot*cos(-gamma(ibuild))
                              if(abs(vo(i,j,k)) .gt. max_velmag)then
                                 print*,'Parameterized V exceeds max in upwind',&
                                    vo(i,j,k),max_velmag,i,j,k
                              endif
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                        endif
                     endif
!w values
                     if(y_w .ge. -ynorm .and. y_w .le. ynorm)then
                        xs_w=((xf2-xf1)/(yf2-yf1))*(y_w-yf1)+xf1
                        xv_w=-Lf(ibuild)*sqrt((1-((y_w/ynorm)**2.))*(1-((zf/(height_factor*vortex_height))**2.)))
                        if(upwindflag .eq. 1)then
                           if(x_w-xs_w .ge. xv_w .and. x_w-xs_w .lt. 0.)then
                              wo(i,j,k)=0.
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                        else
                           if(x_w-xs_w .ge. xv_w .and. x_w-xs_w .lt. length_factor*xv_w &
                                 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j,k-1) .ne. 0)then
                              wo(i,j,k)=retarding_factor*wo(i,j,k)
                              if(abs(wo(i,j,k)) .gt. max_velmag)then
                                 print*,'Parameterized W exceeds max in upwind',&
                                    wo(i,j,k),max_velmag,i,j,k
                              endif
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                           if(x_w-xs_w .ge. length_factor*xv_w .and. x_w-xs_w .lt. 0. &
                                 .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i,j,k-1) .ne. 0)then
                              wo(i,j,k)=-vel_mag*(0.1*cos(((pi*abs(x_w-xs_w))/(length_factor*Lf(ibuild))))-0.05)
                              if(abs(wo(i,j,k)) .gt. max_velmag)then
                                 print*,'Parameterized W exceeds max in upwind',&
                                    wo(i,j,k),max_velmag,i,j,k
                              endif
                              if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                 if(icellflag(i,j,k) .ne. 0)then
                                    icellflag(i,j,k)=2
                                 endif
                              endif
                           endif
                        endif
                     endif
                  enddo   lp001      
               enddo   lp002      
            enddo   lp003
         else
            Lf(ibuild)=-999.0
         endif
         return
      end
