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
      subroutine courtyard
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!  
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible
         implicit none
         integer circle_flag,in_subdomain,perpendicular_flag,k_top
         real uo_h,vo_h,upwind_dir,xco,yco,tol
         real zb,x_u,y_u,x_v,y_v,ymin,ymax
         real dNu,dNv,farwake_exponent,farwake_factor,farwake_velocity
         real thetamax,thetamin,thetai
         real xnorm,ynorm,radius,dxy,xnorm_bisect,cav_fac,wake_fac
         real x1,y1,x2,y2,x3,y3,x4,y4
         real xw1,yw1,xw2,yw2,xw3,yw3,xw4,yw4
         real beta,upwind_rel,upwind_rel_norm
         real theta_u,theta_v,theta_corner,velmag,usign,xmax
         real roof_ratio,zfo_roof,court_frac,ro_u,ro_v,ri_u,ri_v,rr_u,rr_v,r_u,r_v
         real indoor_vel_frac,Lt_roof,Wt_roof
         real xu_rel,xv_rel,recirc_frac,courtyard_Lr,eff_height
         
         recirc_frac=0.5
         farwake_exponent=1.5
         farwake_factor=3
         dxy=0.5*(dx+dy)
         xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))!CENTER of building in QUIC domain coordinates
         yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
         uo_h=uo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         vo_h=vo(nint(xco/dx),nint(yco/dy),kend(ibuild)+1)
         indoor_vel_frac=0.75
! find upwind direction and determine the type of flow regime
         upwind_dir=atan2(vo_h,uo_h)
         select case(bldtype(ibuild))
            case(4) !rectangular stadium
               upwind_rel=upwind_dir-gamma(ibuild)
               if(upwind_rel.gt.pi)upwind_rel=upwind_rel-2*pi
               if(upwind_rel.le.-pi)upwind_rel=upwind_rel+2*pi
               upwind_rel_norm=upwind_rel+0.5*pi
               if(upwind_rel_norm .gt. pi)upwind_rel_norm=upwind_rel_norm-2*pi
               beta=abs(atan2(Lti(ibuild),Wti(ibuild)))
               if(wakeflag .eq. 2)then
                  cav_fac=1.1
                  wake_fac=0.1
               else
                  cav_fac=1.
                  wake_fac=0.
               endif
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
                  xw2=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  yw2=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
                  xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir)
                  yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
                  xw4=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  yw4=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
                  perpendicular_flag=0
                  usign=1
               elseif(upwind_rel .ge. 0.5*pi-tol .and. upwind_rel .le. 0.5*pi+tol)then
                  xw1=x1*cos(upwind_dir)+y1*sin(upwind_dir)
                  yw1=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
                  xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
                  xmax=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  perpendicular_flag=1
               elseif(upwind_rel .gt. tol .and. upwind_rel .lt. 0.5*pi-tol)then
                  xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
                  xw2=x1*cos(upwind_dir)+y1*sin(upwind_dir)
                  yw2=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
                  xw3=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  yw3=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
                  xw4=x3*cos(upwind_dir)+y3*sin(upwind_dir)
                  yw4=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
                  perpendicular_flag=0
                  usign=-1
               elseif(abs(upwind_rel) .le. tol)then
                  xw1=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  yw1=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
                  xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir)
                  yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
                  xmax=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  perpendicular_flag=1
               elseif(upwind_rel .lt. -tol .and. upwind_rel .gt. -0.5*pi+tol)then
                  xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir)
                  yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
                  xw2=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  yw2=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
                  xw3=x1*cos(upwind_dir)+y1*sin(upwind_dir)
                  yw3=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
                  xw4=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  yw4=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
                  perpendicular_flag=0
                  usign=1
               elseif(upwind_rel .lt. -0.5*pi+tol .and. upwind_rel .gt. -0.5*pi-tol)then
                  xw1=x3*cos(upwind_dir)+y3*sin(upwind_dir)
                  yw1=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
                  xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
                  xmax=x1*cos(upwind_dir)+y1*sin(upwind_dir)
                  perpendicular_flag=1
               elseif(upwind_rel .lt. -0.5*pi-tol .and. upwind_rel .gt. -pi+tol)then
                  xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
                  xw2=x3*cos(upwind_dir)+y3*sin(upwind_dir)
                  yw2=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
                  xw3=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  yw3=-x4*sin(upwind_dir)+y4*cos(upwind_dir)
                  xw4=x1*cos(upwind_dir)+y1*sin(upwind_dir)
                  yw4=-x1*sin(upwind_dir)+y1*cos(upwind_dir)
                  perpendicular_flag=0
                  usign=-1
               else
                  xw1=x2*cos(upwind_dir)+y2*sin(upwind_dir)
                  yw1=-x2*sin(upwind_dir)+y2*cos(upwind_dir)
                  xw3=x3*cos(upwind_dir)+y3*sin(upwind_dir)
                  yw3=-x3*sin(upwind_dir)+y3*cos(upwind_dir)
                  xmax=x4*cos(upwind_dir)+y4*sin(upwind_dir)
                  perpendicular_flag=1
               endif
               if(perpendicular_flag .eq. 0)then
                  theta_corner=atan2(yw2,xw2)
                  velmag=sqrt((uo_h**2)+(vo_h**2))
               endif
               in_subdomain=0
               if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
                     xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
                     yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
                     yfo(ibuild)-dy.le.y_subdomain_end) in_subdomain=1
               if(in_subdomain .eq. 1)then
                  courtyard_Lr=7.6*Ht(ibuild)
                  if(atten(ibuild) .gt. 0.)then
                     do k=kstart(ibuild),kend(ibuild)  !erp 7/23/03
                        zb=zm(k)-zfo(ibuild)
                        do j=jstart(ibuild),jend(ibuild)
                           do i=istart(ibuild),iend(ibuild)
                              x_u=((real(i)-1)*dx-xco)*cos(upwind_dir)+		&
                                           ((real(j)-0.5)*dy-yco)*sin(upwind_dir)
                              y_u=-((real(i)-1)*dx-xco)*sin(upwind_dir)+		&
                                           ((real(j)-0.5)*dy-yco)*cos(upwind_dir)
                              x_v=((real(i)-0.5)*dx-xco)*cos(upwind_dir)+	&
                                           ((real(j)-1)*dy-yco)*sin(upwind_dir)
                              y_v=-((real(i)-0.5)*dx-xco)*sin(upwind_dir)+	&
                                           ((real(j)-1)*dy-yco)*cos(upwind_dir)
                              theta_u=atan2(y_u,x_u)
                              theta_v=atan2(y_v,x_v)
! u values           
                              if(y_u.ge.0.)then
                                 ynorm=yw1
                              else
                                 ynorm=yw3
                              endif         
                              if(perpendicular_flag .gt. 0)then
                                 xnorm=xw1
                              else
                                 if(y_u.ge.yw2)then
                                    xnorm=((xw2-xw1)/(yw2-yw1))*(y_u-yw1)+xw1
                                 else
                                    xnorm=((xw3-xw2)/(yw3-yw2))*(y_u-yw2)+xw2
                                 endif
                                 if(y_u.ge.yw4)then
                                    xmax=((xw4-xw1)/(yw4-yw1))*(y_u-yw1)+xw1
                                 else
                                    xmax=((xw3-xw4)/(yw3-yw4))*(y_u-yw4)+xw4
                                 endif
                              endif
! Far wake           
                              if(y_u .ge. yw3 .and. y_u .le. yw1 .and. k .le. kend(ibuild) &
                                    .and. icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i-1,j,k) .ne. 0
                                 dNu=sqrt(exp(-10.*(y_u/ynorm)**2.)*   &
                                             (1.-((zb)/Ht(ibuild))**2.)*courtyard_Lr**2)
                                 if(x_u .gt. xnorm+dNu .and. x_u-xnorm .le. &
                                       farwake_factor*dNu .and. x_u .lt. xmax)then
                                    farwake_velocity=uo(i,j,k)*&
                                       (1.-(dNu/(x_u-xnorm+wake_fac*dNu))**(farwake_exponent))
                                    if(abs(farwake_velocity).lt.abs(uo(i,j,k)))then
                                       uo(i,j,k)=farwake_velocity
                                       wo(i,j,k)=0.
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k) .ne. 0)then
                                             icellflag(i,j,k)=5
                                          endif
                                       endif
                                    endif
                                 endif
                              endif
! Cavity                      
                              if(y_u .ge. yw3 .and. y_u .le. yw1 .and. k .le. kend(ibuild) &
                                    .and. x_u .gt. xnorm .and. x_u .le. dNu+xnorm &
                                    .and. x_u .lt. xmax &
                                    .and. icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i-1,j,k) .ne. 0
                                 uo(i,j,k)=-uo_h*min(sqrt(1.-abs(y_u/ynorm)),1.)*(1.-(x_u-xnorm)/(cav_fac*dNu))**2.
                                 if(abs(uo(i,j,k)) .gt. max_velmag)then
                                    print*,'Parameterized U exceeds max in courtyard',&
                                       uo(i,j,k),max_velmag,i,j,k
                                 endif
                                 wo(i,j,k)=0.
                                 if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                    if(icellflag(i,j,k) .ne. 0)then
                                       icellflag(i,j,k)=4
                                    endif
                                 endif
                              endif
! v values           
                              if(y_v.ge.0.)then
                                 ynorm=yw1
                              else
                                 ynorm=yw3
                              endif
                              if(perpendicular_flag .gt. 0)then
                                 xnorm=xw1
                              else
                                 if(y_v.ge.yw2)then
                                    xnorm=((xw2-xw1)/(yw2-yw1))*(y_v-yw1)+xw1
                                 else
                                    xnorm=((xw3-xw2)/(yw3-yw2))*(y_v-yw2)+xw2
                                 endif
                                 if(y_v.ge.yw4)then
                                    xmax=((xw4-xw1)/(yw4-yw1))*(y_v-yw1)+xw1
                                 else
                                    xmax=((xw3-xw4)/(yw3-yw4))*(y_v-yw4)+xw4
                                 endif
                              endif
! Far wake           
                              if(y_v .ge. yw3 .and. y_v .le. yw1 .and. k .le. kend(ibuild) &
                                    .and. icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i,j-1,k) .ne. 0
                                 dNv=sqrt(exp(-10.*(y_v/ynorm)**2.)*   &
                                             (1.-((zb)/Ht(ibuild))**2.)*courtyard_Lr**2)
                                 if(x_v .gt. xnorm+dNv .and. x_v-xnorm .le. &
                                       farwake_factor*dNv .and. x_v .lt. xmax)then
                                    farwake_velocity=vo(i,j,k)*&
                                       (1.-(dNv/(x_v-xnorm+wake_fac*dNv))**(farwake_exponent))
                                    if(abs(farwake_velocity).lt.abs(vo(i,j,k)))then
                                       vo(i,j,k)=farwake_velocity
                                       wo(i,j,k)=0.
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k) .ne. 0)then
                                             icellflag(i,j,k)=5
                                          endif
                                       endif
                                    endif
                                 endif
                              endif
! Cavity             
                              if(y_v .ge. yw3 .and. y_v .le. yw1 .and. k .le. kend(ibuild) &
                                    .and. x_v .gt. xnorm .and. x_v .le. dNv+xnorm &
                                    .and. x_v .lt. xmax &
                                    .and. icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i,j-1,k) .ne. 0
                                 vo(i,j,k)=-vo_h*min(sqrt(1.-abs(y_v/ynorm)),1.)*(1.-(x_v-xnorm)/(cav_fac*dNv))**2.
                                 if(abs(vo(i,j,k)) .gt. max_velmag)then
                                    print*,'Parameterized V exceeds max in courtyard',&
                                       vo(i,j,k),max_velmag,i,j,k
                                 endif
                                 wo(i,j,k)=0.
                                 if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                    if(icellflag(i,j,k) .ne. 0)then
                                       icellflag(i,j,k)=4
                                    endif
                                 endif
                              endif
                           enddo         
                        enddo         
                     enddo
                  else
                     roof_ratio=0.8
                     zfo_roof=(Ht(ibuild)-zfo_actual(ibuild))*roof_ratio
                     court_frac=abs(atten(ibuild))
                     do j=jstart(ibuild),jend(ibuild)
                        do i=istart(ibuild),iend(ibuild)
                           x_u=((real(i)-1)*dx-xco)*cos(gamma(ibuild))+		&
                                        ((real(j)-0.5)*dy-yco)*sin(gamma(ibuild))
                           y_u=-((real(i)-1)*dx-xco)*sin(gamma(ibuild))+		&
                                        ((real(j)-0.5)*dy-yco)*cos(gamma(ibuild))
                           x_v=((real(i)-0.5)*dx-xco)*cos(gamma(ibuild))+	&
                                        ((real(j)-1)*dy-yco)*sin(gamma(ibuild))
                           y_v=-((real(i)-0.5)*dx-xco)*sin(gamma(ibuild))+	&
                                        ((real(j)-1)*dy-yco)*cos(gamma(ibuild))
                           xu_rel=((real(i)-1)*dx-xco)*cos(upwind_rel)+		&
                                        ((real(j)-0.5)*dy-yco)*sin(upwind_dir)
                           xv_rel=((real(i)-0.5)*dx-xco)*cos(upwind_rel)+	&
                                        ((real(j)-1)*dy-yco)*sin(upwind_dir)
                           do k=kstart(ibuild),kend(ibuild)
                              zb=zm(k)-zfo(ibuild)
                              if(zb .le. zfo_roof)then
                                 if(abs(x_u) .lt. Lt(ibuild) &
                                       .and. abs(y_u) .lt. Wt(ibuild) &
                                       .and. (abs(x_u) .gt. recirc_frac*(Lt(ibuild)-court_frac) &
                                       .or. abs(y_u) .gt. recirc_frac*(Wt(ibuild)-court_frac)) &
                                       .and. icellflag(i,j,k) .ne. 0  &
                                       .and. xu_rel .lt. 0)then !.and. icellflag(i-1,j,k) .ne. 0
                                    uo(i,j,k)=-indoor_vel_frac*uo_h*(1-zb/zfo_roof)
                                    if(abs(uo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized U exceeds max in partial roof courtyard',&
                                          uo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                                 if(abs(x_v) .lt. Lt(ibuild) &
                                       .and. abs(y_v) .lt. Wt(ibuild) &
                                       .and. (abs(x_v) .gt. recirc_frac*(Lt(ibuild)-court_frac) &
                                       .or. abs(y_v) .gt. recirc_frac*(Wt(ibuild)-court_frac)) &
                                       .and. icellflag(i,j,k) .ne. 0 &
                                       .and. xv_rel .lt. 0)then !.and. icellflag(i,j-1,k) .ne. 0
                                    vo(i,j,k)=-indoor_vel_frac*vo_h*(1-zb/zfo_roof)
                                    if(abs(vo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized V exceeds max in partial roof courtyard',&
                                          vo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                              else
                                 Lt_roof=Lt(ibuild)-court_frac*((zb-zfo_roof)/(0.25*zfo_roof))**2.
                                 Wt_roof=Wt(ibuild)-court_frac*((zb-zfo_roof)/(0.25*zfo_roof))**2.
                                 if(abs(x_u) .lt. Lt_roof &
                                       .and. abs(y_u) .lt. Wt_roof &
                                       .and. (abs(x_u) .gt. recirc_frac*(Lt(ibuild)-court_frac) &
                                       .or. abs(y_u) .gt. recirc_frac*(Wt(ibuild)-court_frac)) &
                                       .and. icellflag(i,j,k) .ne. 0 .and. icellflag(i-1,j,k) .ne. 0 &
                                       .and. xu_rel .lt. 0)then
                                    uo(i,j,k)=indoor_vel_frac*uo_h*(4*abs(zb-zfo_roof)/zfo_roof)
                                    if(abs(uo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized U exceeds max in partial roof courtyard',&
                                          uo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                                 if(abs(x_v) .lt. Lt_roof &
                                       .and. abs(y_v) .lt. Wt_roof &
                                       .and. (abs(x_v) .gt. recirc_frac*(Lt(ibuild)-court_frac) &
                                       .or. abs(y_v) .gt. recirc_frac*(Wt(ibuild)-court_frac)) &
                                       .and. icellflag(i,j,k) .ne. 0  &
                                       .and. xv_rel .lt. 0)then !.and. icellflag(i,j-1,k) .ne. 0
                                    vo(i,j,k)=indoor_vel_frac*vo_h*(4*abs(zb-zfo_roof)/zfo_roof)
                                    if(abs(vo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized V exceeds max in partial roof courtyard',&
                                          vo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                              endif
                           enddo
                        enddo
                     enddo
                  endif   
               endif
            case(5) ! Elliptical stadium
               tol=0.01*min(dx,dy)
               if(abs(aa(ibuild)-bb(ibuild)) .lt. tol)then
                  circle_flag=1
               else
                  circle_flag=0
               endif
               if(wakeflag .eq. 2)then
                  cav_fac=1.1
                  wake_fac=0.1
               else
                  cav_fac=1.
                  wake_fac=0.
               endif
               if(circle_flag .eq. 1)then
                  thetamin=-0.5*pi
                  thetamax=0.5*pi
                  ymax=Lt(ibuild)
                  ymin=-Lt(ibuild)
               else
                  y_u=0.
                  y_v=0.
                  do i=1,180
                     thetai=real(180-i)*pi/180.
                     y_u=radius(aa(ibuild),bb(ibuild),thetai,&
                         gamma(ibuild)-upwind_dir)*sin(thetai)
                     if(y_u .lt. y_v)then
                        exit
                     endif
                     y_v=y_u
                  enddo
                  thetamax=thetai+pi/180.
                  thetamin=thetamax-pi
                  ymax=y_v
                  ymin=-y_v
               endif
               in_subdomain=0
               if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
                     xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
                     yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
                     yfo(ibuild)-dy.le.y_subdomain_end) in_subdomain=1
               ynorm=max(ymax,-ymin)
               if(in_subdomain .eq. 1)then
                  courtyard_Lr=7.6*Ht(ibuild)
                  if(atten(ibuild) .gt. 0.)then
                     eff_height=0.9*Ht(ibuild)
                     do k=kstart(ibuild),nz-1
                        k_top=k
                        if(eff_height .lt. zm(k+1))exit
                     enddo
                     do k=kstart(ibuild),k_top
                        zb=zm(k)-zfo(ibuild)
                        do j=jstart(ibuild),jend(ibuild)
                           do i=istart(ibuild),iend(ibuild)
                              x_u=((real(i)-1)*dx-xco)*cos(upwind_dir)+		&
                                           ((real(j)-0.5)*dy-yco)*sin(upwind_dir)
                              y_u=-((real(i)-1)*dx-xco)*sin(upwind_dir)+		&
                                           ((real(j)-0.5)*dy-yco)*cos(upwind_dir)
                              x_v=((real(i)-0.5)*dx-xco)*cos(upwind_dir)+	&
                                           ((real(j)-1)*dy-yco)*sin(upwind_dir)
                              y_v=-((real(i)-0.5)*dx-xco)*sin(upwind_dir)+	&
                                           ((real(j)-1)*dy-yco)*cos(upwind_dir)
! u values           
                              if(y_u .gt. ymin .and. y_u .lt. ymax &
                                    .and. icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i-1,j,k) .ne. 0
                                 if(circle_flag .eq. 1)then 
                                    xmax=sqrt((Lt(ibuild)**2.)-(y_u**2.))
                                    xnorm=-xmax
                                 else
                                    xnorm=-xnorm_bisect(aa(ibuild),bb(ibuild),&
                                          gamma(ibuild)-upwind_dir,-y_u,thetamin,thetamax,dxy)
                                    xmax=xnorm_bisect(aa(ibuild),bb(ibuild),&
                                          gamma(ibuild)-upwind_dir,y_u,thetamin,thetamax,dxy)
                                 endif
                                 dNu=sqrt(exp(-10.*(y_u/ynorm)**2.)*   &
                                             (1.-((zb)/eff_height)**2.)*courtyard_Lr**2)
! far wake                    
                                 if(x_u .gt. xnorm+dNu .and. x_u-xnorm .le. &
                                       farwake_factor*dNu .and. x_u .lt. xmax-dxy)then
                                    farwake_velocity=uo(i,j,k)*&
                                       (1.-(dNu/(x_u-xnorm+wake_fac*dNu))**(farwake_exponent))
                                    if(abs(farwake_velocity).lt.abs(uo(i,j,k)))then
                                       uo(i,j,k)=farwake_velocity
                                       wo(i,j,k)=0.
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k).ne.0)then
                                             icellflag(i,j,k)=5
                                          endif
                                       endif
                                    endif
                                 endif
! cavity                      
                                 if(x_u .gt. xnorm +dxy .and. x_u .le. dNu+xnorm .and. x_u .lt. xmax-dxy)then
                                    uo(i,j,k)=-uo_h*min(sqrt(1.-abs(y_u/ynorm)),1.)*(1.-(x_u-xnorm)/(cav_fac*dNu))**2.
                                    wo(i,j,k)=0.
                                    if(abs(uo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized U exceeds max in courtyard',&
                                          uo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                              endif
! v values           
                              if(y_v .gt. ymin .and. y_v .lt. ymax &
                                    .and. icellflag(i,j,k) .ne. 0 )then !.and. icellflag(i,j-1,k) .ne. 0
                                 if(circle_flag .eq. 1)then
                                    xmax=sqrt((Lt(ibuild)**2.)-(y_v**2.))
                                    xnorm=-xmax
                                 else
                                    xnorm=-xnorm_bisect(aa(ibuild),bb(ibuild),&
                                          gamma(ibuild)-upwind_dir,-y_v,thetamin,thetamax,dxy)
                                    xmax=xnorm_bisect(aa(ibuild),bb(ibuild),&
                                          gamma(ibuild)-upwind_dir,y_v,thetamin,thetamax,dxy)
                                 endif
                                 dNv=sqrt(exp(-10.*(y_v/ynorm)**2.)*   &
                                             (1.-((zb)/eff_height)**2.)*courtyard_Lr**2)
! far wake                    
                                 if(x_v .gt. xnorm+dNv .and. x_v-xnorm .le. &
                                       farwake_factor*dNv .and. x_v .lt. xmax-dxy)then
                                    farwake_velocity=vo(i,j,k)*&
                                       (1.-(dNv/(x_v-xnorm+wake_fac*dNv))**(farwake_exponent))
                                    if(abs(farwake_velocity).lt.abs(vo(i,j,k)))then
                                       vo(i,j,k)=farwake_velocity
                                       wo(i,j,k)=0.
                                       if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                          if(icellflag(i,j,k).ne.0)then
                                             icellflag(i,j,k)=5
                                          endif
                                       endif
                                    endif
                                 endif
! cavity                      
                                 if(x_v .gt. xnorm +dxy .and. x_v .le. dNv+xnorm .and. x_v .lt. xmax-dxy)then
                                    vo(i,j,k)=-vo_h*min(sqrt(1.-abs(y_v/ynorm)),1.)*(1.-(x_v-xnorm)/(cav_fac*dNv))**2.
                                    if(abs(vo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized V exceeds max in courtyard',&
                                          vo(i,j,k),max_velmag,i,j,k
                                    endif
                                    wo(i,j,k)=0.
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                              endif
                           enddo        
                        enddo         
                     enddo
                  else
                     roof_ratio=0.8
                     zfo_roof=(Ht(ibuild)-zfo_actual(ibuild))*roof_ratio
                     court_frac=abs(atten(ibuild))
                     do j=jstart(ibuild),jend(ibuild)
                        do i=istart(ibuild),iend(ibuild)
                           x_u=((real(i)-1)*dx-xco)*cos(upwind_dir)+		&
                                        ((real(j)-0.5)*dy-yco)*sin(upwind_dir)
                           y_u=-((real(i)-1)*dx-xco)*sin(upwind_dir)+		&
                                        ((real(j)-0.5)*dy-yco)*cos(upwind_dir)
                           x_v=((real(i)-0.5)*dx-xco)*cos(upwind_dir)+	&
                                        ((real(j)-1)*dy-yco)*sin(upwind_dir)
                           y_v=-((real(i)-0.5)*dx-xco)*sin(upwind_dir)+	&
                                        ((real(j)-1)*dy-yco)*cos(upwind_dir)
                           r_u=sqrt((x_u**2.)+(y_u**2.))
                           r_v=sqrt((x_v**2.)+(y_v**2.))
                           theta_u=atan2(y_u,x_u)
                           theta_v=atan2(y_v,x_v)
                           ro_u=radius(aa(ibuild),bb(ibuild),theta_u,gamma(ibuild))
                           ro_v=radius(aa(ibuild),bb(ibuild),theta_v,gamma(ibuild))
                           ri_u=radius(aa(ibuild)-court_frac,bb(ibuild)-court_frac,theta_u,gamma(ibuild))
                           ri_v=radius(aa(ibuild)-court_frac,bb(ibuild)-court_frac,theta_v,gamma(ibuild))
                           do k=kstart(ibuild),kend(ibuild)
                              zb=zm(k)-zfo_actual(ibuild)
                              if(zb .le. zfo_roof)then
                                 if(r_u .lt. ro_u-0.75*dxy &
                                    .and. icellflag(i,j,k) .ne. 0  &
                                    .and. x_u .lt. 0. .and. r_u .gt. recirc_frac*ri_u)then !.and. icellflag(i-1,j,k) .ne. 0 
                                    uo(i,j,k)=-indoor_vel_frac*uo_h*(1-zb/zfo_roof)
                                    if(abs(uo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized U exceeds max in courtyard',&
                                          uo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                                 if(r_v .lt. ro_v-0.75*dxy &
                                    .and. icellflag(i,j,k) .ne. 0  &
                                    .and. x_v .lt. 0. .and. r_v .gt. recirc_frac*ri_v)then ! .and. icellflag(i,j-1,k) .ne. 0
                                    vo(i,j,k)=-indoor_vel_frac*vo_h*(1-zb/zfo_roof)
                                    if(abs(vo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized V exceeds max in courtyard',&
                                          vo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                              else
                                 rr_u=ro_u-(ro_u-ri_u)*((zb-zfo_roof)/(0.25*zfo_roof))**2.
                                 rr_v=ro_v-(ro_v-ri_v)*((zb-zfo_roof)/(0.25*zfo_roof))**2.
                                 if(r_u .lt. rr_u-0.75*dxy &
                                    .and. icellflag(i,j,k) .ne. 0 &
                                    .and. x_u .lt. 0..and. r_u .gt. recirc_frac*ri_u)then ! .and. icellflag(i-1,j,k) .ne. 0 
                                    uo(i,j,k)=indoor_vel_frac*uo_h*(4*abs(zb-zfo_roof)/zfo_roof)
                                    if(abs(uo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized U exceeds max in courtyard',&
                                          uo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                                 if(r_v .lt. rr_v-0.75*dxy &
                                    .and. icellflag(i,j,k) .ne. 0  &
                                    .and. x_v .lt. 0. .and. r_v .gt. recirc_frac*ri_v)then ! .and. icellflag(i,j-1,k) .ne. 0
                                    vo(i,j,k)=indoor_vel_frac*vo_h*(4*abs(zb-zfo_roof)/zfo_roof)
                                    if(abs(vo(i,j,k)) .gt. max_velmag)then
                                       print*,'Parameterized V exceeds max in courtyard',&
                                          vo(i,j,k),max_velmag,i,j,k
                                    endif
                                    if(i .lt. nx .and. j .lt. ny .and. k .lt. nz)then
                                       if(icellflag(i,j,k) .ne. 0)then
                                          icellflag(i,j,k)=4
                                       endif
                                    endif
                                 endif
                              endif
                           enddo
                        enddo
                     enddo
                  endif  
               endif
         endselect
         return
      end