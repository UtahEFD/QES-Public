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
      subroutine wake

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc 
! trailing buildings
! now add wake cavity flow behind building
! depending on which quadrant the wind is coming from, 1 of 4
! algorithms is used.
! There are basically 3 coordinate systems: (1) global with an origin 
! at (x=0,y=0,z=0)(2) a building coordinate system with the origin 
! at (xfo,yfo,zfo) and finally the rotated coordinate system with 
! the origin at point P' which is the virtual back center of the 
! effective building width. Below we calculate where the data points 
! are in the new rotated cooridinate system (xpprime,yprime).
! Important variables defined:
! zb - distance above the ground 
! ss - distance between the Pivot corner P and P'
! delta_x - x component of P' with respect to P
! delta_y - y component of P' with respect to P
! 
! 1/25/05 Remove wake entirely 

         use datamodule ! make data from module "datamodule" visible
         implicit none
         
         real tau,ss,zb,delta_x,delta_y,xpprime,ypprime
         real x_loc_u,y_loc_u,x_loc_v,y_loc_v,x_loc_w,y_loc_w
         real omega_u,lill_u,xprimeu,yprimeu
         real omega_v,lill_v,xprimev,yprimev
         real omega_w,lill_w,xprimew,yprimew,LoverH
         real chiu,chiv,chiw,dlv,dNv,dlu,dNu,dlw,dNw
         
         integer kbottom

! erp add subdomain check 5/3/03 building loop 5
            
            !         if(xfo(ibuild)-1.ge.x_subdomain_start .and. &
            !     &   xfo(ibuild)-1.lt.x_subdomain_end   .and. &
            !     &   yfo(ibuild)-1.ge.y_subdomain_start .and. &
            !     &   yfo(ibuild)-1.le.y_subdomain_end)then
! dx change for subdomain erp 6/8/2006
            Weff(ibuild)=Lti(ibuild)*sin(abs(phiprime(ibuild)&
                  -gamma(ibuild)))+Wti(ibuild)*cos(abs(phiprime(ibuild)-gamma(ibuild)))
            Leff(ibuild)=Wti(ibuild)*sin(abs(phiprime(ibuild)&
                  -gamma(ibuild)))+Lti(ibuild)*cos(abs(phiprime(ibuild)-gamma(ibuild)))
! calculate Lr the length of the downwind reciruculation cavity
! Putting lower and upper limit on L/H based on fackrell formulation
            LoverH=Leff(ibuild)/Ht(ibuild)
            if(LoverH.gt.3.)LoverH=3.
            if(LoverH.lt.0.3)LoverH=0.3
            Lr(ibuild)=1.8*Weff(ibuild)/((LoverH**(0.3))*   &
                        (1+0.24*Weff(ibuild)/Ht(ibuild)))
            do k=2,kstart(ibuild)
               kbottom=k
               if(zfo(ibuild) .le. zm(k))exit
            enddo
            if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
                 xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
                 yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
                 yfo(ibuild)-dy.le.y_subdomain_end)then


!               if(w_flag(ibuild).eq.1)then       !check wake flag
!erp 9/19/2005   tau=Lti(ibuild)*sin(phiprime(ibuild))
                  tau=Lti(ibuild)*sin(abs(phiprime(ibuild)-gamma(ibuild)))
     
!If the mean wind coming from the South-West
                  if(theta(ibuild).ge.180.and.theta(ibuild).le.270)then
                     ss=(Weff(ibuild)/2.)-wprime(ibuild)

!erp test 8/17/05
   
                     if(zfo(ibuild).gt.0)then
                        call building_connect
                     endif
!erp end test

!erp  do k=int(zfo(ibuild)),kend(ibuild)  !erp 7/23/03
! int changed nint in line below 8-14-2006

                     
lp018:               do k=kbottom,kend(ibuild)  !erp 7/23/03   !convert to real world unit, TZ 11/16/04
lp017:                  do j=jstart(ibuild),ny-1   !change so wake ends at next bld
lp016:                     do i=istart(ibuild),nx-1
!erp   zb=real(k)-zfo(ibuild)             !erp 7/23/03 distance above the ground
                              zb=zm(k)-zfo(ibuild)      !erp 7/23/03 distance above the ground !convert to real world unit, TZ 11/16/04

                              if(zb.le.Ht(ibuild))then

! if P' is located South East of P
                                 if(tau.lt.(Weff(ibuild)/2.))then
                                    delta_x=ss*abs(sin(phiprime(ibuild)))  !erp 12/17
                                    delta_y=ss*abs(cos(phiprime(ibuild)))  !erp 12/17
                                    xpprime=Lti(ibuild)+delta_x       !effective building origin
                                    ypprime=Wti(ibuild)-delta_y      !effective building origin
                                 endif
! if P' is located North West of P
                                 if(tau.ge.(Weff(ibuild)/2.))then
                                    delta_x=abs(ss)*abs(cos(pi/2. - phiprime(ibuild))) !erp 12/17
                                    delta_y=abs(ss)*abs(sin(pi/2. - phiprime(ibuild))) !erp 12/17
                                    xpprime=Lti(ibuild)-delta_x   !effective building origin
                                    ypprime=Wti(ibuild)+delta_y      !effective building origin
                                 endif

                                 x_loc_u=real(i-istart(ibuild))*dx-xpprime !convert to real world unit, TZ 11/16/04
                                 y_loc_u=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
                                 x_loc_v=(real(i)+0.5-real(istart(ibuild)))*dx-xpprime !convert to real world unit, TZ 11/16/04
                                 y_loc_v=real(j-jstart(ibuild))*dy-ypprime !convert to real world unit, TZ 11/16/04

!erp 8-15-2006
                                 x_loc_w=(real(i)+0.5-real(istart(ibuild)))*dx-xpprime !convert to real world unit, TZ 11/16/04
                                 y_loc_w=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
!end erp 8-15

!if x_loc <0 omega = pi/2 - omega
!u component
!erp eliminate divide by zero
                                 if(x_loc_u.ne.0)then
                                    omega_u=atan(y_loc_u/x_loc_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.gt.0)omega_u= pi - abs(omega_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.lt.0)omega_u=-(pi - abs(omega_u))
                                 endif
                                 if(x_loc_u.eq.0.and.y_loc_u.ge.0)omega_u=pi/2
                                 if(x_loc_u.eq.0.and.y_loc_u.lt.0)omega_u=-pi/2
                                 lill_u=sqrt((x_loc_u)**2+(y_loc_u)**2)
                                 xprimeu=lill_u*cos(omega_u-phiprime(ibuild))
                                 yprimeu=lill_u*sin(omega_u-phiprime(ibuild))
! v component
                                 if(x_loc_v.ne.0)then
                                    omega_v=atan(y_loc_v/x_loc_v)
!if x_loc <0 omega = pi/2 - omega
                                    if(x_loc_v.lt.0.and.y_loc_v.gt.0)omega_v= pi - abs(omega_v)
                                    if(x_loc_v.lt.0.and.y_loc_v.lt.0)omega_v=-(pi - abs(omega_v))
                                 endif
                                 if(x_loc_v.eq.0.and.y_loc_v.ge.0)omega_v=pi/2
                                 if(x_loc_v.eq.0.and.y_loc_v.lt.0)omega_v=-pi/2
                                 lill_v=sqrt((x_loc_v)**2+(y_loc_v)**2)
!erp  10/9/2003
                                 xprimev=lill_v*cos(omega_v-phiprime(ibuild))
                                 yprimev=lill_v*sin(omega_v-phiprime(ibuild))

! w component
                                 if(x_loc_w.ne.0)then
                                    omega_w=atan(y_loc_w/x_loc_w)
!if x_loc <0 omega = pi/2 - omega
                                    if(x_loc_w.lt.0.and.y_loc_w.gt.0)omega_w= pi - abs(omega_w)
                                    if(x_loc_w.lt.0.and.y_loc_w.lt.0)omega_w=-(pi - abs(omega_w))
                                 endif
                                 if(x_loc_w.eq.0.and.y_loc_w.ge.0)omega_w=pi/2
                                 if(x_loc_w.eq.0.and.y_loc_w.lt.0)omega_w=-pi/2
                                 lill_w=sqrt((x_loc_w)**2+(y_loc_w)**2)
                                 xprimew=lill_w*cos(omega_w-phiprime(ibuild))
                                 yprimew=lill_w*sin(omega_w-phiprime(ibuild))
                              
!end

                                 chiu=0.
                                 chiv=0.
                                 chiw=0. !erp 8-15-06

                                 if(phiprime(ibuild).ne.0.)then
! for u-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.ss)then
                                          chiu=abs((yprimeu-(ss))/tan(phiprime(ibuild)))
                                       endif
                                       if(yprimeu.lt.ss)then
                                          chiu=abs((ss)-yprimeu)*tan(phiprime(ibuild))
                                       endif
                                    endif
      
                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.ss)then
                                          chiu=abs(yprimeu-(ss))/tan(phiprime(ibuild))
                                       endif
                                       if(yprimeu.lt.ss)then
                                          chiu=abs(yprimeu-(ss))*tan(phiprime(ibuild))
                                       endif
                                    endif
!for v-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.ss)then
                                          chiv=abs((yprimev-(ss))/tan(phiprime(ibuild)))
                                       endif
                                       if(yprimev.lt.ss)then
                                          chiv=abs((ss)-yprimev)*tan(phiprime(ibuild))
                                       endif
                                    endif
      
                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.ss)then
                                          chiv=abs(yprimev-(ss))/tan(phiprime(ibuild))
                                       endif
                                       if(yprimev.lt.ss)then  !bug yprimeu changed yprimev 8-15-06
                                          chiv=abs(yprimev-(ss))*tan(phiprime(ibuild))
                                       endif
                                    endif

!for w-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.ss)then
                                          chiw=abs((yprimew-(ss))/tan(phiprime(ibuild)))
                                       endif
                                       if(yprimew.lt.ss)then
                                          chiw=abs((ss)-yprimew)*tan(phiprime(ibuild))
                                       endif
                                    endif
      
                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.ss)then
                                          chiw=abs(yprimew-(ss))/tan(phiprime(ibuild))
                                       endif
                                       if(yprimew.lt.ss)then
                                          chiw=abs(yprimew-(ss))*tan(phiprime(ibuild))
                                       endif
                                    endif
!end new w definition erp 8-15-06

                                 endif !phi prime if to avoid divide by zero.

!erp introduce new variables for new modification 10/9/2002
! dlu,dlv,yprimeu, yprimev, dNu and dNv
                                 if(xprimev+chiv.gt.0.and.chiv.le.Leff(ibuild))then
! v-component
                                    dlv = xprimev + chiv
                                    if(abs(yprimev).le.(Weff(ibuild)/2.).and.zb.ge.0)then
                                       dNv=sqrt((1.-(yprimev/(Weff(ibuild)/2.))**2)*   &
                                                    ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiv
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlv.gt.0.and.dlv.le.3.*dNv .and. icellflag(i,j,k).ne.4)then
! erp/mdw bug fix 2/19/2003
!                 vo(i,j,k)=vo(i,j,k)*(1.-dNv/dlv)**(1.5)
                                          vo(i,j,k)=vo(i,j,k)*(1.-(dNv/dlv)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
! calculate cavity flow velocities
                                       if(dlv.le.dNv.and.dNv.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          vo(i,j,k)=-vo(istart(ibuild)-1,jstart(ibuild),kend(ibuild)   &
                                                       +1)*(1.-dlv/dNv)**2
! erp9/12/2003          vo(i,j,k)=-vo(istart(ibuild),jstart(ibuild),kend(ibuild)
!     &          +1)*(1.-dlv/dNv)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif
!u-component          
                                 if(xprimeu+chiu.gt.0.and.chiu.le.Leff(ibuild))then
                                    dlu = xprimeu + chiu

                                    if(abs(yprimeu).le.(Weff(ibuild)/2.).and.zb.ge.0)then  !erp
                                       dNu=sqrt((1.-(yprimeu/(Weff(ibuild)/2.))**2)*   &
                                                        ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiu  !erp

! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlu.gt.0.and.dlu.le.3.*dNu .and. icellflag(i,j,k).ne.4)then
!                                       if(dlu.gt.0 .and. dlu.le.3.*dNu)then
                                          uo(i,j,k)=uo(i,j,k)*(1.-(dNu/dlu)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
! erp 8-15                                    if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
                                          
!end MAN 7/8/2005
!                             endif
                                       endif
! calculate cavity flow velocities
                                       if(dlu.le.dNu.and.dNu.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          uo(i,j,k)=-uo(istart(ibuild)-1,jstart(ibuild),   &
                                                         kend(ibuild)+1)*(1.-dlu/dNu)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif !xprime + chi if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! new cellflag definition section test erp 8-15-06

                                 if(xprimew+chiw.gt.0.and.chiw.le.Leff(ibuild))then
! w-component
                                    dlw = xprimew + chiw
                                    if(abs(yprimew).le.(Weff(ibuild)/2.).and.zb.ge.0)then
                                       dNw=sqrt((1.-(yprimew/(Weff(ibuild)/2.))**2)*   &
                                           ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiw
! calculate wake flow velocities
                                       if(dlw.gt.0.and.dlw.le.3.*dNw .and. icellflag(i,j,k).ne.4)then
                                          if(icellflag(i,j,k).ne.0)then
                                             icellflag(i,j,k)=5  !far wake
                                          endif
                                       endif
! calculate cavity flow velocities
                                       if(dlw.le.dNw.and.dNw.gt.0)then
                                          if(icellflag(i,j,k).ne.0)then
                                             icellflag(i,j,k)=4   !near wake cavity
                                          endif
                                       endif
                                    endif !yprime if
                                 endif

! end new celfflage definition change
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



 14                              format('xprime',f8.4,'yprime ',f8.4,'chi ',f8.4,'dN ',f8.4,'dl ',f8.   &
                                    4)

                              endif       !end zb if
                           enddo   lp016      
                        enddo   lp017      
                     enddo   lp018      
                  endif               !end angle 1
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc    
! north easterly flow
                  if(theta(ibuild).ge.0.and.theta(ibuild).lt.90)then
!erp removed 7/23/03      zbo=2.
                     ss=(Weff(ibuild)/2.)-wprime(ibuild)
                     if(zfo(ibuild).gt.0)then
                        call building_connect
                     endif
!erp end test
! int changed nint in line below 8-14-2006
lp021:               do k=kbottom,kend(ibuild)  !convert to real world unit, TZ 11/16/04

lp020:                  do j=jend(ibuild),1,-1  !change so wake ends at next bld
lp019:                     do i=iend(ibuild),1,-1
!erp mod 7/23/03       zb=real(k)-zbo             !distance above the ground
!erp       zb=real(k)-zfo(ibuild)            !distance above the ground
                              zb=zm(k)-zfo(ibuild)            !distance above the ground !convert to real world unit, TZ 11/16/04
                              if(zb.le.Ht(ibuild))then

! if P' is located South East of P
                                 if(tau.lt.(Weff(ibuild)/2.))then
                                    delta_x=ss*(sin(phiprime(ibuild))) !erp 12/17
                                    delta_y=ss*(cos(phiprime(ibuild))) !erp 12/17
                                    xpprime= -delta_x     !effective building origin
                                    ypprime= +delta_y       !effective building origin
                                 endif
! if P' is located North West of P
                                 if(tau.ge.(Weff(ibuild)/2.))then
                                    delta_x=abs(ss)*(cos(pi/2.-phiprime(ibuild))) !erp 12/17
                                    delta_y=abs(ss)*(sin(pi/2.-phiprime(ibuild))) !erp 12/17
                                    xpprime= +delta_x                 !effective building origin
                                    ypprime= -delta_y       !effective building origin
                                 endif

                                 x_loc_u=-(real(i-istart(ibuild))*dx-xpprime) !convert to real world unit, TZ 11/16/04
                                 y_loc_u=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
                                 x_loc_v=-((real(i)+0.5-real(istart(ibuild)))*dx-xpprime) !convert to real world unit, TZ 11/16/04
                                 y_loc_v=real(j-jstart(ibuild))*dy-ypprime !convert to real world unit, TZ 11/16/04

!erp 8-15-2006
                                 x_loc_w=-((real(i)+0.5-real(istart(ibuild)))*dx-xpprime) !convert to real world unit, TZ 11/16/04
                                 y_loc_w=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
!end erp 8-15

!if x_loc <0 omega = pi/2 - omega
!u component
!erp eliminate divide by zero
                                 if(x_loc_u.ne.0)then
                                    omega_u=atan(y_loc_u/x_loc_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.gt.0)omega_u= pi - abs(omega_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.lt.0)omega_u=-(pi - abs(omega_u))

                                 endif
                                 if(x_loc_u.eq.0.and.y_loc_u.ge.0)omega_u=pi/2
                                 if(x_loc_u.eq.0.and.y_loc_u.lt.0)omega_u=-pi/2
                                 lill_u=sqrt((x_loc_u)**2+(y_loc_u)**2)
                                 xprimeu=lill_u*cos(omega_u+phiprime(ibuild))
                                 yprimeu=lill_u*sin(omega_u+phiprime(ibuild))

! v component
                                 if(x_loc_v.ne.0)then
                                    omega_v=atan(y_loc_v/x_loc_v)
!if x_loc <0 omega = pi/2 - omega
                                    if(x_loc_v.lt.0.and.y_loc_v.gt.0)omega_v= pi - abs(omega_v)
                                    if(x_loc_v.lt.0.and.y_loc_v.lt.0)omega_v=-(pi - abs(omega_v))
                                 endif
                                 if(x_loc_v.eq.0.and.y_loc_v.ge.0)omega_v=pi/2
                                 if(x_loc_v.eq.0.and.y_loc_v.lt.0)omega_v=-pi/2
                                 lill_v=sqrt((x_loc_v)**2+(y_loc_v)**2)
                                 xprimev=lill_v*cos(omega_v+phiprime(ibuild))
                                 yprimev=lill_v*sin(omega_v+phiprime(ibuild))

! w component
                                 if(x_loc_w.ne.0)then
                                    omega_w=atan(y_loc_w/x_loc_w)
!if x_loc <0 omega = pi/2 - omega
                                    if(x_loc_w.lt.0.and.y_loc_w.gt.0)omega_w= pi - abs(omega_w)
                                    if(x_loc_w.lt.0.and.y_loc_w.lt.0)omega_w=-(pi - abs(omega_w))
                                 endif
                                 if(x_loc_w.eq.0.and.y_loc_w.ge.0)omega_w=pi/2
                                 if(x_loc_w.eq.0.and.y_loc_w.lt.0)omega_w=-pi/2
                                 lill_w=sqrt((x_loc_w)**2+(y_loc_w)**2)
                                 xprimew=lill_w*cos(omega_w+phiprime(ibuild))
                                 yprimew=lill_w*sin(omega_w+phiprime(ibuild))

                                 chiu=0.
                                 chiv=0.
                         chiw=0.
                                 if(phiprime(ibuild).ne.0.)then
      
! for u-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.-ss)then
                                          chiu=abs((yprimeu+ss)*tan(phiprime(ibuild)))
                                       endif
                                       if(yprimeu.lt.-ss)then
                                          chiu= abs(ss+yprimeu)/tan(phiprime(ibuild))
                                       endif
                                    endif

                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.abs(ss))then
                                          chiu=(yprimeu-abs(ss))*tan(phiprime(ibuild))
                                       endif
                                       if(yprimeu.lt.abs(ss))then
                                          chiu=abs(abs(yprimeu)+abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

! for v-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.-ss)then
                                          chiv=abs((yprimev+ss)*tan(phiprime(ibuild)))
                                       endif
                                       if(yprimev.lt.-ss)then
                                          chiv= abs(ss+yprimev)/tan(phiprime(ibuild))
                                       endif
                                    endif


                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.abs(ss))then
                                          chiv=(yprimev-abs(ss))*tan(phiprime(ibuild))
                                       endif
                                       if(yprimev.lt.abs(ss))then
                                          chiv=abs(abs(yprimev)+abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

! for w-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.-ss)then
                                          chiw=abs((yprimew+ss)*tan(phiprime(ibuild)))
                                       endif
                                       if(yprimew.lt.-ss)then
                                          chiw= abs(ss+yprimew)/tan(phiprime(ibuild))
                                       endif
                                    endif


                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.abs(ss))then
                                          chiw=(yprimew-abs(ss))*tan(phiprime(ibuild))
                                       endif
                                       if(yprimew.lt.abs(ss))then
                                          chiw=abs(abs(yprimew)+abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif
   



                                 endif !phi prime if to avoid divide by zero.

!erp introduce new variables for new modification 10/9/2003
! dlu,dlv,yprimeu, yprimev, dNu and dNv
                                 if(xprimev+chiv.gt.0.and.chiv.le.Leff(ibuild))then

! v-component
                                    dlv = xprimev + chiv
                                    if(abs(yprimev).le.(Weff(ibuild)/2.).and.zb.ge.0)then
         
                                       dNv=sqrt((1.-(yprimev/(Weff(ibuild)/2.))**2)*   &
                                                      ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiv
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlv.gt.0.and.dlv.le.3.*dNv .and. icellflag(i,j,k).ne.4)then
!                                       if(dlv.gt.0.and.dlv.le.3.*dNv)then
                                          vo(i,j,k)=vo(i,j,k)*(1.-(dNv/dlv)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
! calculate cavity flow velocities
                                       if(dlv.le.dNv.and.dNv.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          vo(i,j,k)=-vo(istart(ibuild)-1,jstart(ibuild),kend(ibuild)   &
                                                       +1)*(1.-dlv/dNv)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif
!u-component          
                                 if(xprimeu+chiu.gt.0.and.chiu.le.Leff(ibuild))then
                                    dlu = xprimeu + chiu

                                    if(abs(yprimeu).le.(Weff(ibuild)/2.).and.zb.ge.0)then  !erp
                                       dNu=sqrt((1.-(yprimeu/(Weff(ibuild)/2.))**2)*   &
                                                     ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiu  !erp
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlu.gt.0.and.dlu.le.3.*dNu .and. icellflag(i,j,k).ne.4)then

!                                       if(dlu.gt.0.and.dlu.le.3.*dNu)then
                                          uo(i,j,k)=uo(i,j,k)*(1.-(dNu/dlu)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
! calculate cavity flow velocities
                                       if(dlu.le.dNu.and.dNu.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          uo(i,j,k)=-uo(istart(ibuild)-1,jstart(ibuild),   &
                                                         kend(ibuild)+1)*(1.-dlu/dNu)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif !xprime + chi if


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! new cellflag definition section test erp 8-15-06

                                 if(xprimew+chiw.gt.0.and.chiw.le.Leff(ibuild))then
! w-component
                                    dlw = xprimew + chiw
                                    if(abs(yprimew).le.(Weff(ibuild)/2.).and.zb.ge.0)then
                                       dNw=sqrt((1.-(yprimew/(Weff(ibuild)/2.))**2)*   &
                                                    ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiw
! calculate wake flow velocities
                                       if(dlw.gt.0.and.dlw.le.3.*dNw .and. icellflag(i,j,k).ne.4)then
                                if(icellflag(i,j,k).ne.0)then
                              icellflag(i,j,k)=5   !far wake
                              endif
                              endif
! calculate cavity flow velocities
                                       if(dlw.le.dNw.and.dNw.gt.0)then
                                       if(icellflag(i,j,k).ne.0)then
                              icellflag(i,j,k)=4   !near wake cavity
                                       endif
                              endif
                                    endif !yprime if
                                 endif

! end new celfflage definition change
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



                              endif       !end zb if
                           enddo   lp019      
                        enddo   lp020      
                     enddo   lp021      
                  endif               !end angle 2

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
! north westerly flow
                  if(theta(ibuild).gt.270.and.theta(ibuild).lt.360)then
                     ss=(Weff(ibuild)/2.)-wprime(ibuild)
                     if(zfo(ibuild).gt.0)then
                        call building_connect
                     endif
! int changed nint in line below 8-14-2006
lp024:               do k=kbottom,kend(ibuild) !erp 7/23/03 !convert to real world unit, TZ 11/16/04
lp023:                  do j=jend(ibuild),1,-1  !change so wake ends at next bld
lp022:                     do i=istart(ibuild),nx-1
                              zb=zm(k)-zfo(ibuild)             !erp 7/23/03 distance above the ground   !convert to real world unit, TZ 11/16/04

                              if(zb.le.Ht(ibuild))then

! if P' is located South East of P
                                 if(tau.lt.(Weff(ibuild)/2.))then
                                    delta_x=ss*(sin(phiprime(ibuild))) !erp 12/17
                                    delta_y=ss*(cos(phiprime(ibuild))) !erp 12/17
                                    xpprime=Lti(ibuild)+delta_x       !effective building origin
                                    ypprime=delta_y                  !effective building origin
                                 endif
! if P' is located North West of P
                                 if(tau.ge.(Weff(ibuild)/2.))then
                                    delta_x=abs(ss)*(cos(pi/2.-phiprime(ibuild))) !erp 12/17
                                    delta_y=abs(ss)*(sin(pi/2.-phiprime(ibuild))) !erp 12/17
                                    xpprime=Lti(ibuild)-delta_x   !effective building origin
                                    ypprime=-delta_y     !effective building origin
                                 endif

                                 x_loc_u=real(i-istart(ibuild))*dx-xpprime !convert to real world unit, TZ 11/16/04
                                 y_loc_u=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
                                 x_loc_v=(real(i)+0.5-real(istart(ibuild)))*dx-xpprime !convert to real world unit, TZ 11/16/04
                                 y_loc_v=real(j-jstart(ibuild))*dy-ypprime !convert to real world unit, TZ 11/16/04
!erp 8-15-2006
                                 x_loc_w=(real(i)+0.5-real(istart(ibuild)))*dx-xpprime !convert to real world unit, TZ 11/16/04
                                 y_loc_w=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
!end erp 8-15

!if x_loc <0 omega = pi/2 - omega
!u component
!erp eliminate divide by zero
                                 if(x_loc_u.ne.0)then
                                    omega_u=atan(y_loc_u/x_loc_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.gt.0)omega_u= pi - abs(omega_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.lt.0)omega_u=-(pi - abs(omega_u))
                                 endif

                                 if(x_loc_u.eq.0.and.y_loc_u.ge.0)omega_u=pi/2
                                 if(x_loc_u.eq.0.and.y_loc_u.lt.0)omega_u=-pi/2
                                 lill_u=sqrt((x_loc_u)**2+(y_loc_u)**2)
                                 xprimeu=lill_u*cos(omega_u+phiprime(ibuild))
                                 yprimeu=lill_u*sin(omega_u+phiprime(ibuild))
! v component
                                 if(x_loc_v.ne.0)then
                                    omega_v=atan(y_loc_v/x_loc_v)
                                    if(x_loc_v.lt.0.and.y_loc_v.gt.0)omega_v= pi - abs(omega_v)
                                    if(x_loc_v.lt.0.and.y_loc_v.lt.0)omega_v=-(pi - abs(omega_v))
                                 endif
                                 if(x_loc_v.eq.0.and.y_loc_v.ge.0)omega_v=pi/2
                                 if(x_loc_v.eq.0.and.y_loc_v.lt.0)omega_v=-pi/2
                                 lill_v=sqrt((x_loc_v)**2+(y_loc_v)**2)
                                 xprimev=lill_v*cos(omega_v+phiprime(ibuild))
                                 yprimev=lill_v*sin(omega_v+phiprime(ibuild))

! w component
                                 if(x_loc_w.ne.0)then
                                    omega_w=atan(y_loc_w/x_loc_w)
                                    if(x_loc_w.lt.0.and.y_loc_w.gt.0)omega_w= pi - abs(omega_w)
                                    if(x_loc_w.lt.0.and.y_loc_w.lt.0)omega_w=-(pi - abs(omega_w))
                                 endif
                                 if(x_loc_w.eq.0.and.y_loc_w.ge.0)omega_w=pi/2
                                 if(x_loc_w.eq.0.and.y_loc_w.lt.0)omega_w=-pi/2
                                 lill_w=sqrt((x_loc_w)**2+(y_loc_w)**2)
                                 xprimew=lill_w*cos(omega_w+phiprime(ibuild))
                                 yprimew=lill_w*sin(omega_w+phiprime(ibuild))

!erp remove divide by zero 10/10/2003
                                 chiu=0.
                                 chiv=0.
                         chiw=0.
                                 if(phiprime(ibuild).ne.0.)then
! for u-component

                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.-abs(ss))then
                                          chiu=abs((yprimeu+abs(ss))*tan(phiprime(ibuild)))
                                       endif
                                       if(yprimeu.lt.-abs(ss))then
                                          chiu=abs(yprimeu-abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.abs(ss))then
                                          chiu=abs(yprimeu-abs(ss))*tan(phiprime(ibuild))
                                       endif
                                       if(yprimeu.lt.abs(ss))then
                                          chiu=abs(yprimeu-abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

!     write(77,718)i,j,xprimeu,yprimeu,chiu,ss+yprimeu
!718  format(i5,i5,4f11.5)

!end bugfix 10/03/2005

!for v-component


                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.-abs(ss))then
                                          chiv=abs((yprimev+abs(ss))*tan(phiprime(ibuild)))
                                       endif
                                       if(yprimev.lt.-abs(ss))then
                                          chiv=abs(yprimev-abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.abs(ss))then
                                          chiv=abs(yprimev-abs(ss))*tan(phiprime(ibuild))
                                       endif
                                       if(yprimev.lt.abs(ss))then
                                          chiv=abs(yprimev-abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

!for w-component

                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.-abs(ss))then
                                          chiw=abs((yprimew+abs(ss))*tan(phiprime(ibuild)))
                                       endif
                                       if(yprimew.lt.-abs(ss))then
                                          chiw=abs(yprimew-abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.abs(ss))then
                                          chiw=abs(yprimew-abs(ss))*tan(phiprime(ibuild))
                                       endif
                                       if(yprimew.lt.abs(ss))then
                                          chiw=abs(yprimew-abs(ss))/tan(phiprime(ibuild))
                                       endif
                                    endif

                                 endif !phi prime if to avoid divide by zero.

!erp introduce new variables for new modification 10/9/2003
! dlu,dlv,yprimeu, yprimev, dNu and dNv
                                 if(xprimev+chiv.gt.0.and.chiv.le.Leff(ibuild))then
                                    ! v-component
                                    dlv = xprimev + chiv
                                    if(abs(yprimev).le.(Weff(ibuild)/2.).and.zb.ge.0)then
                                       dNv=sqrt((1.-(yprimev/(Weff(ibuild)/2.))**2)*   &
                                                    ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiv
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlv.gt.0.and.dlv.le.3.*dNv .and. icellflag(i,j,k).ne.4)then

!                                       if(dlv.gt.0.and.dlv.le.3.*dNv)then
                                          vo(i,j,k)=vo(i,j,k)*(1.-(dNv/dlv)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
! calculate cavity flow velocities
                                       if(dlv.le.dNv.and.dNv.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          vo(i,j,k)=-vo(istart(ibuild)-1,jstart(ibuild),kend(ibuild)   &
                                                       +1)*(1.-dlv/dNv)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif
!u-component          
                                 if(xprimeu+chiu.gt.0.and.chiu.le.Leff(ibuild))then
                                    dlu = xprimeu + chiu

                                    if(abs(yprimeu).le.(Weff(ibuild)/2.).and.zb.ge.0)then  !erp
                                       dNu=sqrt((1.-(yprimeu/(Weff(ibuild)/2.))**2)*   &
                                                    ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiu   !erp
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlu.gt.0.and.dlu.le.3.*dNu .and. icellflag(i,j,k).ne.4)then

 !                                      if(dlu.gt.0.and.dlu.le.3.*dNu)then
                                          uo(i,j,k)=uo(i,j,k)*(1.-(dNu/dlu)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif

! calculate cavity flow velocities
                                       if(dlu.le.dNu.and.dNu.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          uo(i,j,k)=-uo(istart(ibuild)-1,jstart(ibuild),   &
                                                         kend(ibuild)+1)*(1.-dlu/dNu)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif !xprime + chi if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! new cellflag definition section test erp 8-15-06

                                 if(xprimew+chiw.gt.0.and.chiw.le.Leff(ibuild))then
! w-component
                                    dlw = xprimew + chiw
                                    if(abs(yprimew).le.(Weff(ibuild)/2.).and.zb.ge.0)then
                                       dNw=sqrt((1.-(yprimew/(Weff(ibuild)/2.))**2)*   &
                                                    ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiw
! calculate wake flow velocities
                                       if(dlw.gt.0.and.dlw.le.3.*dNw .and. icellflag(i,j,k).ne.4)then
                                       if(icellflag(i,j,k).ne.0)then
                              icellflag(i,j,k)=5   !far wake
                              endif
                              endif
! calculate cavity flow velocities
                                       if(dlw.le.dNw.and.dNw.gt.0)then
                                if(icellflag(i,j,k).ne.0)then
                              icellflag(i,j,k)=4   !near wake cavity
                                        endif
                                       endif
                                    endif !yprime if
                                 endif

! end new celfflage definition change
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



                              endif       !end zb if
                           enddo   lp022      
                        enddo   lp023      
                     enddo   lp024      
                  endif               !end angle 3


!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Wake for South Easterly flow
                  if(theta(ibuild).ge.90.and.theta(ibuild).lt.180)then

                     ss=(Weff(ibuild)/2.)-wprime(ibuild)
   
                     if(zfo(ibuild).gt.0)then
                        call building_connect
                     endif
!erp end test   
!erp     do k=int(zfo(ibuild)),kend(ibuild)
! int changed nint in line below 8-14-2006
lp027:               do k=kbottom,kend(ibuild)  !convert to real world unit, TZ 11/16/04

lp026:                  do j=jstart(ibuild),ny-1   !change so wake ends at next bld
lp025:                     do i=iend(ibuild),1,-1
!erp mod 7/23/03   zb=real(k)-zbo             !distance above the ground
!   zb=real(k)-zfo(ibuild)             !erp 7/23/03 distance above the ground
                              zb=zm(k)-zfo(ibuild)      !erp 7/23/03 distance above the ground !convert to real world unit, TZ 11/16/04

                              if(zb.le.Ht(ibuild))then

! if P' is located South East of P
                                 if(tau.lt.(Weff(ibuild)/2.))then
                                    delta_x=ss*(sin(phiprime(ibuild))) !erp 12/17
                                    delta_y=ss*(cos(phiprime(ibuild))) !erp 12/17
                                    xpprime= -delta_x     !effective building origin
                                    ypprime= Wti(ibuild)-delta_y     !effective building origin
                                 endif
! if P' is located North West of P
                                 if(tau.ge.(Weff(ibuild)/2.))then
                                    delta_x=abs(ss)*(sin(phiprime(ibuild))) !erp 12/17
                                    delta_y=abs(ss)*(cos(phiprime(ibuild))) !erp 12/17
                                    xpprime= delta_x                  !effective building origin
                                    ypprime= Wti(ibuild)+delta_y     !effective building origin
                                 endif

!flip x-coordinates around

                                 x_loc_u=-(real(i-istart(ibuild))*dx-xpprime) !convert to real world unit, TZ 11/16/04
                                 y_loc_u=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
                                 x_loc_v=-((real(i)+0.5-real(istart(ibuild)))*dx-xpprime) !convert to real world unit, TZ 11/16/04
                                 y_loc_v=real(j-jstart(ibuild))*dy-ypprime !convert to real world unit, TZ 11/16/04
!erp 8-15-2006
                                 x_loc_w=-((real(i)+0.5-real(istart(ibuild)))*dx-xpprime) !convert to real world unit, TZ 11/16/04
                                 y_loc_w=(real(j)+0.5-real(jstart(ibuild)))*dy-ypprime !convert to real world unit, TZ 11/16/04
!end erp 8-15

!u component
!erp eliminate divide by zero
                                 if(x_loc_u.ne.0)then
                                    omega_u=atan(y_loc_u/x_loc_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.gt.0)omega_u= pi - abs(omega_u)
                                    if(x_loc_u.lt.0.and.y_loc_u.lt.0)omega_u=-(pi - abs(omega_u))

                                 endif
                                 if(x_loc_u.eq.0.and.y_loc_u.ge.0)omega_u=pi/2
                                 if(x_loc_u.eq.0.and.y_loc_u.lt.0)omega_u=-pi/2
                                 lill_u=sqrt((x_loc_u)**2+(y_loc_u)**2)
                                 xprimeu=lill_u*cos(omega_u-phiprime(ibuild))
                                 yprimeu=lill_u*sin(omega_u-phiprime(ibuild))

! v component
                                 if(x_loc_v.ne.0)then
                                    omega_v=atan(y_loc_v/x_loc_v)
                                    if(x_loc_v.lt.0.and.y_loc_v.gt.0)omega_v= pi - abs(omega_v)
                                    if(x_loc_v.lt.0.and.y_loc_v.lt.0)omega_v=-(pi - abs(omega_v))
                                 endif
                                 if(x_loc_v.eq.0.and.y_loc_v.ge.0)omega_v=pi/2
                                 if(x_loc_v.eq.0.and.y_loc_v.lt.0)omega_v=-pi/2
                                 lill_v=sqrt((x_loc_v)**2+(y_loc_v)**2)
!erp  10/9/2003
                                 xprimev=lill_v*cos(omega_v-phiprime(ibuild))
                                 yprimev=lill_v*sin(omega_v-phiprime(ibuild))
! w component
                                 if(x_loc_w.ne.0)then
                                    omega_w=atan(y_loc_w/x_loc_w)
                                    if(x_loc_w.lt.0.and.y_loc_w.gt.0)omega_w= pi - abs(omega_w)
                                    if(x_loc_w.lt.0.and.y_loc_w.lt.0)omega_w=-(pi - abs(omega_w))
                                 endif
                                 if(x_loc_w.eq.0.and.y_loc_w.ge.0)omega_w=pi/2
                                 if(x_loc_w.eq.0.and.y_loc_w.lt.0)omega_w=-pi/2
                                 lill_w=sqrt((x_loc_w)**2+(y_loc_w)**2)
                                 xprimew=lill_w*cos(omega_w-phiprime(ibuild))
                                 yprimew=lill_w*sin(omega_w-phiprime(ibuild))


                                 chiu=0.
                                 chiv=0.
                         chiw=0.

                                 if(phiprime(ibuild).ne.0.)then
! for u-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.ss)then
                                          chiu=abs((yprimeu-(ss))/tan(phiprime(ibuild)))
                                       endif
                                       if(yprimeu.lt.ss)then
                                          chiu=abs((ss)-yprimeu)*tan(phiprime(ibuild))
                                       endif
                                    endif
      
                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimeu.ge.ss)then
                                          chiu=abs(yprimeu-(ss))/tan(phiprime(ibuild))
                                       endif
                                       if(yprimeu.lt.ss)then
                                          chiu=abs(yprimeu-(ss))*tan(phiprime(ibuild))
                                       endif
                                    endif
!for v-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.ss)then
                                          chiv=abs((yprimev-(ss))/tan(phiprime(ibuild)))
                                       endif
                                       if(yprimev.lt.ss)then
                                          chiv=abs((ss)-yprimev)*tan(phiprime(ibuild))
                                       endif
                                    endif
      
                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimev.ge.ss)then
                                          chiv=abs(yprimev-(ss))/tan(phiprime(ibuild))
                                       endif
                                       if(yprimev.lt.ss)then ! Bug changed yprimeu to yprimev 8-15-06
                                          chiv=abs(yprimev-(ss))*tan(phiprime(ibuild))
                                       endif
                                    endif

!for w-component
                                    if(tau.lt.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.ss)then
                                          chiw=abs((yprimew-(ss))/tan(phiprime(ibuild)))
                                       endif
                                       if(yprimew.lt.ss)then
                                          chiw=abs((ss)-yprimew)*tan(phiprime(ibuild))
                                       endif
                                    endif
      
                                    if(tau.ge.(Weff(ibuild)/2.))then
                                       if(yprimew.ge.ss)then
                                          chiw=abs(yprimew-(ss))/tan(phiprime(ibuild))
                                       endif
                                       if(yprimew.lt.ss)then ! Bug changed yprimeu to yprimev 8-15-06
                                          chiw=abs(yprimew-(ss))*tan(phiprime(ibuild))
                                       endif
                                    endif

                                 endif !phi prime if to avoid divide by zero.

!erp introduce new variables for new modification 10/9/2003
! dlu,dlv,yprimeu, yprimev, dNu and dNv
                                 if(xprimev+chiv.gt.0.and.chiv.le.Leff(ibuild))then

! v-component

                                    dlv = xprimev + chiv
                                    if(abs(yprimev).le.(Weff(ibuild)/2.).and.zb.ge.0)then

                                       dNv=sqrt((1.-(yprimev/(Weff(ibuild)/2.))**2)*   &
                                                     ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiv
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlv.gt.0.and.dlv.le.3.*dNv .and. icellflag(i,j,k).ne.4)then
!                                       if(dlv.gt.0.and.dlv.le.3.*dNv)then
                                          vo(i,j,k)=vo(i,j,k)*(1.-(dNv/dlv)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
! calculate cavity flow velocities
                                       if(dlv.le.dNv.and.dNv.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          vo(i,j,k)=-vo(istart(ibuild)-1,jstart(ibuild),kend(ibuild)   &
                                                       +1)*(1.-dlv/dNv)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif
!u-component          
                                 if(xprimeu+chiu.gt.0.and.chiu.le.Leff(ibuild))then
                                    dlu = xprimeu + chiu

                                    if(abs(yprimeu).le.(Weff(ibuild)/2.).and.zb.ge.0)then  !erp
                                       dNu=sqrt((1.-(yprimeu/(Weff(ibuild)/2.))**2)*   &
                                                   ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiu !erp
! calculate wake flow velocities
! new check to see if a near cavity was written already; if a near cavity exists
! do not overwrite cell with far wake
                                       if(dlu.gt.0.and.dlu.le.3.*dNu .and. icellflag(i,j,k).ne.4)then
!                                       if(dlu.gt.0.and.dlu.le.3.*dNu)then
                                          uo(i,j,k)=uo(i,j,k)*(1.-(dNu/dlu)**(1.5))
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=5
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
! calculate cavity flow velocities
                                       if(dlu.le.dNu.and.dNu.gt.0)then
!erp 9/12/2003 rooftop flag fix vo needs to be the rooftop velocity
!just upstream of the building
                                          uo(i,j,k)=-uo(istart(ibuild)-1,jstart(ibuild),   &
                                                         kend(ibuild)+1)*(1.-dlu/dNu)**2
                                          wo(i,j,k)=0.
! MAN 7/8/2005 Celltype definition change
!                                          if(i.lt.nx.and.j.lt.ny.and.k.lt.nz)then
!                                             if(icellflag(i,j,k).ne.0)then
!                                                icellflag(i,j,k)=4
!                                             endif
!                                          endif
!end MAN 7/8/2005
                                       endif
                                    endif !yprime if
                                 endif !xprime + chi if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! new cellflag definition section test erp 8-15-06

                                 if(xprimew+chiw.gt.0.and.chiw.le.Leff(ibuild))then
! w-component
                                    dlw = xprimew + chiw
                                    if(abs(yprimew).le.(Weff(ibuild)/2.).and.zb.ge.0)then
                                       dNw=sqrt((1.-(yprimew/(Weff(ibuild)/2.))**2)*   &
                                                    ((1.-(zb/(Ht(ibuild)))**2)*Lr(ibuild)**2))+chiw
! calculate wake flow velocities
                                       if(dlw.gt.0.and.dlw.le.3.*dNw .and. icellflag(i,j,k).ne.4)then
                                if(icellflag(i,j,k).ne.0)then
                              icellflag(i,j,k)=5   !far wake
                                        endif
                              endif
! calculate cavity flow velocities
                                       if(dlw.le.dNw.and.dNw.gt.0)then
                                if(icellflag(i,j,k).ne.0)then
                              icellflag(i,j,k)=4   !near wake cavity
                                       endif
                              endif
                                    endif !yprime if
                                 endif

! end new celfflage definition change
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                              endif       !end zb if
                           enddo   lp025      
                        enddo   lp026      
                     enddo   lp027      
                  endif               !end angle 4

!               endif       !end wakeflow

            endif ! erp end subdomain check 5/3/03 building loop 5

!  endif ! erp end if for street canyon flag check 03/09/04
         return
      end
