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
      subroutine defbuild
!************************************************************************
! defbuild- define building geometries		
! 	  - called by bcsetup.f			
!	  - calls pentagon.f							
! this is a new subroutine as of 7/26/03 that allows buildings that are
! none orthagonal to the main coordinate system to be used.
! gamma - building angle, is valid +/- 45 degrees
!
!ERP 2003			
!************************************************************************
         use datamodule ! make data from module "datamodule" visible

         implicit none
         real x1,x2,x3,x4,y1,y2,y3,y4,xL1,xL2,yL3,yL4,x_c,y_c,z_c
         real x1in,x2in,x3in,x4in,y1in,y2in,y3in,y4in,xL1in,xL2in,yL3in,yL4in
         real xfoin,yfoin,court_frac
         real slope,xmin,ymin,xmax,ymax
         !real testcany,testcanx,sstaryN,sstaryS,sstarxW,sstarxE,LoverH ! MAN 03/09/2007
         real yco,xco
!         real chk,chk2,chkH,Hlow_east,Hlow_west,Hlow_north,Hlow_south !MAN 7/6/2006
!         real Lup_north,Lup_south,Wup_north,Wup_south,Lup_east,Lup_west,Wup_east,Wup_west ! MAN 03/09/2007
         real radius,thetacell,radius_out,radius_in,r_c,wall_thickness,roof_ratio,roof_zfo
         real z_c_x,z_c_y
         integer kroof
         
         pi=4.*atan(1.0) !NLB,pi
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! this is just a building generation part
! define cellflags & initialize all cells as fluid cells
         do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
                  celltype(i,j,k)=40		!initialize fluid with no boundarys erp/4/16/04
                  icellflag(i,j,k)=1.		!fluid  ! MAN 7/8/2005 Celltype definition change
! if cells are on the boundary make them inflow outflow to start
! they will be changed if necessary
                  if(i .eq. 1 .or. i .eq. nx-1 .or. j .eq. 1 .or. j   &
                        .eq. ny-1 .or. k .eq. 1 .or. k .eq. nz-1)then
                     celltype(i,j,k)=41
                     e(i,j,k)=0.
                     f(i,j,k)=0.
                     g(i,j,k)=0.
                     h(i,j,k)=0.
                     m(i,j,k)=0.
                     n(i,j,k)=0.
                     o(i,j,k)=1.
                     p(i,j,k)=0.
                     q(i,j,k)=0.
                  endif
               enddo
            enddo
         enddo
! make solid gound, ie the floor
         icellflag(:,:,1)=0 ! MAN 7/8/2005 Celltype definition change
         ibldflag(:,:,:)=0 ! MAN 8/29/2007 building flags
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc	
! Generate buildings
! this now includes a building generation loop that allows
! for multiple buildings
! calculate building spacing s and set wake flags
         do ibuild=1,inumbuild
            xbo(ibuild)=xfo(ibuild)+Lti(ibuild)	!cent back of cube
            ybo(ibuild)=yfo(ibuild)
            xlo(ibuild)=xfo(ibuild)+Lt(ibuild)		!cent bott. of cube
            ylo(ibuild)=yfo(ibuild)-Wt(ibuild)
            xuo(ibuild)=xlo(ibuild)				!center upper of cube
            yuo(ibuild)=yfo(ibuild)+Wt(ibuild)
            istart(ibuild)=nint(xfo(ibuild)/dx)+1		!front of the building	!convert back to real world unit, TZ 10/29/04
            iend(ibuild)=istart(ibuild)+nint(Lti(ibuild)/dx-1)	!back of the bld	!convert back to real world unit, TZ 10/29/04
! next two lines int's changed to nint's 8-14-2006
            jend(ibuild)=nint((yfo(ibuild)+Wt(ibuild))/dy)  !far side of bld	!convert back to real world unit, TZ 10/29/04
            jstart(ibuild)=nint(1+(yfo(ibuild)-Wt(ibuild))/dy)!close side of bld	!convert back to real world unit, TZ 10/29/04
         enddo
!building Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1 begin
! Begin First Loop through all buildings
!lp001:   do ibuild=1,inumbuild				!start building loop 1
!! erp add subdomain check 5/2/03 building loop 1
!            !         if(xfo(ibuild)-1.ge.x_subdomain_start .and. &
!            !     &   xfo(ibuild)-1.lt.x_subdomain_end   .and. &
!            !     &   yfo(ibuild)-1.ge.y_subdomain_start .and. &
!            !     &   yfo(ibuild)-1.le.y_subdomain_end)then
!! dx change for subdomain erp 6/8/2006
!            if(xfo(ibuild)-dx.ge.x_subdomain_start .and.   &
!                 xfo(ibuild)-dx.lt.x_subdomain_end   .and.   &
!                 yfo(ibuild)-dy.ge.y_subdomain_start .and.   &
!                 yfo(ibuild)-dy.le.y_subdomain_end)then
!! initially all buildings have front and wake flags set to yes
!! and canyon flags set to no
!               f_flag(ibuild)=1
!               w_flag(ibuild)=1
!               c_flag_E(ibuild)=0 !MAN 7/5/2006
!               c_flag_W(ibuild)=0 !MAN 7/5/2006
!               c_flag_N(ibuild)=0 !MAN 7/5/2006
!               c_flag_S(ibuild)=0 !MAN 7/5/2006
!!               Sx_west(ibuild)=9999.	!erp 3/10/05 new check for upwind distance !MAN 7/5/2006
!!               Sy_south(ibuild)=9999.	!erp 3/10/05 new check for upwind distance !MAN 7/5/2006
!!               Sx_east(ibuild)=9999. !MAN 7/5/2006
!!               Sy_north(ibuild)=9999. !MAN 7/5/2006
!! calculate building spacings Sx & Sy and set canyon flow flags
!               if(inumbuild.gt.1)then
!! begin changes PKK 05/12/03
!                  Havx(ibuild)=0	!initialize average building height
!                  Hlowx(ibuild)=0	!initialize lower building height
!                  Hdifx(ibuild)=0	!initialize height	difference
!! end changes PKK 05/12/03
!                  Hlow_east=dz*0.0001 !MAN 7/6/2006
!                  Hlow_west=dz*0.0001 !MAN 7/6/2006
!                  Hlow_north=dz*0.0001 !MAN 7/6/2006
!                  Hlow_south=dz*0.0001 !MAN 7/6/2006
!                  Lup_north=dz*0.0001
!                  Lup_south=dz*0.0001
!                  Wup_north=dz*0.0001
!                  Wup_south=dz*0.0001
!                  Lup_east=dz*0.0001
!                  Lup_west=dz*0.0001
!                  Wup_east=dz*0.0001
!                  Wup_west=dz*0.0001
!! MAN 7/6/2006 Modified code to check for street canyons using minimum of S/H ratio.	
!!                  Sx_east(ibuild)=9999.  !MAN 7/5/2006
!                  do j= 1,inumbuild
!                     if(ibuild.ne.j)then
!                        chk=-(xbo(ibuild)-xfo(j))
!                        chk2=-(xbo(j)-xfo(ibuild))
!                        chkH=min(Ht(ibuild),Ht(j))                           
!! MAN 7/6/2006 Checking vertical overlap
!                        if(zfo_actual(ibuild).lt.Ht(j).and.Ht(ibuild).gt.zfo_actual(j))then !MAN 7/6/2006
!! MAN 7/6/2006 Checking for street canyon on east side of building
!                           if(chk/chkH.lt.Sx_east(ibuild)/Hlow_east.and.chk.gt.0)then
!!                              if(yfo(ibuild).le.yuo(j).and.yfo(ibuild).ge.ylo(j))then !MAN 7/5/2006
!                              if(ylo(ibuild).lt.yuo(j).and.yuo(ibuild).gt.ylo(j))then !MAN 7/5/2006
!                                 Sx_east(ibuild)=chk
!                                 jstart_canyon_E(ibuild)=max(jstart(ibuild),jstart(j)) !MAN 7/5/2006
!                                 jend_canyon_E(ibuild)=min(jend(ibuild),jend(j)) !MAN 7/5/2006
!                                 ! MAN 07/25/2008 stretched vertical grid
!                                 do k=2,kend(ibuild)
!                                    if(chkH .lt. zm(k+1))then
!                                       kend_canyon_E(ibuild)=k
!                                       exit
!                                    endif
!                                 enddo
!! begin changes PKK 05/12/03
!! calculate average and lower building height, and height diff. 
!! needed for CPB pot. flow street can. initialization
!                                 Havx(ibuild)=(Ht(ibuild)+Ht(j))/2	!average building height
!                                 Hlowx(ibuild)=min(Ht(ibuild),Ht(j))	!lower building height
!                                 Hdifx(ibuild)=abs(Ht(ibuild)-Ht(j))	!height	difference
!! end changes PKK 05/12/03
!                                 Hlow_east=chkH !MAN 7/6/2006
!! MAN 03/09/2007 check for upwind building
!                                 if(theta(ibuild) .ge. 180.)then
!                                    Lup_east=Lti(ibuild)
!                                    Wup_east=Wti(ibuild)
!                                 else
!                                    Lup_east=Lti(j)
!                                    Wup_east=Wti(j)
!                                 endif
!! end MAN 03/09/2007
!                              endif
!                           endif
!                        !erp
!! MAN 7/6/2006 Checking for street canyon on west side of building
!                           if(chk2/chkH.lt.Sx_west(ibuild)/Hlow_west.and.chk2.gt.0)then 
!!                              if(yfo(ibuild).le.yuo(j).and.yfo(ibuild).ge.ylo(j))then
!                              if(ylo(ibuild).lt.yuo(j).and.yuo(ibuild).gt.ylo(j))then 
!                                 Sx_west(ibuild)=chk2
!                                 jstart_canyon_W(ibuild)=max(jstart(ibuild),jstart(j)) !MAN 7/5/2006
!                                 jend_canyon_W(ibuild)=min(jend(ibuild),jend(j)) !MAN 7/5/2006
!                                 ! MAN 07/25/2008 stretched vertical grid
!                                 do k=2,kend(ibuild)
!                                    if(chkH .lt. zm(k+1))then
!                                       kend_canyon_W(ibuild)=k
!                                       exit
!                                    endif
!                                 enddo
!	                              Havx(ibuild)=(Ht(ibuild)+Ht(j))/2	!average building height
!                                 Hlowx(ibuild)=min(Ht(ibuild),Ht(j))	!lower building height
!                                 Hdifx(ibuild)=abs(Ht(ibuild)-Ht(j))	!height	difference
!                                 Hlow_west=chkH !MAN 7/6/2006
!! MAN 03/09/2007 check for upwind building
!                                 if(theta(ibuild) .le. 180.)then
!                                    Lup_west=Lti(ibuild)
!                                    Wup_west=Wti(ibuild)
!                                 else
!                                    Lup_west=Lti(j)
!                                    Wup_west=Wti(j)
!                                 endif
!! end MAN 03/09/2007
!                              endif 
!                           endif 
!                        endif
!!erp end
!                     endif
!                  enddo
!! begin changes PKK 05/12/03
!                  Havy(ibuild)=0	!initialize average building height
!                  Hlowy(ibuild)=0	!initialize lower building height
!                  Hdify(ibuild)=0	!initialize height	difference
!! end changes PKK 05/12/03
!!                  Sy_north(ibuild)=9999. !MAN 7/5/2006
!                  do j= 1,inumbuild
!                     if(ibuild.ne.j)then
!                        chk=-(yuo(ibuild)-ylo(j))
!                        chk2=-(yuo(j)-ylo(ibuild))	!erp 3/10/05
!                        chkH=min(Ht(ibuild),Ht(j))
!! MAN 7/6/2006 Checking vertical overlap
!                        if(zfo_actual(ibuild).lt.Ht(j).and.Ht(ibuild).gt.zfo_actual(j))then !MAN 7/6/2006
!! MAN 7/6/2006 Checking for street canyon on north side of building
!                           if(chk/chkH.lt.Sy_north(ibuild)/Hlow_north.and.chk.gt.0.)then
!!                              if(xfo(ibuild).le.xbo(j).and.xfo(ibuild).ge.xfo(j))then !MAN 7/5/2006
!                              if(xfo(ibuild).lt.xbo(j).and.xbo(ibuild).gt.xfo(j))then
!                                 Sy_north(ibuild)=chk
!                                 istart_canyon_N(ibuild)=max(istart(ibuild),istart(j)) !MAN 7/5/2006
!                                 iend_canyon_N(ibuild)=min(iend(ibuild),iend(j)) !MAN 7/5/2006
!                                 ! MAN 07/25/2008 stretched vertical grid
!                                 do k=2,kend(ibuild)
!                                    if(chkH .lt. zm(k+1))then
!                                       kend_canyon_N(ibuild)=k
!                                       exit
!                                    endif
!                                 enddo
!! begin changes PKK 05/12/03
!! calculate average and lower building height, and height diff. 
!! needed for CPB pot. flow street can. initialization
!                                 Havy(ibuild)=(Ht(ibuild)+Ht(j))/2	!average building height
!                                 Hlowy(ibuild)=min(Ht(ibuild),Ht(j))	!lower building height
!                                 Hdify(ibuild)=abs(Ht(ibuild)-Ht(j))	!height	difference
!! end changes PKK 05/12/03
!                                 Hlow_north=chkH !MAN 7/6/2006
!! MAN 03/09/2007 check for upwind building
!                                 if(theta(ibuild) .ge. 90. .and. theta(ibuild) .lt. 270.)then
!                                    Lup_north=Lti(ibuild)
!                                    Wup_north=Wti(ibuild)
!                                 else
!                                    Lup_north=Lti(j)
!                                    Wup_north=Wti(j)
!                                 endif
!! end MAN 03/09/2007
!                              endif
!                           endif
!!erp 3/10/05
!! MAN 7/6/2006 Checking for street canyon on south side of building
!                           if(chk2/chkH.lt.Sy_south(ibuild)/Hlow_south.and.chk2.gt.0.)then 
!!                              if(xfo(ibuild).le.xbo(j).and.xfo(ibuild).ge.xfo(j))then !MAN 7/5/2006
!                              if(xfo(ibuild).lt.xbo(j).and.xbo(ibuild).gt.xfo(j))then !MAN 7/5/2006
!                                 Sy_south(ibuild)=chk2 
!                                 istart_canyon_S(ibuild)=max(istart(ibuild),istart(j)) !MAN 7/5/2006
!                                 iend_canyon_S(ibuild)=min(iend(ibuild),iend(j)) !MAN 7/5/2006
!                                 ! MAN 07/25/2008 stretched vertical grid
!                                 do k=2,kend(ibuild)
!                                    if(chkH .lt. zm(k+1))then
!                                       kend_canyon_S(ibuild)=k
!                                       exit
!                                    endif
!                                 enddo
!	                              Havy(ibuild)=(Ht(ibuild)+Ht(j))/2	!average building height
!                                 Hlowy(ibuild)=min(Ht(ibuild),Ht(j))	!lower building height
!                                 Hdify(ibuild)=abs(Ht(ibuild)-Ht(j))	!height	difference
!                                 Hlow_south=chkH !MAN 7/6/2006
!! MAN 03/09/2007 check for upwind building
!                                 if(theta(ibuild) .ge. 90. .and. theta(ibuild) .lt. 270.)then
!                                    Lup_south=Lti(j)
!                                    Wup_south=Wti(j)
!                                 else
!                                    Lup_south=Lti(ibuild)
!                                    Wup_south=Wti(ibuild)
!                                 endif
!! end MAN 03/09/2007
!                              endif
!                           endif 
!                        endif 
!!erp end
!                     endif
!                  enddo
!                  testcany=Sy_north(ibuild)*sin(phi(ibuild)*pi/180.)
!                  testcanx=Sx_east(ibuild)*cos(phi(ibuild)*pi/180.)
!                  write(44,*)'Adding Flow Features to building no.',ibuild
!                  write(44,*)ibuild,'Sx = ',Sx_east(ibuild),testcanx
!
!! original roeckle scheme   :Wt replaced with Wti AAG 09-26-06
!! MAN 03/09/2007 made sstar directionally dependent and use the smaller building height 
!                  if(streetcanyonflag .ne. 4)then
!                     sstarxE=Hlow_east*(1.25 + 0.15*Wup_east/Hlow_east)
!                     if(Wup_east/Hlow_east.ge.2.)sstarxE=1.55*Hlow_east
!                     sstarxW=Hlow_west*(1.25 + 0.15*Wup_west/Hlow_west)
!                     if(Wti(ibuild)/Hlow_west.ge.2.)sstarxW=1.55*Hlow_west
!                  endif
!! AAG changed defination of sstarx    8/16/06
!! using the fackrell cavity length formula & put a upper and lower limit on LoverH
!                  if(streetcanyonflag .eq. 4)then
!                     LoverH=Lup_east/Hlow_east
!			            if(LoverH.gt.3.)LoverH=3.
!			            if(LoverH.lt.0.3)LoverH=0.3
!                     sstarxE=1.8*Wup_east/((LoverH**(0.3))*(1+0.24*Wup_east/Hlow_east))
!                     LoverH=Lup_west/Hlow_west
!			            if(LoverH.gt.3.)LoverH=3.
!			            if(LoverH.lt.0.3)LoverH=0.3
!                     sstarxW=1.8*Wup_west/((LoverH**(0.3))*(1+0.24*Wup_west/Hlow_west))
!                  endif
!! check to see if flow is skimming or isolated                   
!                  if(Sx_east(ibuild).le. sstarxE .and. abs(Sx_east(ibuild)*cos(phi(ibuild)*pi/180.)   &
!                     ).gt. 0.01)then
!                     c_flag_E(ibuild)=1				!add street canyon
!                  endif
!                  if(Sx_west(ibuild).le. sstarxW .and. abs(Sx_west(ibuild)*cos(phi(ibuild)*pi/180.)   &
!                     ).gt. 0.01)then
!                     c_flag_W(ibuild)=1				!add street canyon
!                  endif
!                  !erp 3/10/05
!!	 if(Sx_west(ibuild).le. sstarx .and. abs(Sx_east(ibuild)*cos(phi*pi/180.)
!!     &).gt. 0.01)then
!!	   c_flag_E(ibuild)=1				!add street canyon
!!	 endif 
!!erp end
!
!! original roeckle scheme   :Lt replaced with Lti AAG 09-26-06
!                  if(streetcanyonflag .ne. 4)then
!                     sstaryN=Hlow_north*(1.25 + 0.15*Lup_north/Hlow_north)
!                     if(Lup_north/Hlow_north.ge.2.)sstaryN=1.55*Hlow_north
!                     sstaryS=Hlow_south*(1.25 + 0.15*Lup_south/Hlow_south)
!                     if(Lup_south/Hlow_south.ge.2.)sstaryS=1.55*Hlow_south
!                  endif      
!! AAG changed defination of sstarx    8/16/06
!! using the fackrell cavity length formula & put a upper and lower limit on LoverH  
!                  if(streetcanyonflag .eq. 4)then
!                     LoverH=Wup_north/Hlow_north
!			            if(LoverH.gt.3.)LoverH=3.
!			            if(LoverH.lt.0.3)LoverH=0.3
!                     sstaryN=1.8*Lup_north/((LoverH**(0.3))*(1+0.24*Lup_north/Hlow_north))
!                     LoverH=Wup_south/Hlow_south
!			            if(LoverH.gt.3.)LoverH=3.
!			            if(LoverH.lt.0.3)LoverH=0.3
!                     sstaryS=1.8*Lup_south/((LoverH**(0.3))*(1+0.24*Lup_south/Hlow_south))
!                  endif
!! check to see if flow is skimming or isolated    
!                  if(Sy_north(ibuild).le.sstaryN .and. abs(Sy_north(ibuild)*sin(phi(ibuild)*pi/180.))   &
!                      .gt. 0.01)then
!                     c_flag_N(ibuild)=1				!add street canyon
!                  endif
!                  if(Sy_south(ibuild).le.sstaryS .and. abs(Sy_south(ibuild)*sin(phi(ibuild)*pi/180.))   &
!                      .gt. 0.01)then
!                     c_flag_S(ibuild)=1				!add street canyon
!                  endif
!               endif
!!end MAN 03/09/2007
!! AAG 09/13/06 rooftop recircualtion turn on/off logic for individual building
!               rooftop_flag(ibuild)=1
!               print*,ibuild,theta(ibuild),Ht(ibuild)
!               print*,'north',c_flag_N(ibuild),Hlow_north
!               print*,'east',c_flag_E(ibuild),Hlow_east
!               print*,'south',c_flag_S(ibuild),Hlow_south
!               print*,'west',c_flag_W(ibuild),Hlow_west
!               if(theta(ibuild).gt.320.or.theta(ibuild).le.40) then
!                  if(c_flag_N(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_north) then
!                     rooftop_flag(ibuild)=0
!                  endif
!               endif
!               if(theta(ibuild).gt.50.and.theta(ibuild).le.130) then
!                  if( c_flag_E(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_east) then
!                     rooftop_flag(ibuild)=0
!                  endif
!               endif
!               if(theta(ibuild).gt.140.and.theta(ibuild).le.220) then
!                  if(c_flag_S(ibuild).eq.1.and. Ht(ibuild).eq.Hlow_south) then
!                     rooftop_flag(ibuild)=0
!                  endif
!               endif
!               if(theta(ibuild).gt.230.and.theta(ibuild).le.310) then
!                  if(c_flag_W(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_west) then
!                     rooftop_flag(ibuild)=0
!				      endif
!               endif
!               if(theta(ibuild).gt.40.and.theta(ibuild).le.50) then
!                  if(c_flag_N(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_north .and. &
!                        c_flag_E(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_east) then
!                     rooftop_flag(ibuild)=0
!                  endif
!               endif
!               if(theta(ibuild).gt.130.and.theta(ibuild).le.140) then
!                  if( c_flag_E(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_east .and. &
!                        c_flag_S(ibuild).eq.1.and. Ht(ibuild).eq.Hlow_south ) then
!                     rooftop_flag(ibuild)=0
!                  endif
!               endif
!               if(theta(ibuild).gt.220.and.theta(ibuild).le.230) then
!                  if(c_flag_S(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_south .and. &
!                        c_flag_W(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_west) then
!                     rooftop_flag(ibuild)=0
!                  endif
!               endif
!               if(theta(ibuild).gt.310.and.theta(ibuild).le.320) then
!			         if(c_flag_W(ibuild).eq.1.and.Ht(ibuild).eq.Hlow_west  .and. &
!			               c_flag_N(ibuild).eq.1.and. Ht(ibuild).eq.Hlow_north) then
!				         rooftop_flag(ibuild)=0
!				      endif
!               endif
!! done calculating Sx and Sy
!               write(44,*)'cflag_x',c_flag_E(ibuild)
!               write(44,*)'cflag_y',c_flag_N(ibuild)
!               write(44,*)'fflag',f_flag(ibuild)
!               write(44,*)'wflag',w_flag(ibuild)
!            endif ! erp end subdomain check 5/2/03 loop 1
!         enddo   lp001      				!end building gen loop 1	
!building Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1Loop1 end

!building Loop2Loop2Loop2!Loop2Loop2Loop2!Loop2Loop2Loop2!Loop2Loop2 begin
!ccccc need an even number of cells in building
lp002:   do ibuild=1,inumbuild			!begin building gen loop 2
!erp removed 7/23/03  	zbo=2.	!the ground
!erp	istart(ibuild)=nint(xfo(ibuild))		!front of the building !MAN 7/5/2006 moved section up
!erp	iend(ibuild)=istart(ibuild)+nint(Lti(ibuild)-1)	!back of the bld
!erp	jend(ibuild)=nint(yfo(ibuild))+int(Wt(ibuild)-1)  !far side of bld
!erp	jstart(ibuild)=nint(yfo(ibuild))-int(Wt(ibuild))!close side of bld
!erp	kend(ibuild)=nint(Ht(ibuild))+1			!top of the bld
!            istart(ibuild)=nint(xfo(ibuild)/dx)+1		!front of the building	!convert back to real world unit, TZ 10/29/04
!            iend(ibuild)=istart(ibuild)+nint(Lti(ibuild)/dx-1)	!back of the bld	!convert back to real world unit, TZ 10/29/04
!            jend(ibuild)=nint(yfo(ibuild)/dy)+1+int(Wt(ibuild)/dy-1)  !far side of bld	!convert back to real world unit, TZ 10/29/04
!            jstart(ibuild)=nint(yfo(ibuild)/dy)+1-int(Wt(ibuild)/dy)!close side of bld	!convert back to real world unit, TZ 10/29/04
!            kend(ibuild)=nint(Ht(ibuild)/dz)+1			!top of the bld	!convert back to real world unit, TZ 10/29/04
            write(44,*)'ibuild,istart(ibuild),iend(ibuild),jstart(ibuild),jend(ibuild)'
            write(44,*)ibuild,istart(ibuild),iend(ibuild),jstart(ibuild),jend(ibuild)
!erp	write(44,*),int(Wt(ibuild)),int(Lti(ibuild)),int(Lt(ibuild))
            write(44,*)Wt(ibuild),Lti(ibuild),Lt(ibuild)	!convert back to real world unit, TZ 10/29/04
! set non-fluid cell type flags
! icellflag = 0 is a solid cell

! if the building is orthoganol to the main coordinate system
! just set the cellflags as follows
            select case(bldtype(ibuild))
               case(0,1,6) !Rectangular buildings, bridges, and negative buildings
                  if(gamma(ibuild) .eq. 0)then
!erp	do k=int(zfo(ibuild)),kend(ibuild)
! int changed to nint on next line 8-14-06	
                     do k=kstart(ibuild),kend(ibuild)	!convert back to real world unit, TZ 10/29/04
                        do j=jstart(ibuild),jend(ibuild)
                           do i=istart(ibuild),iend(ibuild)
                              if(bldtype(ibuild) .eq. 0)then
                                 icellflag(i,j,k)=1 ! MAN 7/8/2005 Celltype definition change
                                 ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags
                              else
                                 icellflag(i,j,k)=0 ! MAN 7/8/2005 Celltype definition change
                                 ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags
                              endif
                           enddo
                        enddo
                     enddo
! if the building is NON-orthoganol to the main coordinate system
! use the following algorithm
                  else
!calculate corner coordinates of the building
                     x1=xfo(ibuild)+Wt(ibuild)*sin(gamma(ibuild))
                     y1=yfo(ibuild)-Wt(ibuild)*cos(gamma(ibuild))
                     x2=x1+Lti(ibuild)*cos(gamma(ibuild))
                     y2=y1+Lti(ibuild)*sin(gamma(ibuild))
                     x4=xfo(ibuild)-Wt(ibuild)*sin(gamma(ibuild))
                     y4=yfo(ibuild)+Wt(ibuild)*cos(gamma(ibuild))
                     x3=x4+Lti(ibuild)*cos(gamma(ibuild))
                     y3=y4+Lti(ibuild)*sin(gamma(ibuild))
 271                 format(8f8.3)
                     if(gamma(ibuild).gt.0)then
                        xmin=x4
                        xmax=x2
                        ymin=y1
                        ymax=y3
                     endif
                     if(gamma(ibuild).lt.0)then
                        xmin=x1
                        xmax=x3
                        ymin=y2
                        ymax=y4
                     endif
                     istart(ibuild)=nint(xmin/dx)
                     iend(ibuild)=nint(xmax/dx)
                     jstart(ibuild)=nint(ymin/dy)
                     jend(ibuild)=nint(ymax/dy)
!erp 	do k=int(zfo(ibuild)),kend(ibuild)	
!erp        do j=int(ymin),int(ymax)
!erp	   do i=int(xmin),int(xmax)
!erp	   x_c=real(i)	!x coordinate to be checked
!erp	   y_c=real(j)	!y coordinate to be checked
! changed int to nint in next three lines 8-14-06
                     do k=kstart(ibuild),kend(ibuild)	!convert back to real world unit, TZ 10/29/04
                        do j=nint(ymin/dy)+1,nint(ymax/dy)+1	!convert back to real world unit, TZ 10/29/04
                           do i=nint(xmin/dx)+1,nint(xmax/dx)+1	!convert back to real world unit, TZ 10/29/04
                              x_c=(real(i)-0.5)*dx	!x coordinate to be checked	!convert back to real world unit, TZ 10/29/04
                              y_c=(real(j)-0.5)*dy	!y coordinate to be checked	!convert back to real world unit, TZ 10/29/04
!calculate the equations of the lines making up the 4 walls of the
!building
                              slope = (y4-y1)/(x4-x1) !slope of L1
                              xL1 = x4 + (y_c-y4)/slope
                              slope = (y3-y2)/(x3-x2) !slope of L2
                              xL2 = x3 + (y_c-y3)/slope
                              slope = (y2-y1)/(x2-x1) !slope of L3
                              yL3 = y1 + slope*(x_c-x1)
                              slope = (y3-y4)/(x3-x4) !slope of L4
                              yL4 = y4 + slope*(x_c-x4)
                  
                              if(x_c.gt.xL1.and.x_c.lt.xL2.and.y_c.gt.yL3.and.y_c.lt.yL4)then
                                 if(bldtype(ibuild) .eq. 0)then
                                    icellflag(i,j,k)=1 ! MAN 7/8/2005 Celltype definition change
                                    ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags
                                 else
                                    icellflag(i,j,k)=0 ! MAN 7/8/2005 Celltype definition change
                                    ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags
                                 endif
                              endif
                           enddo
                        enddo
                     enddo
                  endif
! generate cylindrical buildings
! need to specify a and b as the major and minor axis of
! the ellipse
! xco and yco are the coordinates of the center of the ellipse
               case(2)
                  if(aa(ibuild) .gt. 0. .and. bb(ibuild) .gt. 0.)then
                     if(gamma(ibuild) .ne. 0.)then
                        xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))
                        yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
                        istart(ibuild)=nint((xco-max(Lt(ibuild),Wt(ibuild)))/dx)
                        iend(ibuild)=nint((xco+max(Lt(ibuild),Wt(ibuild)))/dx)
                        jstart(ibuild)=nint((yco-max(Lt(ibuild),Wt(ibuild)))/dy)
                        jend(ibuild)=nint((yco+max(Lt(ibuild),Wt(ibuild)))/dy)
                     else
                        xco = xfo(ibuild) + Lt(ibuild)
                        yco = yfo(ibuild)
                     endif
                     if(istart(ibuild) .le. 0)istart(ibuild)=1
                     if(iend(ibuild) .le. 0)iend(ibuild)=nx-1
                     if(jstart(ibuild) .le. 0)jstart(ibuild)=1
                     if(jend(ibuild) .le. 0)jend(ibuild)=ny-1
!erp 7/23/03 do k=1,kend(ibuild)
!erp 	do k=int(zfo(ibuild)),kend(ibuild)	!erp 7/23/03
! int changed to nint in next line 8-14-06
                     do k=kstart(ibuild),kend(ibuild)	!erp 7/23/03	!convert back to real world unit, TZ 10/29/04
                        do j=jstart(ibuild),jend(ibuild)
                           do i=istart(ibuild),iend(ibuild)
                              x_c=(real(i)-0.5)*dx-xco
                              y_c=(real(j)-0.5)*dy-yco
                              thetacell=atan2(y_c,x_c)
                              if(sqrt(x_c**2.+y_c**2.) .le. radius(aa(ibuild),bb(ibuild),&
                                    thetacell,gamma(ibuild)))then
                                 icellflag(i,j,k)=0
                                 ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags
                              endif
                           enddo
                        enddo
                     enddo
                  else
                     write(44,*)'Error either the major or minor axis for an ellipse'
                     write(44,*)'is zero!!!'
                  endif
! build a Pentagon shaped building
               case(3)
                  call pentagon
! Rectangular Stadium Building
               case(4)
!calculate corner coordinates of the building
                  x1=xfo(ibuild)+Wt(ibuild)*sin(gamma(ibuild))
                  y1=yfo(ibuild)-Wt(ibuild)*cos(gamma(ibuild))
                  x2=x1+Lti(ibuild)*cos(gamma(ibuild))
                  y2=y1+Lti(ibuild)*sin(gamma(ibuild))
                  x4=xfo(ibuild)-Wt(ibuild)*sin(gamma(ibuild))
                  y4=yfo(ibuild)+Wt(ibuild)*cos(gamma(ibuild))
                  x3=x4+Lti(ibuild)*cos(gamma(ibuild))
                  y3=y4+Lti(ibuild)*sin(gamma(ibuild))
                  xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))
                  yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
                  if(gamma(ibuild) .gt. 0.)then
                     xmin=x4
                     xmax=x2
                     ymin=y1
                     ymax=y3
                  elseif(gamma(ibuild) .lt. 0.)then
                     xmin=x1
                     xmax=x3
                     ymin=y2
                     ymax=y4
                  else
                     xmin=x1
                     xmax=x3
                     ymin=y2
                     ymax=y4
                  endif
                  istart(ibuild)=nint(xmin/dx)
                  iend(ibuild)=nint(xmax/dx)
                  jstart(ibuild)=nint(ymin/dy)
                  jend(ibuild)=nint(ymax/dy)
                  if(atten(ibuild) .gt. 0.)then
                     roof_ratio=1
                     roof_zfo=Ht(ibuild)
                  else
                     roof_ratio=0.8
                     roof_zfo=(Ht(ibuild)-zfo_actual(ibuild))*roof_ratio+zfo_actual(ibuild)
                  endif
                  ! MAN 07/25/2008 stretched vertical grid
                  do k=kstart(ibuild),nz-1
                     kroof=k
                     if(roof_zfo .le. z(k-1))exit
                  enddo
! changed int to nint in next three lines 8-14-06
                  do k=kstart(ibuild),kroof	
                     z_c=zm(k)-zfo_actual(ibuild)	!z coordinate to be checked
                     court_frac=abs(atten(ibuild))*(1.-z_c/(roof_zfo-zfo_actual(ibuild)))
                     xfoin=xfo(ibuild)+court_frac*cos(gamma(ibuild))
                     yfoin=yfo(ibuild)+court_frac*sin(gamma(ibuild))
                     x1in=xfoin+(Wt(ibuild)-court_frac)*sin(gamma(ibuild))
                     y1in=yfoin-(Wt(ibuild)-court_frac)*cos(gamma(ibuild))
                     x2in=x1in+(Lti(ibuild)-2.*court_frac)*cos(gamma(ibuild))
                     y2in=y1in+(Lti(ibuild)-2.*court_frac)*sin(gamma(ibuild))
                     x4in=xfoin-(Wt(ibuild)-court_frac)*sin(gamma(ibuild))
                     y4in=yfoin+(Wt(ibuild)-court_frac)*cos(gamma(ibuild))
                     x3in=x4in+(Lti(ibuild)-2.*court_frac)*cos(gamma(ibuild))
                     y3in=y4in+(Lti(ibuild)-2.*court_frac)*sin(gamma(ibuild))
                     do j=nint(ymin/dy)+1,nint(ymax/dy)+1	
                        y_c=(real(j)-0.5)*dy	!y coordinate to be checked
                        do i=nint(xmin/dx)+1,nint(xmax/dx)+1	
                           x_c=(real(i)-0.5)*dx	!x coordinate to be checked	
!calculate the equations of the lines making up the 4 walls of the
!building
                           if(gamma(ibuild) .eq. 0.)then
                              xL1 = x1
                              xL1in = x1in
                              xL2 = x2
                              xL2in = x2in
                              yL3 = y1
                              yL3in = y1in
                              yL4 = y3
                              yL4in = y3in
                           else
                              slope = (y4-y1)/(x4-x1) !slope of L1
                              xL1 = x4 + (y_c-y4)/slope
                              xL1in = x4in + (y_c-y4in)/slope
                              slope = (y3-y2)/(x3-x2) !slope of L2
                              xL2 = x3 + (y_c-y3)/slope
                              xL2in = x3in + (y_c-y3in)/slope
                              slope = (y2-y1)/(x2-x1) !slope of L3
                              yL3 = y1 + slope*(x_c-x1)
                              yL3in = y1in + slope*(x_c-x1in)
                              slope = (y3-y4)/(x3-x4) !slope of L4
                              yL4 = y4 + slope*(x_c-x4)
                              yL4in = y4in + slope*(x_c-x4in)
                           endif
                           if(abs(xL1-xL1in) .lt. dx)xL1in=xL1+dx
                           if(abs(xL2-xL2in) .lt. dx)xL2in=xL2-dx
                           if(abs(yL3-yL3in) .lt. dy)yL3in=yL3+dy
                           if(abs(yL4-yL4in) .lt. dx)yL4in=yL4-dy
                           if(x_c .gt. xL1 .and. x_c .lt. xL2 .and. &
                                 y_c .gt. yL3 .and. y_c .lt. yL4)then
                              icellflag(i,j,k)=0
                              ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags    
                              if(x_c .gt. xL1in .and. x_c .lt. xL2in .and. &
                                     y_c .gt. yL3in .and. y_c .lt. yL4in)then
                                 icellflag(i,j,k)=1 ! MAN 7/8/2005 Celltype definition change
                                 ibldflag(i,j,k)=0 ! MAN 8/29/2007 building flags
                              endif
                           endif
                        enddo
                     enddo
                  enddo
                  if(atten(ibuild) .lt. 0.)then
                     court_frac=abs(atten(ibuild))
                     do j=jstart(ibuild),jend(ibuild)
                        do i=istart(ibuild),iend(ibuild)
                           x_c=((real(i)-0.5)*dx-xco)*cos(gamma(ibuild)) + ((real(j)-0.5)*dy-yco)*sin(gamma(ibuild))
                           y_c=-((real(i)-0.5)*dx-xco)*sin(gamma(ibuild)) + ((real(j)-0.5)*dy-yco)*cos(gamma(ibuild))
                           if(abs(x_c) .lt. Lt(ibuild) .and. abs(y_c) .lt. Wt(ibuild))then
                              if(abs(x_c) .gt. Lt(ibuild)-court_frac .or. abs(y_c) .gt. Wt(ibuild)-court_frac)then
                                 z_c_x=roof_zfo+(Ht(ibuild)-roof_zfo)*sqrt((Lt(ibuild)-abs(x_c))/court_frac)
                                 z_c_y=roof_zfo+(Ht(ibuild)-roof_zfo)*sqrt((Wt(ibuild)-abs(y_c))/court_frac)
                                 z_c=min(z_c_x,z_c_y)
                                 ! MAN 07/25/2008 stretched vertical grid
                                 do k=kstart(ibuild),nz-1
                                    kroof=k
                                    if(z_c .le. z(k-1))exit
                                 enddo
                                 ibldflag(i,j,kroof)=ibuild ! MAN 8/29/2007 building flags
                                 icellflag(i,j,kroof)=0
                              endif
                           endif
                        enddo
                     enddo
                  endif
! Elliptical stadium building
               case(5)
!calculate corner coordinates of the building
                  if(aa(ibuild) .gt. 0. .and. bb(ibuild) .gt. 0.)then
                     if(gamma(ibuild) .ne. 0.)then
                        xco = xfo(ibuild) + Lt(ibuild)*cos(gamma(ibuild))
                        yco = yfo(ibuild) + Lt(ibuild)*sin(gamma(ibuild))
                        istart(ibuild)=nint((xco-max(Lt(ibuild),Wt(ibuild)))/dx)
                        iend(ibuild)=nint((xco+max(Lt(ibuild),Wt(ibuild)))/dx)
                        jstart(ibuild)=nint((yco-max(Lt(ibuild),Wt(ibuild)))/dy)
                        jend(ibuild)=nint((yco+max(Lt(ibuild),Wt(ibuild)))/dy)
                     else
                        xco = xfo(ibuild) + Lt(ibuild)
                        yco = yfo(ibuild)
                     endif
                     if(atten(ibuild) .gt. 0.)then
                        roof_ratio=1
                        roof_zfo=Ht(ibuild)
                     else
                        roof_ratio=0.8
                        roof_zfo=(Ht(ibuild)-zfo_actual(ibuild))*roof_ratio+zfo_actual(ibuild)
                     endif
                     if(istart(ibuild) .le. 0)istart(ibuild)=1
                     if(iend(ibuild) .le. 0)iend(ibuild)=nx-1
                     if(jstart(ibuild) .le. 0)jstart(ibuild)=1
                     if(jend(ibuild) .le. 0)jend(ibuild)=ny-1
                     wall_thickness=max(dx,dy)
                     ! MAN 07/25/2008 stretched vertical grid
                     do k=kstart(ibuild),nz-1
                        kroof=k
                        if(roof_zfo .le. z(k-1))exit
                     enddo
!erp 7/23/03 do k=1,kend(ibuild)
!erp 	do k=int(zfo(ibuild)),kend(ibuild)	!erp 7/23/03
! int changed to nint in next line 8-14-06
                     do k=kstart(ibuild),kroof
                        z_c=zm(k)-zfo_actual(ibuild)
                        court_frac=abs(atten(ibuild))*(1.-z_c/(roof_zfo-zfo_actual(ibuild)))
                        do j=jstart(ibuild),jend(ibuild)
                           y_c=(real(j)-0.5)*dy-yco
                           do i=istart(ibuild),iend(ibuild)
                              x_c=(real(i)-0.5)*dx-xco
                              thetacell=atan2(y_c,x_c)
                              radius_out=radius(aa(ibuild),bb(ibuild),thetacell,gamma(ibuild))
                              radius_in=radius(aa(ibuild)-court_frac,bb(ibuild)-court_frac,thetacell,gamma(ibuild))
                              if(radius_out-radius_in .le. 1.25*wall_thickness)radius_in=radius_out-1.25*wall_thickness
                              r_c=sqrt(x_c**2.+y_c**2.)
                              if(r_c .le. radius_out .and. r_c .gt. radius_in)then
                                 ibldflag(i,j,k)=ibuild ! MAN 8/29/2007 building flags
                                 icellflag(i,j,k)=0
                              endif
                           enddo
                        enddo
                     enddo
                     if(atten(ibuild) .lt. 0.)then
                        court_frac=abs(atten(ibuild))
                        do j=jstart(ibuild),jend(ibuild)
                           y_c=(real(j)-0.5)*dy-yco
                           do i=istart(ibuild),iend(ibuild)
                              x_c=(real(i)-0.5)*dx-xco
                              thetacell=atan2(y_c,x_c)
                              radius_out=radius(aa(ibuild),bb(ibuild),thetacell,gamma(ibuild))
                              radius_in=radius(aa(ibuild)-court_frac,bb(ibuild)-court_frac,thetacell,gamma(ibuild))
                              r_c=sqrt(x_c**2.+y_c**2.)
                              if(r_c .gt. radius_in .and. r_c .lt. radius_out)then
                                 z_c=roof_zfo+(Ht(ibuild)-roof_zfo)*sqrt((radius_out-r_c)/(radius_out-radius_in))
                                 ! MAN 07/25/2008 stretched vertical grid
                                 do k=kstart(ibuild),nz-1
                                    kroof=k
                                    if(z_c .le. z(k-1))exit
                                 enddo
                                 ibldflag(i,j,kroof)=ibuild ! MAN 8/29/2007 building flags
                                 icellflag(i,j,kroof)=0
                              endif
                           enddo
                        enddo
                     endif
                  else
                     write(44,*)'Error either the major or minor axis for an ellipse'
                     write(44,*)'is zero!!!'
                  endif
            endselect
! erp 1/31/2003
         enddo   lp002      
! end building generation 2a
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         return
      end
