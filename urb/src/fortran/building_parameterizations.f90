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
      subroutine building_parameterizations
!************************************************************************
! bcsetup - boundary condition setup program for qwic urb      
!    - called by main.f90                 
!    - calls defbuild.f90, upwind.f90, street_intersec.f90  
! The empirical parameterizations are applied in the following order:
!  1. uninterupted boundary layer
!  2. upwind vortex cavity (calls upwind.f90)
!  3. rooftop recirculation
!  4. wakes
! 
!                 
! THIS VERSION OF QWIC-URB CONTAINS MULTIPLE BUILDING CAPABILITY  
! all velocities are cell face quantities          
! icellflag,celltype,Lagrange multipliers (p1,p2) and their    
! coeficients  (e,f,g,m,n,o,p,q,h,p) are a cell centered quantities. 
!                          
! Lfx is the length of the front eddy on a building in xdirection 
! Lfy is the length of the front eddy on a building in ydirection 
! Lr is the length of the rear vortex cavity behind  a building      
! theta is the angle of mean wind speed (meters/sec)        
! inumbuild is the number of buildings in the city array    
! xfo is the x coord. left front center of bld
! yfo is the y coord. left front center of bld
! zfo is the z coord of building base 
! xlo is the x coord. lower front center of bld
! xuo is the x coord. upper front center of bld
!                          
! REFERENCE FRAME DEFINITIONS:                  
! below  is defined as  lesser  z value (ie z(k) is below z(k+1)  
! above  is defined as  greater z value (ie z(k+1) is above z(k)  
! right  is defined as  greater x value (ie x(i+1) is right of x(i)  
! left   is defined as  lesser  x value (ie x(i) is right of x(i+1)  
! front  is defined as  greater y value (ie y(j+1) is in front of y(j)  
! behind is defined as  lesser  y value (ie y(j) is behind y(j+1) 
! ERP dec/2000                      
! moving  over various wind angle routine from earlier version of code  
! ERP August/2002                   
! Most recent Modification:
! Changing of coordintate system to reflect true stagard grid   
! ERP Sept/Oct 2002
! erp 12/17 changes fixing north/south flows
! mdw 1/8/2003 changes include fix to jstartb
! erp 7/23/03 modifications to allow for building blocks to be stacked
!  this involves modifications to k loops,zfo and zbo
! erp 11/13/03 fixing Lfx bug
! erp 11/13/03 fixing rooftop vortex bug
! erp 1/29/04 added Petra Kastner-Klein's finalized street canyon technique
!  that she idependantly tested and verified
!  an option is included to use the CPB potential flow formulas for the velocity 
!  field initialization in the street canyon
!  the input file was modified, in line 9 the initialization method is chosen
!    streetcanyonflag=1 ==> original Roeckle approach
!    streetcanyonflag=2 ==> CPB approach
!  potential flow formulas are not applied near the lateral edges of the canyon
!  depth of lateral vortex zone equal to canyon heigth, u is set to zero, v=u0(z)
! erp 2/11/04 added Nilesh's upwind vortex 
! erp 03/09/04 added canyon flag check to remove wake for skimming flow?
! NLB 02/10/04 Added Upwind Vortex Parameterizations for Perpendicular and Varying Incident Wind Angles
! NLB 10/11/04 Added Rooftop Parameterizations for Perpendicular and Varying Incident Wind Angles 
! erp 03/02/05 This subroutine now can writeout the building grid locations in
!     both a binary (QU_celltype.bin) and ASCII format (celltype2.dat). If
!     format_flag = 1, ASCII only. If format_flag=2, binary only and if
!     format_flag = 3, both the ASCII and binaries are written out
!
! ERP 6/8/2006 this version includes rooftop fixes for both off angle and normal
!  angle calculations (Suhas Pols implementation of off angle fixes)
! Cellflag designations 
!  icellflag = 0  building
!  icellflag = 1  fluid cell BL parameterization
!  icellflag = 2  upwind cavity parameterization
!  icellflag = 3  rooftop parameterization
!  icellflag = 4  near wake cavity parameterization
!  icellflag = 5  far wake cavity parameterization
!  icellflag = 6  street canyon parameterization
!  icellflag = 8  vegetation parameterization
!  icellflag = 9  street intersection parameterization
!  icellflag = 10  parking garage parameterization
!
!
!************************************************************************
         use datamodule ! make data from module "datamodule" visible

         implicit none
         real denom,num,disp_len
         real cu,cv
         real vegvelfrac,avg_atten,num_atten

!      integer icelltemp

!erp 3/1/2006 Rooftop

         

!erp 3/1/2006 Rooftop
         
         if (i_time .eq.1)then
! variables added, PKK 05/12/03
            allocate(Havx(inumbuild),Havy(inumbuild))     !average building height of a street can.
            allocate(Hlowx(inumbuild),Hlowy(inumbuild))     !lower building height of a street can.
            allocate(Hdifx(inumbuild),Hdify(inumbuild))    !lower building height of a street can.
            allocate(kref(inumbuild))    !reference level
            allocate(kcantop(inumbuild))    ! level of can. top
! end variables added, PKK 05/12/03
            allocate(celltype(nx-1,ny-1,nz-1))

! allocate arrays which are needed in bcsetup, PKK 10/01/02
! they are not passed to other subroutines and will be deallocated at the end of bcsetup 
   
            allocate(istart(inumbuild),iend(inumbuild))
            allocate(jstart(inumbuild),jend(inumbuild))
            allocate(kstart(inumbuild),kend(inumbuild))
            !allocate(istart_canyon_N(inumbuild),iend_canyon_N(inumbuild)) !MAN7/6/2006
            !allocate(jstart_canyon_E(inumbuild),jend_canyon_E(inumbuild)) !MAN7/6/2006
            !allocate(istart_canyon_S(inumbuild),iend_canyon_S(inumbuild)) !MAN7/6/2006
            !allocate(jstart_canyon_W(inumbuild),jend_canyon_W(inumbuild)) !MAN7/6/2006
            !allocate(kend_canyon_W(inumbuild),kend_canyon_E(inumbuild)) !MAN7/6/2006
            !allocate(kend_canyon_N(inumbuild),kend_canyon_S(inumbuild)) !MAN7/6/2006
!            allocate(f_flag(inumbuild),w_flag(inumbuild),f_flagchk(inumbuild))
            !allocate(c_flag_E(inumbuild),c_flag_W(inumbuild),c_flag_N(inumbuild),c_flag_S(inumbuild))
            allocate(xuo(inumbuild),yuo(inumbuild))            !erp 1/31/2003
            allocate(xlo(inumbuild),ylo(inumbuild))
            allocate(xbo(inumbuild),ybo(inumbuild))
            allocate(Lf(inumbuild),Lr(inumbuild),Sx_east(inumbuild),Sy_north(inumbuild))
            allocate(Sx_west(inumbuild),Sy_south(inumbuild))
            allocate(Weff(inumbuild),Leff(inumbuild))
            allocate(Wt(inumbuild),Lt(inumbuild),wprime(inumbuild))
!            allocate(Lfx(inumbuild),Lfy(inumbuild))
!            allocate(Lfx1(inumbuild),Lfy1(inumbuild))  !NLB 02/10/04
!            allocate(Roofcx(nx,ny,nz))   !NLB 10/11/04 For Rooftop
            allocate(Rscale(inumbuild),Rcx(inumbuild))  !NLB 10/11/04 For Rooftop
            allocate(vo_roof(nx,ny,nz),uo_roof(nx,ny,nz)) !NLB 10/10/05
            allocate(phi(inumbuild),phiprime(inumbuild),theta(inumbuild)) !TMB 3/20/04
            allocate(build_uin(inumbuild),vmult(inumbuild),umult(inumbuild)) !TMB 4/13/04
         endif    !end 1st time through if
! added AAG 09/20/06  for multiple time steps, uo_roof was not atllocated  
        
! ERP 8/8/2005 Vegetative canopy TEST ONLY
         !ANU Plant Canopy 08/04/05
         !right now plantinit can only work for logarithmic inputs
!erp  if(inumcanopy.gt.0.and.site_blayer_flag(kk).eq.2)then


!end veg 


!TMB 3/20/04
!find theta and uin for each building by finding velocity components at the center of each building
lp001:   do ibuild=1,inumbuild
            do k=2,nz-1
               kstart(ibuild)=k
               if(zfo_actual(ibuild) .le. zm(k))exit
            enddo
            do k=kstart(ibuild),nz-1
               kend(ibuild)=k
               if(Ht(ibuild) .lt. zm(k+1))exit
            enddo
         enddo   lp001
lp002:   do i=1,inumbuild
     
!calculate the velocity at top center of building
!cu and cv are the horizontal velocity componenets at the top center of each
!building

! MAN 7/7/2005 var dz conversion
            cu = uo(nint((xfo(i)+Lti(i)/2.)/dx)+1,nint(yfo(i)/dy)+1,kend(i))
            cv = vo(nint((xfo(i)+Lti(i)/2.)/dx)+1,nint(yfo(i)/dy)+1,kend(i))
!end MAN 7/7/2005

!create uin for each building
            build_uin(i) = sqrt(cu**2+cv**2)

!prevent division by 0
            if (cu .eq. 0 .and. cv .gt. 0) theta(i) = 180
            if (cu .eq. 0 .and. cv .le. 0) theta(i) = 0
    
            if(cu .lt. 0 .and. cv .lt. 0) theta(i) = 90 - atan(cv/cu)*180/pi
            if(cu .lt. 0 .and. cv .ge. 0) theta(i) = 90 + atan(-cv/cu)*180/pi
            if(cu .gt. 0 .and. cv .ge. 0) theta(i) = 270 - atan(cv/cu)*180/pi
            if(cu .gt. 0 .and. cv .lt. 0) theta(i) = 270 + atan(-cv/cu)*180/pi


!calculating the umult, vmult for each building
            if(theta(i) .le. 90) then
               umult(i)=-sin((theta(i))*pi/180.)
               vmult(i)=-cos((theta(i))*pi/180.)
            endif
            if(theta(i) .gt. 90 .and. theta(i) .le.180) then
               umult(i)=-cos((theta(i)-90)*pi/180.)
               vmult(i)=sin((theta(i)-90)*pi/180.)
            endif
            if(theta(i).gt.180.and.theta(i).le.270) then
               umult(i)=sin((theta(i)-180)*pi/180.)
               vmult(i)=cos((theta(i)-180)*pi/180.)
            endif
            if(theta(i).gt.270.and.theta(i).le.360) then
               umult(i)=cos((theta(i)-270)*pi/180.)
               vmult(i)=-sin((theta(i)-270)*pi/180.)
            endif

            phi(i)=270.- theta(i)
            if(theta(i).gt.270) phi(i)=theta(i)-270.

            if(phi(i).le.90.)phiprime(i)=phi(i)*pi/180.
            if(phi(i).gt.90.and.phi(i).le.180)phiprime(i) =(180.-phi(i))*pi/180.
            if(phi(i).gt.180.and.phi(i).le.270)phiprime(i)=(phi(i)-180.)*pi/180.
            if(phi(i).gt.270.and.phi(i).le.360)phiprime(i)=(360.-phi(i))*pi/180.
         enddo   lp002      


!erp 1/30/2003
         num = 0     !erp
         denom = 0      !erp
!erp 1/30/2003

!ccccccccccccccccccccccccccccccccccccccccccc
! calculate the effective width and length based on incident wind angle.
! convert from meteorological wind angle theta to engineering
! wind angle phi. 

         do i=1,inumbuild
!            Lfx(i)=-999.
!            Lfy(i)=-999.
            Lf(i)=-999.
            Sx_east(i)=9999.
            Sy_north(i)=9999.
            Sx_west(i)=9999.
            Sy_south(i)=9999.  
            
! calculate building half widths 
! the width Wti is divided by a factor of 2 based on the Rockle 
! formula for multiple buildings
!erp 3/1/05 variable angle test
            Weff(i)=Lti(i)*sin(abs(phiprime(i)-gamma(i)))+Wti(i)*cos(abs(phiprime(i)-gamma(i)))
            Leff(i)=Wti(i)*sin(abs(phiprime(i)-gamma(i)))+Lti(i)*cos(abs(phiprime(i)-gamma(i)))
!  wprime(i)=Lti(i)*sin(phiprime(i))
            wprime(i)=Lti(i)*sin(abs(phiprime(i)-gamma(i)))
            Wt(i)=Wti(i)/2.
            Lt(i)=Lti(i)/2.
! calculate the building array displacement height and roughness length
!erp 1/30/2003

            num = num + Wti(i)*Lti(i)*Ht(i)  !erp
            denom = denom + Wti(i)*Lti(i)    !erp
         enddo

         disp_len = 0.8*num/denom      !erp
!         write(44,*)'D = ', disp_len, '  zo = ',zo

!erp 1/30/2003
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
! Generate buildings
! this now includes a building generation loop that allows
! for multiple buildings
! calculate building spacing s and set wake flags
! erp 3/9/05 Note that defbuild calls pentagon
         call defbuild  !erp 1/05/05
         
         
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  
! Generate Canopies         
         if(inumcanopy.gt.0.)then
!define the right size of arrays
!  allocate(cH(inumcanopy),cW(inumcanopy),cL(inumcanopy))
!  allocate(cnum(inumcanopy),ctype(inumcanopy))
!  allocate(cYfo(inumcanopy),cXfo(inumcanopy))
!  allocate(ca(inumcanopy))
!read the size of the canopy
!  read(55,*) !skip the description line for canopy size
!  do i=1,inumcanopy
!     read(55,*)cnum(i),ctype(i),cH(i),cW(i),cL(i),cXfo(i),
!    &      cYfo(i), ca(i)
!  enddo
!            if(i_time.eq.1)then
!               allocate(canopy_uin(inumcanopy),vmult_can(inumcanopy),umult_can(inumcanopy)) !ERP 8/12/05
!               allocate(canopy_uin_prof(inumcanopy,nz))  !erp
!            endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!calculate the velocity at top center of the vegetative canopy
!cu and cv are the horizontal velocity componenets at the top center of each
!canopy ERP 8/12/2005

!            canopy_uin_prof(1:inumcanopy,1:nz) = 0

!TMB 3/20/04
!find theta and uin for each building by finding velocity components at the center of each building

!lp001:      do i=1,inumcanopy
!     
!!calculate the velocity at top center of building
!!cu and cv are the horizontal velocity componenets at the top center of each
!!building
!
!
!               do j = nint(cH(i)/dz)+2,nz-2
!                  cu = uo(nint((cXfo(i)+cL(i)/2.)/dx)+1,nint(cYfo(i)/dy)+1,j)
!                  cv = vo(nint((cXfo(i)+cL(i)/2.)/dx)+1,nint(cYfo(i)/dy)+1,j)
!                  canopy_uin_prof(i,j) = sqrt(cu**2+cv**2)
!               enddo
!               cu = uo(nint((cXfo(i)+cL(i)/2.)/dx)+1,nint(cYfo(i)/dy)+1,nint(cH(i)/dz)+2)
!               cv = vo(nint((cXfo(i)+cL(i)/2.)/dx)+1,nint(cYfo(i)/dy)+1,nint(cH(i)/dz)+2)
!
!!create uin for each building
!               canopy_uin(i) = sqrt(cu**2+cv**2)
!
!!prevent division by 0
!               if (cu.eq.0.and.cv.gt.0) theta_can = 180
!               if (cu.eq.0.and.cv.le.0) theta_can = 0
!    
!               if(cu.lt.0.and.cv.lt.0) theta_can = 90 - atan(cv/cu)*180/pi
!               if(cu.lt.0.and.cv.ge.0) theta_can = 90 + atan(-cv/cu)*180/pi
!               if(cu.gt.0.and.cv.ge.0) theta_can = 270 - atan(cv/cu)*180/pi
!               if(cu.gt.0.and.cv.lt.0) theta_can = 270 + atan(-cv/cu)*180/pi
!
!
!!calculating the umult, vmult for each plant canopy
!               if(theta_can.le.90) then
!                  umult_can(i)=-sin((theta_can)*pi/180.)
!                  vmult_can(i)=-cos((theta_can)*pi/180.)
!               endif
!               if(theta_can.gt.90.and.theta_can.le.180) then
!                  umult_can(i)=-cos((theta_can-90)*pi/180.)
!                  vmult_can(i)=sin((theta_can-90)*pi/180.)
!               endif
!               if(theta_can.gt.180.and.theta_can.le.270) then
!                  umult_can(i)=sin((theta_can-180)*pi/180.)
!                  vmult_can(i)=cos((theta_can-180)*pi/180.)
!               endif
!               if(theta_can.gt.270.and.theta_can.le.360) then
!                  umult_can(i)=cos((theta_can-270)*pi/180.)
!                  vmult_can(i)=-sin((theta_can-270)*pi/180.)
!               endif
!            enddo   lp001        !end loop through vegetative canopies

! call plant canopy subroutine
            call plantinit
         endif
         !close(55)
         
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Call upwind - routine to generate the upwind cavity on a building
         do ibuild=1,inumbuild
            select case(bldtype(ibuild))
               case(1,4,10)
                  call upwind
            endselect
         enddo
         
         
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  
! wake section
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  
         if(wakeflag .ne. 0)then
            do ibuild=1,inumbuild
               if(wakeflag .eq. 9 .and. bldtype(ibuild) .ne. 3)then
                  call wake
               else
                  select case(bldtype(ibuild))
                     case(1,4,10)
                        if(zfo(ibuild).gt.0)then
                           call building_connect
                        endif
                        call rectanglewake
                     case(2,5)
                        if(zfo(ibuild).gt.0)then
                           call building_connect
                        endif
                        call cylinderwake
                     case(6)
                        call bridgewake
                  endselect
               endif
            enddo
         endif
         
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  
! street canyon section
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  

         if(streetcanyonflag .ne. 0)then
            call streetcanyon
         endif
         
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  
! rooftop or courtyard section
!ccccccccccccccccccccccccccccccccccccccccccccccccccc 
         do ibuild=1,inumbuild
            select case(bldtype(ibuild))
               case(1,2,10)
                  call rooftop
               case(4,5)
                  call courtyard
            endselect
         enddo
         
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  
! parking garage section
!ccccccccccccccccccccccccccccccccccccccccccccccccccc 
         if(i_time .eq. 1)then
! define the new number of building slabs that need to be reallocated in the building arrays
            num_new_builds = 0
            do ibuild=1,inumbuild
               if(bldtype(ibuild) .eq. 10)then
                  num_new_builds = num_new_builds + atten(ibuild) !remember atten(i) is number of stories/bld
               endif
            enddo
         endif
         if(num_new_builds .gt. 0)then
            do ibuild=1,inumbuild
               if(bldtype(ibuild) .eq. 10)then
                  call parking_garage
               endif
            enddo
            call build_garage
         endif
         
! end End Building parameterization section

!ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! MAN 4/12/2007 Moved computation Poisson boundary conditions
! (a,b,c,etc.) to new subroutine "wallbc"         
         call wallbc

!ccccccccccccccccccccccccccccccccccccccccccccccccccc  
! street intersection section
!ccccccccccccccccccccccccccccccccccccccccccccccccccc  

         if(streetcanyonflag .ne. 0 .and. intersectionflag .eq. 1)then 
            if(inumbuild.gt.0)then
               call street_intersect
               call poisson ! AG 04/06/2007 blends street intersection winds
               !print*,"Called street_intersect and poisson"
            endif
         endif
! MAN 4/12/2007 Moved Poisson boundary conditions to new subroutine "wallbc"

!ANU 01/04/2006vegetation parameterization
         if(inumcanopy.gt.0.)then
            do j=1,ny-1
               do i=1,nx-1
                  if(canopy_top(i,j) .gt. 0.)then
                     do k=2,canopy_ktop(i,j)
                        if(canopy_atten(i,j,k) .gt. 0.)then
                           avg_atten = canopy_atten(i,j,k)
                           if(canopy_atten(i,j,k+1) .ne. canopy_atten(i,j,k) &
                                 .or. canopy_atten(i,j,k-1) .ne. canopy_atten(i,j,k))then
                              num_atten=1.
                              if(canopy_atten(i,j,k+1) .gt. 0.)then
                                 avg_atten = avg_atten + canopy_atten(i,j,k+1)
                                 num_atten=num_atten+1.
                              endif
                              if(canopy_atten(i,j,k-1) .gt. 0.)then
                                 avg_atten = avg_atten + canopy_atten(i,j,k-1)
                                 num_atten=num_atten+1.
                              endif
                              avg_atten=avg_atten/num_atten
                           endif
                           vegvelfrac=log(canopy_top(i,j)/canopy_zo(i,j))*&
                                 exp(avg_atten*((zm(k)/canopy_top(i,j))-1.))/&
                                 log(zm(k)/canopy_zo(i,j))
                           if(vegvelfrac .lt. 1.)then
                              uo(i,j,k)=uo(i,j,k)*vegvelfrac
                              vo(i,j,k)=vo(i,j,k)*vegvelfrac
                              if(j .lt. ny-1)then
                                 if(canopy_atten(i,j+1,k) .eq. 0.)then
                                    vo(i,j+1,k)=vo(i,j+1,k)*vegvelfrac
                                 endif
                              endif
                              if(i .lt. nx-1)then
                                 if(canopy_atten(i+1,j,k) .eq. 0.)then
                                    uo(i+1,j,k)=uo(i+1,j,k)*vegvelfrac
                                 endif
                              endif
                           endif
                           if(icellflag(i,j,k) .gt. 0)then
                              icellflag(i,j,k)=8
                           endif
                        endif
                     enddo
                  endif
               enddo
            enddo
         endif
            
            

!ccccccccccccccccccccccccccccccccccccccccccccccccccc
!Celltype Coeficient Section
!ccccccccccccccccccccccccccccccccccccccccccccccccccc
! begin defining celltypes using the cellflags based on boundary cells
         do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
                  if(icellflag(i,j,k).eq.0)then ! MAN 7/8/2005 Celltype definition change
                     celltype(i,j,k)=0
! all cells thatcall divergence are buildings have a zero velocity within them
                     uo(i,j,k)  =0.
                     uo(i+1,j,k)=0.
                     vo(i,j,k)  =0.
                     vo(i,j+1,k)=0.
                     wo(i,j,k)  =0.
                     wo(i,j,k+1)=0.
                  endif
               enddo
            enddo
         enddo
! MAN 7/8/2005 Celltype definition change



! Tecplot format output
! commented out for qwicurb GUI
!        open(unit=29,file="celltype.dat",status="unknown")
! man 7/12/2005 changes to celltype
!         if(format_flag.eq.1 .or. format_flag.eq.3)then
!            if(i_time.eq.1)then
!               open(unit=33,file="QU_celltype.dat",status="unknown")
!            endif
!        write(29,*)'VARIABLES = "I", "J", "K","celltype"'
!        write(29,*)'ZONE I=', nx-1,',J=',ny-1,',K=',nz-1,',F=POINT'

!        write(33,*)'VARIABLES = "I", "J", "K","celltype"'
!        write(33,*)'ZONE I=', nx-1,',J=',ny-1,',K=',nz-1,',F=POINT'

!            do k=1,nz-1
!               do j=1,ny-1
!                  do i=1,nx-1
!                     write(33,71)(real(i)-.5)*dx,(real(j)-.5)*dy,zm(k),icellflag(i,j,k)
!                  enddo
!               enddo
!            enddo
!         endif
!erp 3/02/2005 lines added to write out unformatted for binary read into Matlab
!         if(format_flag.eq.2 .or. format_flag.eq.3)then  !erp 3/2/05
!            if(i_time.eq.1)then   
!                  open(unit=39,file="QU_celltype.bin",form='unformatted',status="unknown")
!            endif
!  allocate(x_write(nx-1,ny-1,nz-1),y_write(nx-1,ny-1,nz-1),z_write(nx-1,ny-1,nz-1))
!        do k=1,nz-1
!        do j=1,ny-1
!        do i=1,nx-1
!     x_write(i,j,k) = (real(i)-.5)*dx
!     y_write(i,j,k) = (real(j)-.5)*dy
!     z_write(i,j,k) = (real(k)-1.5)*dz
!        enddo
!        enddo
!        enddo 

!  write(39)(((x_write(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),
!     &  (((y_write(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),
!     &  (((z_write(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1),
!     &  (((icellflag(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1)
!            write(39)(((icellflag(i,j,k),i=1,nx-1),j=1,ny-1),k=1,nz-1)

!  deallocate(x_write,y_write,z_write)
!  close(39)
!         endif !end format_flag if erp 3/03/05



!  close(29)
!  close(33)
!  close(46)

! 71      format(3(1x,f8.3),i5)
!  erp multirun statements

! new print out removed from upwind and now located here because of the
! removal of the upwind cavity under certain conidtions
! erp 12/06/04
!         if(i_time .eq. 1)then 
!            write(46,*)inumbuild+inumcanopy,'! total number of buildings'
!            write(46,*)inumcanopy+inumbuildneg, '! total number of vegitative canopies'
!            write(46,*)num_new_builds, '! number of new buildings - parking garage'
!         endif

!         do ibuild=1,inumbuild
!            if(bldtype(ibuild) .ne. 0)then
!               write(44,*)'Theta = ',theta(ibuild),'  Phi = ',phi(ibuild),&
!                  '  phip = ',phiprime(ibuild)
               
               ! write to screenout.dat erp 1/15/2004
!               write(44,*)'Weff = ',Weff(ibuild)
!               write(44,*)'Leff = ',Leff(ibuild)
!               write(44,*)'Wti = ',Wti(ibuild)
!               write(44,*)'Lti = ',Lti(ibuild)
!               write(44,*)
               
!               write(44,*)'Lf =', Lf(ibuild)
!               write(44,*)'Lfy =', Lfy(ibuild)
!               write(44,*)'Lr =', Lr(ibuild)
! write to buildout.dat erp 1/15/2004
!               write(46,920)bldnum(ibuild)
!               write(46,925)bldtype(ibuild),gamma(ibuild)*180/pi
!MAN 8/30/2005 stacked building fix
!               write(46,926)Ht(ibuild)-zfo_actual(ibuild),Wti(ibuild),Lti(ibuild)
! MAN 9/13/2005 pentagon xfo fix
!               if(bldtype(ibuild).eq.3)then
!                  write(46,927)xfo(ibuild)+Wti(ibuild)/2.,yfo(ibuild),zfo_actual(ibuild)
!               else
!                  write(46,927)xfo(ibuild),yfo(ibuild),zfo_actual(ibuild)
!               endif
! end MAN 9/13/2005
! MAN 9/15/2005
!               if(bldtype(ibuild).eq.3)then
!                  write(46,921)Wti(ibuild),Lti(ibuild)
!               else
!                  write(46,921)Weff(ibuild),Leff(ibuild)
!               endif
! end MAN 9/15/2005
!               write(46,922)Lf(ibuild),Lr(ibuild),Atten(ibuild)
!               write(46,924)Sx_east(ibuild),Sy_north(ibuild)
!            endif
! 920        format('Building Number = ',i4)
! 921        format(' Weff = ',f9.4,' Leff = ',f9.4)
! 922        format('  Lf = ',f9.4,'  Lr = ',f9.4,' Att = ',f9.4)
! 924        format(' Sx = ',f9.4,' Sy = ',f9.4)
! 925        format(' Type = ',i4,' gamma = ',f9.4)
! 926        format(' Ht = ',f9.4,' W = ',f9.4,' L = ',f9.4)
! 927        format(' xfo = ',f9.4,' yfo = ',f9.4,' zfo = ',f9.4)
!         enddo
         
         if(num_new_builds .gt. 0)then
            call unbuild_garage
         endif
         
         do k=2,nz
            do j=1,ny
               do i=1,nx
                  if((uo(i,j,k) .ne. uo(i,j,k)) .or. (vo(i,j,k) .ne. vo(i,j,k)) .or. (wo(i,j,k) .ne. wo(i,j,k)))then
                     print*,'NaN found at ',i,j,k
                  endif
               enddo
            enddo
         enddo
!end change
         if(i_time.eq.num_time_steps)then

            deallocate(istart,iend)             !twh - added this line 01/08/03
            deallocate(jstart,jend)             !twh - added this line 01/08/03
            deallocate(kstart,kend)             !twh - added this line 01/08/03
            !deallocate(istart_canyon_N,iend_canyon_N) !MAN 7/5/2006
            !deallocate(jstart_canyon_E,jend_canyon_E) !MAN 7/5/2006
            !deallocate(istart_canyon_S,iend_canyon_S) !MAN 7/5/2006
            !deallocate(jstart_canyon_W,jend_canyon_W) !MAN 7/5/2006
            !deallocate(kend_canyon_W,kend_canyon_E) !MAN 7/5/2006
            !deallocate(kend_canyon_S,kend_canyon_N) !MAN 7/5/2006
            !deallocate(f_flag,w_flag,f_flagchk)  !twh - added this line 01/08/03 !MAN 7/5/2006
            !deallocate(c_flag_E,c_flag_W,c_flag_N,c_flag_S)      !twh - added this line 01/08/03 !MAN 7/5/2006
            deallocate(xuo,yuo)                 !twh - added this line 01/08/03
            deallocate(xlo,ylo)                 !twh - added this line 01/08/03
            deallocate(xbo,ybo)                 !twh - added this line 01/08/03
            deallocate(Lf,Lr,Sx_east,Sy_north,Sx_west,Sy_south)!MAN 7/5/2006
            deallocate(Weff,Leff)               !twh - added this line 01/08/03
            deallocate(Wt,Lt,wprime)            !twh - added this line 01/08/03
!            deallocate(Lfx,Lfy)                 !twh - added this line 01/08/03
!            deallocate(Lfx1,Lfy1)                 !NLB - added this line 02/10/04
            deallocate(Rscale,Rcx)              !NLB - added this line 10/11/04
!            deallocate(Roofcx)                  !NLB - added this line 10/11/04
            deallocate(celltype)                !twh - added this line 01/08/03
         endif

         return
      end
