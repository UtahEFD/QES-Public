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
      subroutine sensorinit
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Subroutine to initialize the initial velocity field (uo,vo,wo) for each time step
!
! This program interpolates irregularly spaced data onto a 
! uniform grid using the Barnes Objective Map Analysis  
! scheme as implemented in Koch et al., 1983
! 
! This program has been modified from Eric Pardyjak's original 2D code to work with 
! quicurb 3D		
!
! this subroutine uses a quasi-3D barnes objective mapping technique 
! quasi-3D is just using sensor height (zref) and sensor wind speed (uref)
! to extrapolate a vertical velocity profile at each sensor location 
! to get a velocity at every height at each location
! from these hieght varying velocities, a regular 2D barnes objective map analysis
! is done at each planar height throughout the whole domain.
!
! Called by met_init.f90
!
! 
!
! Tom Booth 2/17/04
!
! Variable information:
!	site_xcoord,site_ycoord,site_zcoord - the coordinates of each site (meters)
!	t_site - is the time step for each site
!	dir_site - is the direction of the wind for each site at each time step
!	vel_site - is the magnitude of the wind for each site at each time step
!	total_time_step - is the total number of time steps in a 24 hr period
!	num_sites - is the total number of measurement sites
!	sgamma - numerical convergence parameter
!	lamda - weight parameter (ko)
!	deln - average Radius (i.e. computed data spacing)
! TMB/ERP 9/20/05 
!	Added a logarithmic interpolation below the the lowest input data point for
!	input files that read in wind profile data

!	
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule
         implicit none

         integer kk !,site_num
         integer iwork, jwork
         integer met_input_flag


         !real site_time_incr
         real site_zref, theta_site, mag_site, umult_site, vmult_site

         real yc, xc, rc, rcsum, rcval
         real sumw, sumwv, sumwu
         real sgamma, lamda, deln
         real dxx, dyy, u12, u34, v12, v34
! MAN 04/05/2007 Time varying profile parameters
         real bisect
         real xtemp,psi_m !erp 9/18/2006 stability variables
         
         real, allocatable :: x(:,:), y(:,:)	  !Locations of grid cell centers in meters

         allocate(x(nx,ny),y(nx,ny))
         !read in whole file on first quasi-time step
         if (i_time.eq.1) then
            vk=0.4 !Von Karman's constant
            read(36,*) ! QUIC version header line
            read(36,*)met_input_flag
! MAN 4/17/2007 moved met data read statements to separate subroutines and added architecture for new 3D vectorfield input
            select case(met_input_flag)
               case(0)
                  call read_quic_met
               case(1)
                  call read_ittmm5_met
               case(2)
                  call read_hotmac_met
            endselect
         endif !if i_time = 1
         !---------------------------------------------------------------------------------
         !This do loop defines each vertical velocity profile at each sensor site so the
         !Barnes Mapping interpolation scheme has a velocity at each site for each height
lp003:   do kk=1,num_sites
            if(site_blayer_flag(kk,i_time).eq.4)then  !data entry profile
               allocate(u_data(site_nz_data(kk,i_time)),v_data(site_nz_data(kk,i_time)))
               do ii=1,site_nz_data(kk,i_time)
                  ! convert wind speed wind direction into u and v
                  if(site_wd_data(kk,i_time,ii).le.90) then
                     u_data(ii)=-site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii))*pi/180.)
                     v_data(ii)=-site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii))*pi/180.)
                  endif
                  if(site_wd_data(kk,i_time,ii).gt.90.and.site_wd_data(kk,i_time,ii).le.180) then
                     u_data(ii)=-site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii)-90)*pi/180.)
                     v_data(ii)= site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii)-90)*pi/180.)
                  endif
                  if(site_wd_data(kk,i_time,ii).gt.180.and.site_wd_data(kk,i_time,ii).le.270) then
                     u_data(ii)= site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii)-180)*pi/180.)
                     v_data(ii)= site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii)-180)*pi/180.)
                  endif
                  if(site_wd_data(kk,i_time,ii).gt.270.and.site_wd_data(kk,i_time,ii).le.360) then
                     u_data(ii)= site_ws_data(kk,i_time,ii)*cos((site_wd_data(kk,i_time,ii)-270)*pi/180.)
                     v_data(ii)=-site_ws_data(kk,i_time,ii)*sin((site_wd_data(kk,i_time,ii)-270)*pi/180.)
                  endif
               enddo
            endif
lp002:      do k=2,nz		!loop through vertical cell indices
               if(site_blayer_flag(kk,i_time) .lt. 4)then
                  theta_site = site_wd_data(kk,i_time,1)
                  mag_site	= site_ws_data(kk,i_time,1)
                  site_zref = site_z_data(kk,i_time,1)
                  if(theta_site.le.90) then
                     umult_site=-sin((theta_site)*pi/180.)
                     vmult_site=-cos((theta_site)*pi/180.)
                  endif
                  if(theta_site.gt.90.and.theta_site.le.180) then
                     umult_site=-cos((theta_site-90)*pi/180.)
                     vmult_site=sin((theta_site-90)*pi/180.)
                  endif
                  if(theta_site.gt.180.and.theta_site.le.270) then
                     umult_site=sin((theta_site-180)*pi/180.)
                     vmult_site=cos((theta_site-180)*pi/180.)
                  endif
                  if(theta_site.gt.270.and.theta_site.le.360) then
                     umult_site=cos((theta_site-270)*pi/180.)
                     vmult_site=-sin((theta_site-270)*pi/180.)
                  endif
!power law profile
                  if(site_blayer_flag(kk,i_time) .eq. 2)then
                     ! MAN 07/25/2008 stretched vertical grid
                     u_prof(kk,k)=umult_site*mag_site*((zm(k)/site_zref)**site_pp(kk,i_time))
                     v_prof(kk,k)=vmult_site*mag_site*((zm(k)/site_zref)**site_pp(kk,i_time))
                  endif !erp 2/6/2003 end power law profile
!logrithmic velocity profile
                  if(site_blayer_flag(kk,i_time) .eq. 1)then
! MAN 05/15/2007 adjust for stability
                     if(k.eq.2)then
                        if(site_zref*site_rL(kk,i_time) .ge. 0)then
                           psi_m=4.7*site_zref*site_rL(kk,i_time)
                        else
                           xtemp=(1.-15*site_zref*site_rL(kk,i_time))**(0.25)
                           psi_m=-2*log(0.5*(1+xtemp))-log(0.5*(1+xtemp**2.))+2*atan(xtemp)-0.5*pi
                        endif
                        ustar=mag_site*vk/(log(site_zref/site_pp(kk,i_time))+psi_m)
                     endif
! end MAN 05/15/2007        
                     ! MAN 07/25/2008 stretched vertical grid             
                     if(zm(k)*site_rL(kk,i_time) .ge. 0)then
                        psi_m=4.7*zm(k)*site_rL(kk,i_time)
                     else
                        xtemp=(1.-15*zm(k)*site_rL(kk,i_time))**(0.25)
                        psi_m=-2*log(0.5*(1+xtemp))-log(0.5*(1+xtemp**2.))+2*atan(xtemp)-0.5*pi
                     endif
                     u_prof(kk,k)=(umult_site*ustar/vk)*(log(zm(k)/site_pp(kk,i_time))+psi_m)
                     v_prof(kk,k)=(vmult_site*ustar/vk)*(log(zm(k)/site_pp(kk,i_time))+psi_m)
                  endif !erp 2/6/2003 end log law profile
!Canopy profile
                  if(site_blayer_flag(kk,i_time) .eq. 3)then
                     if(k.eq.2)then !only calculate d once
! MAN 05/15/2007 adjust for stability                        
                        if(site_zref*site_rL(kk,i_time) .ge. 0)then
                           psi_m=4.7*site_zref*site_rL(kk,i_time)
                        else
                           xtemp=(1.-15*site_zref*site_rL(kk,i_time))**(0.25)
                           psi_m=-2*log(0.5*(1+xtemp))-log(0.5*(1+xtemp**2.))+2*atan(xtemp)-0.5*pi
                        endif
                        ustar = mag_site*vk/(log(site_zref/site_pp(kk,i_time))+psi_m)                        
                        d = bisect(ustar,site_pp(kk,i_time),site_H(kk,i_time),site_ac(kk,i_time),vk,psi_m)
! end MAN 05/15/2007
                        if(site_H(kk,i_time)*site_rL(kk,i_time) .ge. 0)then
                           psi_m=4.7*(site_H(kk,i_time)-d)*site_rL(kk,i_time)
                        else
                           xtemp=(1.-15*(site_H(kk,i_time)-d)*site_rL(kk,i_time))**(0.25)
                           psi_m=-2*log(0.5*(1+xtemp))-log(0.5*(1+xtemp**2.))+2*atan(xtemp)-0.5*pi
                        endif
                        uH = (ustar/vk)*(log((site_H(kk,i_time)-d)/site_pp(kk,i_time))+psi_m);
                        if(site_zref .le. site_H(kk,i_time))then
                           mag_site=mag_site/(uH*exp(a*(site_zref/site_H(kk,i_time) -1.)))
                        else
                           if(site_zref*site_rL(kk,i_time) .ge. 0)then
                              psi_m=4.7*(site_zref-d)*site_rL(kk,i_time)
                           else
                              xtemp=(1.-15*(site_zref-d)*site_rL(kk,i_time))**(0.25)
                              psi_m=-2*log(0.5*(1+xtemp))-log(0.5*(1+xtemp**2.))+2*atan(xtemp)-0.5*pi
                           endif
                           mag_site=mag_site/((ustar/vk)*(log((site_zref-d)/zo)+psi_m))
                        endif
                        ustar=mag_site*ustar
                        uH=mag_site*uH
                     endif
                     ! MAN 07/25/2008 stretched vertical grid
                     if(zm(k) .le. site_H(kk,i_time))then !lower canopy profile
                        u_prof(kk,k) = umult_site * uH*exp(site_ac(kk,i_time)&
                                       *(zm(k)/site_H(kk,i_time) -1))
                        v_prof(kk,k) = vmult_site * uH*exp(site_ac(kk,i_time)&
                                       *(zm(k)/site_H(kk,i_time) -1))
                     endif
                     if(zm(k) .gt. site_H(kk,i_time))then !upper canopy profile
                        if(zm(k)*site_rL(kk,i_time) .ge. 0)then
                           psi_m=4.7*(zm(k)-d)*site_rL(kk,i_time)
                        else
                           xtemp=(1.-15*(zm(k)-d)*site_rL(kk,i_time))**(0.25)
                           psi_m=-2*log(0.5*(1+xtemp))-log(0.5*(1+xtemp**2.))+2*atan(xtemp)-0.5*pi
                        endif
                        u_prof(kk,k)=(umult_site*ustar/vk)*&
                                  (log((zm(k)-d)/site_pp(kk,i_time))+psi_m)
                                  
                        v_prof(kk,k)=(vmult_site*ustar/vk)*&
                                  (log((zm(k)-d)/site_pp(kk,i_time))+psi_m)
                     endif !end urban canopy TMB 6/16/03
                  endif
               endif
!new 2/7/2005 velocity profile entry
               if(site_blayer_flag(kk,i_time) .eq. 4)then  !data entry profile
                  do ii=1,site_nz_data(kk,i_time) !loop through the data points in input profile each time step
! begin interpolation input velocity to computational grid
! MAN 07/25/2008 stretched vertical grid
                     if(zm(k) .eq. site_z_data(kk,i_time,ii))then
                        u_prof(kk,k)=u_data(ii)
                        v_prof(kk,k)=v_data(ii)
                        goto 500
                     endif
!erp 9/23/05	logarithmically interpolate to zero velocity at zo below lowest data point
!MAN 01/21/07 logarithmic interpolation uses the first data point instead of the second
                     if(zm(k) .gt. 0 .and. zm(k) .lt. site_z_data(kk,i_time,1))then
                        if(zm(k) .gt. site_pp(kk,i_time))then
                           u_prof(kk,k)= (u_data(1)/(log(site_z_data(kk,i_time,1)/site_pp(kk,i_time))))*   &
                                                   log(zm(k)/site_pp(kk,i_time))
                           v_prof(kk,k)= (v_data(1)/(log(site_z_data(kk,i_time,1)/site_pp(kk,i_time))))*   &
                                                   log(zm(k)/site_pp(kk,i_time))
                        else
                           u_prof(kk,k)= 0
                           v_prof(kk,k)= 0
                        endif
                        goto 500
                     endif
!erp 9/23/05
                     if(ii .gt. 1)then
                        ! MAN 07/25/2008 stretched vertical grid
                        if(zm(k) .gt. site_z_data(kk,i_time,ii-1) .and. zm(k).lt.site_z_data(kk,i_time,ii))then
                           u_prof(kk,k)=((u_data(ii)-u_data(ii-1))/(site_z_data(kk,i_time,ii)-site_z_data(kk,i_time,ii-1)))   &
                                           *(zm(k)-site_z_data(kk,i_time,ii-1)) + u_data(ii-1)
                           v_prof(kk,k)=((v_data(ii)-v_data(ii-1))/(site_z_data(kk,i_time,ii)-site_z_data(kk,i_time,ii-1)))   &
                                           *(zm(k)-site_z_data(kk,i_time,ii-1)) + v_data(ii-1)
                           goto 500
                        endif
                     endif
                  enddo !end ii loop
 500              continue
                  
                  ! extrapolate logarithmically for data beyond input velocity
                  ! MAN 07/25/2008 stretched vertical grid
                  if(zm(k) .gt. site_z_data(kk,i_time,site_nz_data(kk,i_time))) then
                     u_prof(kk,k)=(log(zm(k)/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1))/   &
                        		log(site_z_data(kk,i_time,site_nz_data(kk,i_time))/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1)))*   &
                        		(u_data(site_nz_data(kk,i_time))-   &
                        		u_data(site_nz_data(kk,i_time)-1)) + u_data(site_nz_data(kk,i_time)-1)
                     v_prof(kk,k)=(log(zm(k)/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1))/   &
                        		log(site_z_data(kk,i_time,site_nz_data(kk,i_time))/site_z_data(kk,i_time,site_nz_data(kk,i_time)-1)))*   &
                        		(v_data(site_nz_data(kk,i_time))-   &
                        		v_data(site_nz_data(kk,i_time)-1)) + v_data(site_nz_data(kk,i_time)-1)
                  endif
                  if(k .eq. nz) deallocate(u_data,v_data)
               endif !erp 2/6/2003 end data entry
            enddo   lp002       !k=nz
         enddo   lp003       !num_sites kk
! end MAN 04/05/2007
         !---------------------------------------------------------------------------------
         ! find average distance of each measuring station relative to all other
         ! measuring stations
         
         if(num_sites .eq. 1)then
            do k=2,nz
               uo(:,:,k)=u_prof(1,k)
               vo(:,:,k)=v_prof(1,k)
            enddo
         else !Barnes Mapping Scheme
            rcsum=0.
            do kk=1,num_sites
               rcval=1000000. !ignore anything over 1000 kilometers
               do k=1,num_sites
                  xc=site_xcoord(k)-site_xcoord(kk)
                  yc=site_ycoord(k)-site_ycoord(kk)
                  rc=sqrt(xc**2+yc**2)
                  if(rc .lt. rcval .and. k .ne. kk) rcval=rc !shortest distance
               enddo
               rcsum=rcval+rcsum !sum of shortest distances
            enddo
            deln=rcsum/real(num_sites)  !average Radius (i.e. computed data spacing)
            lamda=5.052*(2.*deln/pi)**2 !weight parameter
            !lamda=ffact*lamda			!distance dependant factor
            !numerical convergence parameter
            sgamma = 0.2			!gamma=.2 max detail, gamma=1 min detail
! MAN 7/7/2005 var dz conversion
! calculate grid cell center locations in meters  
            do j=1,ny
               do i=1,nx
                  x(i,j)=real(i-1)*dx-.5*dx !calculating in meters  ddx=(meters/grid)
                  y(i,j)=real(j-1)*dy-.5*dy
               enddo
            enddo
! end MAN 7/7/2005

            !------------------------------------------------------------------------------
            !first and second barnes pass done for each cell level in z direction
            
            !compute weight of each site on point (i,j)
            
            do j=1,ny
               do i=1,nx
                  do kk=1,num_sites
                     wm(kk,i,j)=exp(-1/lamda*(site_xcoord(kk)-x(i,j))**2   &
                           	 -1/lamda*(site_ycoord(kk)-y(i,j))**2)
                     wms(kk,i,j)=exp(-1/(sgamma*lamda)*(site_xcoord(kk)-x(i,j))**2   &
                           			-1/(sgamma*lamda)*(site_ycoord(kk)-y(i,j))**2)
                  enddo
                  if(sum(wm(:,i,j)) .eq. 0.)then
                     wm(:,i,j)=1e-20
                  endif
               enddo
            enddo
lp004:      do k=2,nz
               ! interpolate onto the grid
               ! do first Barnes pass
               do j=1,ny
                  do i=1,nx
                     sumwu=sum(wm(:,i,j)*u_prof(:,k))
                     sumwv=sum(wm(:,i,j)*v_prof(:,k))
                     sumw=sum(wm(:,i,j))
                     uo(i,j,k)=sumwu/sumw
                     vo(i,j,k)=sumwv/sumw
                  enddo !i=nx
               enddo !j=ny
! before doing the 2nd pass for the Barnes Method  	
! use a 4-point bilinear interpolation  		
! scheme to get estimated values at measured point (+)  
! using the 1st pass calculated data at grid points (*) 
!  							
!     *       *				1        2		
!							
!          +   					  +	     !definition of points
!    							
!     *       *            3        4
!     |    |  |		
!     | dxx|  |  
!     |       |  !definition of measurements
!     |  ddx  |
!			
               do kk=1,num_sites
                  do j=1,ny
                     do i=1,nx
                        !find closest grid location on left side of site
                        if(x(i,j).lt.site_xcoord(kk)) iwork=i
                        !find closest grid location on lower side of site
                        if(y(i,j).lt.site_ycoord(kk)) jwork=j
                     enddo !i=nx
                  enddo !j=ny
                  !distance to site point from lower and left sides
                  dxx=site_xcoord(kk)-x(iwork,jwork)
                  dyy=site_ycoord(kk)-y(iwork,jwork)
! MAN 7/7/2005 fixed interpolation of velocities and var dz conversion	
                  !upper u interpolated velocity
                  u12 = (1-dxx/dx)*uo(iwork,jwork+1,k)+(dxx/dx)*uo(iwork+1,jwork+1,k)
                  !lower u interplotaed velocity
                  u34 = (1-dxx/dx)*uo(iwork,jwork,k)+(dxx/dx)*uo(iwork+1,jwork,k)
                  !total interpolated u velocity
                  uoint(kk)=(dyy/dy)*u12+(1-dyy/dy)*u34
                  
                  !upper v interpolated velocity
                  v12 = (1-dxx/dx)*vo(iwork,jwork+1,k)+(dxx/dx)*vo(iwork+1,jwork+1,k)
                  !lower v interplotaed velocity
                  v34 = (1-dxx/dx)*vo(iwork,jwork,k)+(dxx/dx)*vo(iwork+1,jwork,k)
                  !total interpolated u velocity
                  voint(kk)=(dyy/dy)*v12+(1-dyy/dy)*v34
! end MAN 7/7/2005 
               enddo !kk=num_sites
! end bilinear interpolation section
! Begin 2nd Barnes pass
               do j=1,ny
                  do i=1,nx
                     sumwu=sum(wms(:,i,j)*(u_prof(:,k)-uoint(:)))
                     sumwv=sum(wms(:,i,j)*(v_prof(:,k)-voint(:)))
                     sumw=sum(wms(:,i,j))
                     if(sumw.eq.0)then
                        uo(i,j,k)=uo(i,j,k)
                        vo(i,j,k)=vo(i,j,k)
                     else
                        uo(i,j,k)=uo(i,j,k) + sumwu/sumw
                        vo(i,j,k)=vo(i,j,k) + sumwv/sumw
                     endif
                  enddo
               enddo
! determine the norm between the measurement and interpolation
!      do kk=1,num_sites

!	 do j=1,ny
!       do i=1,nx
!
!       if(x(i,j).lt.site_xcoord(kk)) iwork=i
!       
!       if(y(i,j).lt.site_ycoord(kk)) jwork=j
!       
!       enddo
!       enddo

!      dxx=site_xcoord(kk)-x(iwork,jwork)
!       dyy=site_ycoord(kk)-y(iwork,jwork)

!       u12 = (ddx-dxx)*uo(iwork,jwork+1,k)+dxx*uo(iwork+1,jwork+1,k)
!       u34 = (ddx-ddx)*uo(iwork,jwork,k)+dxx*uo(iwork+1,jwork,k)
!       uoint(kk)=dyy*u12+(1-dyy)*u34

!       v12 = (ddx-dxx)*vo(iwork,jwork+1,k)+dxx*vo(iwork+1,jwork+1,k)
!       v34 = (ddx-dxx)*vo(iwork,jwork,k)+dxx*vo(iwork+1,jwork,k)
!       voint(kk)=dyy*v12+(1-dyy)*v34

            !      unorm=unorm+abs(u_prof(kk,k)-uoint(kk))
!       vnorm=vnorm+abs(v_prof(kk,k)-voint(kk))
!       enddo

            !        unorm=unorm/real(num_sites)
            !        vnorm=vnorm/real(num_sites)
!         write(*,*)
!         write(*,*)'interpolated u-velocity absolute norm: ',unorm
!         write(*,*)'interpolated v-velocity absolute norm: ',vnorm
!         write(*,*)

            enddo   lp004       !k=1:nz
         endif
         
!	open(unit=38,file='uofield1.dat',status='unknown')
	

         !if this is the last time loop then deallocate this subroutines variables
         !deallocate arrays
         !deallocate(u_prof,v_prof)
         deallocate(x,y)
         !deallocate(uoint,voint)
         !deallocate(wm)

         !----------------------------------------------------------------------
         !     Writing the uosensorfield.dat file

!	if(i_time.eq.1)then

!	 open(unit=51,file="uosensorfield.dat",status="unknown")
!      write(51,*)'% Inital sensor velocity field x,y,z,uo,vo,wo'

!	endif

!912	format(i9,  '       !Time Increment')
!913	format(i9,  '       !Number of time steps')
!914	format(f9.4,'       !Time')
!916	format(3i6,1x,3(f17.5,1x))
!101   format(6(f11.5,1x))

!	if(i_time.eq.1)then
!       write(51,913)num_time_steps
!	endif

!	write(51,*)'%Start next time step'
!	write(51,912)i_time
!	write(51,914)time

!	  do k=1,nz-1
!        do j=1,ny-1
!        do i=1,nx-1

         !actual coordinate of center of cells
!	   x1=(0.5*(real(i+1)+real(i))-1.)*ddx
!         y1=(0.5*(real(j+1)+real(j))-1.)*ddx
!         z1=(0.5*(real(k+1)+real(k))-2.)*ddx

         !        write(51,101)x1,y1,z1,uo(i,j,k),vo(i,j,k),wo(i,j,k)
!      enddo
!       enddo
!       enddo

!	if(i_time.eq.num_time_steps) close(51)
         !------------------------------------------------------------------------------

! Determine the max_velmag
         max_velmag=0.
         do i=1,nx
            do j=1,ny
               max_velmag=max(max_velmag,sqrt((uo(i,j,nz-1)**2.)+(vo(i,j,nz-1)**2.)))
            enddo
         enddo


         !initialize vector fields TMB 7/11/03
         u(1:nx,1:ny,1:nz)=0.
         v(1:nx,1:ny,1:nz)=0.
         w(1:nx,1:ny,1:nz)=0.
! erp initialize upwind array erp 6/18/04
         uo_bl(1:nz)=uo(1,1,1:nz)
         vo_bl(1:nz)=vo(1,1,1:nz)

!         write(44,*)'upstream winds:'
!         write(44,*)'uo',uo(5,5,10)
!         write(44,*)'vo',vo(5,5,10)
         return
      end
