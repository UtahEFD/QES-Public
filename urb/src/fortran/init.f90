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
      subroutine init
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! subroutine to initialize main array variables				
! reads in data from the input.dat file
! init.f90 subroutine is called by main.f90
! init.f90 calls zwake.f90 function			
! ERP 2001		
! Variable information:
! uo,vo,wo - are the initial velocity that are prescribed prior to mass 
!            conservation.
! u,v,w - final velocity field
! xfo,yfo - denote the x and y locations that represent the 
!           left side center of the building
! nx,ny, and nz are the number of cells in the x,y and z directions						
! theta - wind angle in standard meteorological format. 0 degrees is 	
! 	  out of the north, 90 degrees is a wind out of the east, etc		
! inumbuild - number of buildings in building array		
!
! Building type designations
!	bldtype = 1	regular building (rectangular parrelpiped)
!	bldtype = 2	cylindrical/elliptical
!	bldtype = 3 pentagon shape
!	bldtype	= 9 vegetation, requires an attenuation coefficient (see Cionco, 1965)
!
!
! * note that the velocity field is initialized at the end of the 	
!   subroutine.	
!erp 6/2/03	modifications to allow for data input velocity profiles. For
!		example wind direction can be varied as a function of height
!erp 6/2/03	modifications to allow for variable grid resolutions to
!		be entered via the ddx variable. ddx is specified in meters
!erp 6/5/03	added empirical parameterization subdomain. A subdomain box
!		may be defined in which the empirical parameterizations are 
!		applied. Outside of this domain conservation of mass is applied only
!		to an incoming boundary layer profile.
!erp 7/25/03	This version of qwicurb has the added array zfo(ibuild) 
!		which allows buildings of different sizes to be stacked on one another.	
!erp 8/14/03 This version of the code incorporates Tom Booth's upwind urban
!		boundary layer work. As such it calls the function zwake.
!erp 2/11/04 This version has been modified for Nilesh's upwind vortex
!erp 6/08/04 Modifications to canyon parameterization 
!erp 10/05/04	This version removes the meteorological input information
!		and puts it in the subroutine met_init.f90 to allow for multi-run
!		capability.	
!erp 6/30/05 This version adds variable dx,dy and dz capability based on the
!		work of Tau.
! 				
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
         use datamodule
         implicit none
!erp 2/17/04 NLB modifications

!	real Az,Bz
!	real logvel,Az,Bz
!erp 4/20/04 end changes logvel moved to datamodule

!	real zw, ustar, lc, d, vk, ac !TMB 7/10/03 canopy variables
!         real zwake
         integer count,qcfd_flag ! MAN 05-24-2007       

         pi=4.*atan(1.0)
	
! open and read input file input.dat
         open(unit=35,file='QU_simparams.inp',status='old')
         open(unit=36,file='QU_metparams.inp',status='old')
         open(unit=37,file='QU_buildings.inp',status='old')
         open(unit=47,file='QU_fileoptions.inp',status='old')
         open(unit=44,file='QU_screenout.dat',status='unknown')
         open(unit=46,file='QP_buildout.inp',status='unknown')
         !open(unit=48,file='QP_buildorder.inp',status='unknown')

!read file read writing option data from QU_options.dat
         read(47,*) ! QUIC version header line
         read(47,*)format_flag	!output format flag 1=ascii,2=binary,3=both
         read(47,*)uofield_flag  !write out uofield.dat 1=yes, 0=no
         read(47,*)uosensor_flag !write out flag for sensor velocities 1=yes, 0=no
         read(47,*)staggered_flag !write out flag for staggered velocities 1=yes, 0=no

! read input data from QU_domain.inp
         read(35,*) ! QUIC version header line
         read(35,*)nx		!nx defined in input file
         read(35,*)ny		!ny defined in input file
         read(35,*)nz      !nz defined in input file
         read(35,*)dx		!dx defined in input file
         read(35,*)dy		!dy defined in input file
         
!man 1/14/05 account for difference in grid cell definitions
         nx=nx+1
         ny=ny+1
         nz=nz+2
!end man 1/14/05 account for difference in grid cell definitions
! MAN 07/25/2008 stretched vertical grid
         allocate(z(nz),zm(nz),dz_array(nz))
         read(35,*)stretchgridflag   !Stretched grid flag (0= dz constant with z)
         z(:)=0.
         zm(:)=0.
         dz_array(:)=0.
         select case(stretchgridflag)
            case(0) !uniform
               read(35,*)dz		!dz defined in input file
               dz_array(:)=dz
            case(1) !custom
               read(35,*)
               do k=2,nz-1
                  read(35,*)dz_array(k)
               enddo
            case(2,3,4) !parabolic or exponential
               read(35,*)
               read(35,*)
               read(35,*)
               do k=2,nz-1
                  read(35,*)dz_array(k)
               enddo
         endselect
         dz_array(1)=dz_array(2)
         dz_array(nz)=dz_array(nz-1)
         zm(1)=-0.5*dz_array(1)
         z(1)=0.0
         do k=2,nz
            z(k)=z(k-1)+dz_array(k)
            zm(k)=z(k)-0.5*dz_array(k)
         enddo
         
         dz=minval(dz_array)
         
!ADL 7/1/2009 moved here from main.f90
! SOR coefficients
         alpha1=1.	!correlation of wind components term
         alpha2=1.		!correlation of wind components term
         eta=(alpha1/alpha2)**2	!ratio of gaussian precision moduli

         ! since omega is also used for an angle in bcsetup

         omegarelax=1.78		!acceleration term in SOR
         
         
!erp 6/30.05 coefficients for sor solver
         A=dx**2/dy**2
         B=eta*(dx**2/dz**2)
         
         
         
!MAN 09/02/2008 time inforation
!         read(35,*)start_time		!decimal start time
!         read(35,*)time_incr         !time increment
         read(35,*)num_time_steps    !total time increments
         allocate(time(num_time_steps))
         read(35,*) !day of year
         read(35,*) !UTC conversion
         read(35,*) !header line
         do i_time=1,num_time_steps
            read(35,*)time(i_time)
         enddo
! building parameterization flags
         read(35,*)roofflag	!rooftop recirc flag
         read(35,*)upwindflag	!upwind cavity flag
         read(35,*)streetcanyonflag    ! street canyon initialization method, added PKK 05/12/03
         read(35,*)intersectionflag    ! MAN 7/11/2006
         read(35,*)wakeflag ! MAN 06/29/2007 added wake flag to QU_simparams.inp
! MAN 7/10/2006 convergence criteria
         read(35,*)itermax	!max number of iterations
         read(35,*)residual_reduction		! MAN 09/26/2006 added residual reduction to input
! AAG 08/25/2006 turbulent diffusion parameters
         read(35,*)diffusion_flag	!turns on diffusion
         read(35,*)diffstep		!diffusion iterations
! MAN 02/05/2007 Geo-referencing parameters
         read(35,*)domain_rotation
         read(35,*)utmx
         read(35,*)utmy
         read(35,*)utmzone
! MAN 05-24-2007 QUIC-CFD flag turns off the SOR solver        
         read(35,*)qcfd_flag
         if(qcfd_flag .gt. 0)then
            itermax=0
            diffusion_flag=0
         endif
         
!input from QU_buildings.inp
         read(37,*) ! QUIC version header line
         read(37,*)x_subdomain_start	! Subdomain coordinates
         read(37,*)y_subdomain_start	!
         read(37,*)x_subdomain_end	!
         read(37,*)y_subdomain_end	!
         read(37,*)zo	! MAN 8-19-2005 Updated input output file structures
! MAN 7/27/2005 var dz subdomain changed into meters
         x_subdomain_start = x_subdomain_start*dx
         y_subdomain_start = y_subdomain_start*dy
         x_subdomain_end = x_subdomain_end*dx
         y_subdomain_end = y_subdomain_end*dy
!end MAN 7/27/2005

         read(37,*)inumbuild	!number of buildings
         read(37,*)

         allocate(uo(nx,ny,nz),vo(nx,ny,nz),wo(nx,ny,nz))
! need to move to met_init.f90
         allocate(uo_bl(nz),vo_bl(nz)) !erp 6/08/04

         allocate(u(nx,ny,nz),v(nx,ny,nz),w(nx,ny,nz))
         allocate(p1(nx-1,ny-1,nz-1),p2(nx-1,ny-1,nz-1),r(nx-1,ny-1,nz-1))
         !MAN 8/30/2005 stacked building fix
         allocate(zfo_actual(inumbuild))
         allocate(xfo(inumbuild),yfo(inumbuild),zfo(inumbuild))	!erp 7/25/03
         allocate(gamma(inumbuild))	!erp 7/26/03
         allocate(aa(inumbuild),bb(inumbuild))					!erp 1/31/2003
         allocate(icellflag(nx-1,ny-1,nz-1),ibldflag(nx-1,ny-1,nz-1))
         allocate(e(nx-1,ny-1,nz-1),f(nx-1,ny-1,nz-1),g(nx-1,ny-1,nz-1))
         allocate(m(nx-1,ny-1,nz-1),n(nx-1,ny-1,nz-1),o(nx-1,ny-1,nz-1))
         allocate(h(nx-1,ny-1,nz-1),p(nx-1,ny-1,nz-1),q(nx-1,ny-1,nz-1))
         allocate(denoms(nx-1,ny-1,nz-1))
         allocate(Ht(inumbuild),Wti(inumbuild),Lti(inumbuild))
         allocate(bldnum(inumbuild),bldtype(inumbuild),group_id(inumbuild))
         allocate(atten(inumbuild))	!erp 1/3/2006
		   allocate(rooftop_flag(inumbuild)) ! AAG 09/13/06 
         rooftop_flag(:)=0 ! AAG 09/13/06  intialized rooftop flag to 1
         if(diffusion_flag .gt. 0)allocate(Fxd(nx,ny,nz),Fyd(nx,ny,nz),Fzd(nx,ny,nz),visc(nx,ny,nz))
! Read in the building number, type, height, width, length ,xfo,yfo,zfo,gamma and atten
! atten - attenuation coefficient for vegetation
!erp 1/31/2003
! note that for now if the building is cylindrical, enter Lti = 0.

         do i=1,inumbuild
            read(37,*)bldnum(i),group_id(i),bldtype(i),Ht(i),Wti(i),Lti(i),xfo(i),yfo(i),zfo(i),gamma(i),atten(i)
            ! MAN 07/25/2008 stretched vertical grid building dimensions already in meters
            Ht(i)=Ht(i)+zfo(i)
!            Ht(i)=(Ht(i)+zfo(i))*dz	!convert back to real world unit, TZ 10/29/04
!            Wti(i)=Wti(i)*dy	!convert back to real world unit, TZ 10/29/04
!            Lti(i)=Lti(i)*dx	!convert back to real world unit, TZ 10/29/04
!            xfo(i)=xfo(i)*dx	!convert back to real world unit, TZ 10/29/04
!            yfo(i)=yfo(i)*dy	!convert back to real world unit, TZ 10/29/04
!            zfo(i)=zfo(i)*dz	!convert back to real world unit, TZ 10/29/04
            gamma(i)=gamma(i)*pi/180.	!erp 7/25/03
!            Ht(i)=Ht(i)+zfo(i)			!erp 7/25/03
!erp	    zfo(i)=zfo(i)+2				!erp 7/23/03
            if(bldtype(i) .eq. 2 .or. bldtype(i) .eq. 5)then		!if the building is a cylinder/ellipse
               bb(i)=Wti(i)/2.			!set minor axis to input Width
               aa(i)=Lti(i)/2.			!set major axis to input Lenth
            endif
!erp 10/22/04 Pentagon addition
            if(bldtype(i).eq.3)then		!if the building is a Pentagon
               bb(i)=Wti(i)/2.				!Radius Pentagon is inscribed in
               xfo(i)=xfo(i)-bb(i)
            endif
         enddo

	
!erp 1/3/2006 check to see if building is actually vegetation
         inumcanopy = 0
         inumbuildneg = 0
         do i=1,inumbuild
            if(bldtype(i).eq.9)	then
               inumcanopy=inumcanopy+1			!total number of vegative canopies
            endif
            if(bldtype(i).eq.0)	then
               inumbuildneg=inumbuildneg+1			!total number of negative buildings
            endif
         enddo


!if vegetation exist rename variables to be consistent with plantinit.f90 rountine	
         if(inumcanopy.gt.0)then
            allocate(cH(inumcanopy),cW(inumcanopy),cL(inumcanopy))
            allocate(cnum(inumcanopy),ctype(inumcanopy),cgroup(inumcanopy))
            allocate(cYfo(inumcanopy),cXfo(inumcanopy),cZfo(inumcanopy))
            allocate(ca(inumcanopy),cgamma(inumcanopy))
            allocate(canopy_ktop(nx-1,ny-1),canopy_top(nx-1,ny-1),canopy_atten(nx-1,ny-1,nz-1))
            allocate(canopy_zo(nx-1,ny-1),canopy_ustar(nx-1,ny-1),canopy_d(nx-1,ny-1))
            count = 0
            do i=1,inumbuild

               if(bldtype(i).eq.9)	then
                  count=count+1			!total number of vegative canopies

                  cnum(count)= bldnum(i)
                  cgroup(count)= group_id(i)		!new not used
                  ctype(count)= bldtype(i)
                  cH(count)= Ht(i)
                  cW(count)= Wti(i)
                  cL(count)= Lti(i)
                  cXfo(count)=xfo(i)
                  cYfo(count)=yfo(i)
                  cZfo(count)=zfo(i)		              
                  cgamma(count)=gamma(i)
                  ca(count)= atten(i)			!attenuation coeffecient
               endif

            enddo

         endif



!end 1/3/2006

!erp 10/2004 move to met_init.f
!erp 1/31/2003
! convert from real world units to grid units
!
!	zref=zref/ddx
!	uin=uin/ddx
!	!modify zo in grid terms
!	if(blayer_flag.eq.2.or.blayer_flag.eq.4)pp=pp/ddx	
! end erp 10/2004 move to met_init.f

!calculate domain Length width and height erp 1/30/2003
         Lx=(nx-1)*dx
         Ly=(ny-1)*dy
         ! MAN 07/25/2008 stretched vertical grid
         Lz=z(nz-1)
!calculate domain Length width and height erp 1/30/2003

! erp 1/30/2003
! printout read in data to the screen
!erp	write(44,*)'Lx = ',Lx,'Ly  ',Ly,'Lz = ',Lz
!         write(44,*)'Lx = ',Lx,'Ly = ',Ly,'Lz = ',Lz	! '=' added to Ly, TZ 10/29/04

!         write(44,*)'subdomain southwest corner x coordinate',x_subdomain_start
!         write(44,*)'subdomain southwest corner y coordinate',y_subdomain_start
!         write(44,*)'subdomain northeast corner x coordinate',x_subdomain_end
!         write(44,*)'subdomain northeast corner y coordinate',y_subdomain_end
!         write(44,*)'dx = ',dx
!         write(44,*)'dy = ',dy
!         write(44,*)'dz = ',dz
!erp 10/2004 move to met_init.f
!	write(44,*)'Incident wind angle = ',theta
!	write(44,*)'Upstream WS = ',uin
!erp 10/2004 move to met_init.f
!         write(44,*)

!         do i=1,inumbuild
!            write(44,*)'building #  ',bldnum(i),'  building type  ',bldtype(i)
!            write(44,100)

!            write(44,200)Ht(i),Wti(i),Lti(i),xfo(i),yfo(i),zfo(i),gamma(i),atten(i)!erp 7/25/03
!            write(44,*)
!         enddo
! 100     format(1x,'Height',3x,'Width',3x,'Length',3x,'xfo',3x,'yfo',3x,'zfo',3x,'gamma',3x,'AttenCoef')
! 200     format(8(f6.2,2x))

!       initialize p1 and p2 - they are Lagrange multipliers
!TMB using vectors instead of DO loops 
 
         p1(1:nx-1,1:ny-1,1:nz-1)=0.1
         p2(1:nx-1,1:ny-1,1:nz-1)=0.
         r(1:nx-1,1:ny-1,1:nz-1)=0.


! erp 1/17/03 initilize all arrays at zero to start	
         uo(1:nx,1:ny,1:nz)=0.
         vo(1:nx,1:ny,1:nz)=0.
         wo(1:nx,1:ny,1:nz)=0.
	
         close(35)
         close(37)
         close(47)
         return
      end
