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
!
!
! Subroutine to determine the location of a street intersection
! erp 1/2006
! NOTE: Need to make modification to handle multiple runs 1/24/2006

      subroutine street_intersect


         use datamodule
         implicit none

         integer changeflag,intersect_flag,istart_intflag,jstart_intflag,NS_flag !,EW_flag
         integer int_istart,int_jstart,int_istop,int_jstop,int_kstart,int_kstop
!         real ubot,utop,uleft,uright,uceil,ufloor
!         real vbot,vtop,vleft,vright,vceil,vfloor
!         real wbot,wtop,wleft,wright,wceil,wfloor
!         real Cx,Cy,Cz
	
         real, allocatable :: u_x(:,:,:),u_y(:,:,:),u_z(:,:,:)
         real, allocatable :: v_x(:,:,:),v_y(:,:,:),v_z(:,:,:)
         real, allocatable :: w_x(:,:,:),w_y(:,:,:),w_z(:,:,:)

         integer, allocatable :: intersect(:,:,:),intersect_1(:,:,:)
         integer, allocatable :: intersect_2(:,:,:),intersect_1opp(:,:,:)
         integer, allocatable :: intersect_2opp(:,:,:) 
		   integer, allocatable :: E_W_flag(:,:,:),W_E_flag(:,:,:),N_S_flag(:,:,:),S_N_flag(:,:,:)  !SUP
         !open(10011,file='E_W_flag.txt',status='unknown')
		   !open(10022,file='W_E_flag.txt',status='unknown')   SUP writes text files
		   !open(10033,file='S_N_flag.txt',status='unknown')
		   !open(10044,file='N_S_flag.txt',status='unknown')
		   !open(10055,file='intersect.txt',status='unknown')
		   !open(10066,file='icellflag.txt',status='unknown')
		   !open(10077,file='intersect_1.txt',status='unknown')


         allocate(intersect(nx-1,ny-1,nz-1),intersect_1(nx-1,ny-1,nz-1),intersect_2(nx-1,ny-1,nz-1))
         allocate(intersect_1opp(nx-1,ny-1,nz-1),intersect_2opp(nx-1,ny-1,nz-1))
	      allocate(E_W_flag(nx-1,ny-1,nz-1),W_E_flag(nx-1,ny-1,nz-1),N_S_flag(nx-1,ny-1,nz-1),S_N_flag(nx-1,ny-1,nz-1)) !SUP
         allocate(u_x(nx,ny,nz),u_y(nx,ny,nz),u_z(nx,ny,nz))
         allocate(v_x(nx,ny,nz),v_y(nx,ny,nz),v_z(nx,ny,nz))
         allocate(w_x(nx,ny,nz),w_y(nx,ny,nz),w_z(nx,ny,nz))

!	if(i_time.eq.1)then
!	open(unit=74,file='QU_intersect.dat',status='unknown')
!	open(unit=75,file='QU_icellflagtest.dat',status='unknown')
!	endif

         intersect_flag=0

! make sure that all cells that are buildings have a zero velocity within them
         do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
                  if(icellflag(i,j,k).eq.0.)then ! MAN 7/8/2005 celltype definition change
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

         intersect(1:nx-1,1:ny-1,1:nz-1) = 0
         intersect_1(1:nx-1,1:ny-1,1:nz-1) = 0 !x-direction flag
         intersect_2(1:nx-1,1:ny-1,1:nz-1) = 0 !y-direction flag
         intersect_1opp(1:nx-1,1:ny-1,1:nz-1)=0
		   intersect_2opp(1:nx-1,1:ny-1,1:nz-1)=0
		   E_W_flag=0
		   W_E_flag=0
		   N_S_flag=0
		   S_N_flag=0
	
         changeflag = 0


! sweep through (x) to find intersections	
         do k=1,nz-1
            do j=1,ny-1
!SUP sweep through +x
               do i=2,nx-1
!determine where the street interesection begins
                  if(icellflag(i-1,j,k).eq.6.and.icellflag(i,j,k).ne.6 .and.icellflag(i,j,k).ne.0)then
                     changeflag=1
                     istart_intflag = i
                  endif
!determine where the street intersection ends
                  if(changeflag.eq.1 .and. icellflag(i,j,k).eq.6 .or.	&	!run into another street canyon   
                        changeflag.eq.1 .and. icellflag(i,j,k).eq.0 .or.&       !run into another building 
						      changeflag.eq.1 .and. icellflag(i,j,k).eq.1)then		!run into free atm.
                     changeflag=0
                  endif

                  intersect_1(i,j,k) = changeflag

               enddo
!if we get to the end of a row and changeflag = 1, then no SI exists reset those 
               if(changeflag.eq.1)then
                  intersect_1(istart_intflag:nx-1,j,k) = 0
               endif

               changeflag = 0	!reset flag


!SUP sweep through -x
              do i=nx-2,2,-1
			     !print*,i,j,k,icellflag(i,j,k)
				 !print*,i,j,k,icellflag(i+1,j,k),'+1'
!determine where the street interesection begins
                  if(icellflag(i+1,j,k).eq.6.and.icellflag(i,j,k).ne.6 .and.icellflag(i,j,k).ne.0)then
                     changeflag=1
                     istart_intflag = i
                  endif
!determine where the street intersection ends
                  if(changeflag.eq.1 .and. icellflag(i,j,k).eq.6 .or.	&	!run into another street canyon   
                        changeflag.eq.1 .and. icellflag(i,j,k).eq.0.or. &           !run into another building
						changeflag.eq.1 .and. icellflag(i,j,k).eq.1 )then		!run into free atm.
                     changeflag=0
                  endif

                  intersect_1opp(i,j,k) = changeflag

               enddo
!if we get to the end of a row and changeflag = 1, then no SI exists reset those 
               if(changeflag.eq.1)then
                  intersect_1opp(nx-1:istart_intflag:-1,j,k) = 0
               endif

               changeflag = 0	!reset flag



            enddo
         enddo


! now sweep in the j direction
         changeflag = 0
	
         do k=1,nz-1
            do i=1,nx-1
               do j=2,ny-1

                  if(icellflag(i,j-1,k).eq.6.and.icellflag(i,j,k).ne.6.and.icellflag(i,j,k).ne.0)then
                     changeflag=1
                     jstart_intflag = j
                  endif
!determine where the street intersection ends
                  if(changeflag.eq.1 .and. icellflag(i,j,k).eq.6 .or.	&	!run into another street canyon   
                        changeflag.eq.1 .and. icellflag(i,j,k).eq.0.or. &           !run into another building
						changeflag.eq.1 .and. icellflag(i,j,k).eq.1)then		!run into free atm.
                     changeflag=0
                  endif

                  intersect_2(i,j,k) = changeflag

               enddo
!if we get to the end of a row and changeflag = 1, then no SI exists reset those 
               if(changeflag.eq.1)then
                  intersect_2(i,jstart_intflag:ny-1,k) = 0 !SUP changed intersect_1 to _2
               endif
               changeflag = 0

!SUP sweep through -y
			   do j=ny-2,2,-1

                  if(icellflag(i,j+1,k).eq.6.and.icellflag(i,j,k).ne.6.and.icellflag(i,j,k).ne.0)then
                     changeflag=1
                     jstart_intflag = j
                  endif
!determine where the street intersection ends
                  if(changeflag.eq.1 .and. icellflag(i,j,k).eq.6 .or.	&	!run into another street canyon   
                        changeflag.eq.1 .and. icellflag(i,j,k).eq.0.or.  &          !run into another building   
						changeflag.eq.1 .and. icellflag(i,j,k).eq.1 )then		!run into free atm.
                     changeflag=0
                  endif

                  intersect_2opp(i,j,k) = changeflag

               enddo
!if we get to the end of a row and changeflag = 1, then no SI exists reset those 
               if(changeflag.eq.1)then
                  intersect_2opp(i,ny-1:jstart_intflag:-1,k) = 0
               endif
               changeflag = 0
            enddo
         enddo


                 do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
!                  if(intersect_1(i,j,k).eq.1 .and. intersect_2(i,j,k).eq.1)icellflag(i,j,k)=9 SUP
                   if((intersect_1(i,j,k).eq.1 .or.intersect_1opp(i,j,k).eq.1) & 
				     .and. (intersect_2(i,j,k).eq.1.or.intersect_2opp(i,j,k).eq.1))intersect(i,j,k)=1

				   !	if(k.eq.4)then
!	write(74,123)i,j,k,intersect_1(i,j,k),intersect_2(i,j,k),icellflag(i,j,k)
!	write(75,123)i,j,k,icellflag(i,j,k)
!	endif
               enddo
            enddo
         enddo
!write(10077,'(4i6)'),(((i,j,k,intersect_1(i,j,k),i=1,nx-1),j=1,ny-1),k=1,1)

!SUP looking to make sure that there are street canyons on 2 or more adjacent sides
         do k=1,nz-1
            do j=2,ny-1
			      NS_flag=0
               do i=2,nx-1
                  if(intersect(i,j,k).eq.1 .and. icellflag(i-1,j,k).eq.6) NS_flag=1
				      if(intersect(i,j,k).ne.1 .and. NS_flag.eq.1) NS_flag=0
                  if(NS_flag.eq.1) E_W_flag(i,j,k)=1
               enddo
               NS_flag=0
			      do i=nx-2,2,-1
                  if(intersect(i,j,k).eq.1 .and. icellflag(i+1,j,k).eq.6) NS_flag=1
				      if(intersect(i,j,k).ne.1 .and. NS_flag.eq.1) NS_flag=0
                  if(NS_flag.eq.1) W_E_flag(i,j,k)=1
			      enddo
            enddo
         enddo
!123	format(6i6)
!SUP


!SUP
         do k=1,nz-1
            do i=2,nx-1
			      NS_flag=0
               do j=2,ny-1
                  if(intersect(i,j,k).eq.1 .and. icellflag(i,j-1,k).eq.6) NS_flag=1
				      if(intersect(i,j,k).ne.1 .and. NS_flag.eq.1) NS_flag=0
                  if(NS_flag.eq.1) S_N_flag(i,j,k)=1
               enddo
               NS_flag=0
			      do j=ny-1,2,-1
                  if(intersect(i,j,k).eq.1 .and. icellflag(i,j+1,k).eq.6) NS_flag=1
				      if(intersect(i,j,k).ne.1 .and. NS_flag.eq.1) NS_flag=0
                  if(NS_flag.eq.1) N_S_flag(i,j,k)=1
			      enddo
            enddo
         enddo
!123	format(6i6)
!SUP
!write(10011,'(4i6)'),(((i,j,k,E_W_flag(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)
!write(10022,'(4i6)'),(((i,j,k,W_E_flag(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)
!write(10033,'(4i6)'),(((i,j,k,N_S_flag(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)
!write(10044,'(4i6)'),(((i,j,k,S_N_flag(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)
!write(10055,'(4i6)'),(((i,j,k,intersect(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)


!SUP
         do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
!                  if(intersect_1(i,j,k).eq.1 .and. intersect_2(i,j,k).eq.1)icellflag(i,j,k)=9 SUP
									!print*,""
									!print*,"EW = ", E_W_flag(i,j,k)
									!print*,"WE = ", W_E_flag(i,j,k)
									!print*,"SN = ", S_N_flag(i,j,k)
									!print*,"NS = ", N_S_flag(i,j,k)
                  if((E_W_flag(i,j,k).eq.1 .or. W_E_flag(i,j,k).eq.1).and.&
				         (S_N_flag(i,j,k).eq.1 .or. N_S_flag(i,j,k).eq.1))icellflag(i,j,k)=9

				   !	if(k.eq.4)then
!	write(74,123)i,j,k,intersect_1(i,j,k),intersect_2(i,j,k),icellflag(i,j,k)
!	write(75,123)i,j,k,icellflag(i,j,k)
!	endif
               enddo
            enddo
         enddo
!SUP

!write(10066,'(4i6)'),(((i,j,k,icellflag(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)
!write(10077,'(4i6)'),(((i,j,k,intersect_1(i,j,k),i=1,nx-1),j=1,ny-1),k=2,2)
 

!test
!find intersection limits
         do k=1,nz-1
            do j=1,ny-1
               do i=1,nx-1
                  u_x(i,j,k)=0	!initialize
                  u_y(i,j,k)=0	!initialize
                  u_z(i,j,k)=0	!initialize
                  v_x(i,j,k)=0	!initialize
                  v_y(i,j,k)=0	!initialize
                  v_z(i,j,k)=0	!initialize
                  w_x(i,j,k)=0	!initialize
                  w_y(i,j,k)=0	!initialize
                  w_z(i,j,k)=0	!initialize
	               if(icellflag(i,j,k).eq.9)then
                     int_istart=i
                     int_jstart=j
                     int_kstart=k
                     goto 130
                  endif
               enddo
            enddo
         enddo
 130     continue
         do k=nz-1,1,-1
            do j=ny-1,1,-1
               do i=nx-1,1,-1
                  if(icellflag(i,j,k).eq.9)then
                     int_istop=i+1
                     int_jstop=j+1
                     int_kstop=k+1
                     intersect_flag=1
                     goto 131
                  endif
               enddo
            enddo
         enddo
 131     continue

! MAN 12/11/2007 Removed unnecessary loops from intersection algorithm since interpolation is being performed in poisson

!!!ERP 6/12/2006 please NOTE THAT THE FOLLOWING CODE IS UNDER DEVELOPMENT 
!! trilinearly interpolate, need to add + 0.5 where necessary and variable dx,dy,dz
!! Weighted average Coefs, can be made to be a function of space
!         if(intersect_flag.eq.1)then	!are there any intersection nodes?
!
!            Cx = 1.	!weighting coefficent in x-direction
!            Cy = 1.	!weighting coefficent in y-direction
!            Cz = 0.	!weighting coefficent in z-direction
!
!!	print*,int_istart,int_jstart,int_kstart
!!	print*,int_istop,int_jstop,int_kstop
!
!            int_istart = int_istart - 3
!            int_jstart = int_jstart - 3
!
!            do k=int_kstop,int_kstart,-1
!               do j=int_jstart+1,int_jstop-1
!                  do i=int_istart+1,int_istop-1
!!		  if(icellflag(i,j,k).eq.9)print*, 'icell flag = 9'	!actually interpolate
!                     if(icellflag(i,j,k).eq.9)then	!actually interpolate
!                        u_x(i,j,k)=uo(int_istart,j,k)+((uo(int_istop,j,k)-uo(int_istart,j,k))/(int_istop-int_istart))*(i-int_istart)
!                        u_y(i,j,k)=uo(i,int_jstart,k)+((uo(i,int_jstop,k)-uo(i,int_jstart,k))/(int_jstop-int_jstart))*(j-int_jstart)
!!		u_z(i,j,k)=uo(i,j,int_kstart)+((uo(i,j,int_kstop)-uo(i,j,int_kstart))/(int_kstop-int_kstart))*(k-int_kstart)
!                        u_z(i,j,k)=0 + ((uo(i,j,int_kstop)-uo(i,j,int_kstart))/(int_kstop-int_kstart))*(k-int_kstart)
!                        uo(i,j,k)=Cx*u_x(i,j,k)!+Cy*u_y(i,j,k)+Cz*u_z(i,j,k)
!                        if(abs(uo(i,j,k)) .gt. max_velmag)then
!                           print*,'Parameterized U exceeds max in intersection',&
!                              uo(i,j,k),max_velmag,i,j,k
!                        endif
!!		uo(i,j,k)= 0
!                     endif
!                  enddo
!               enddo
!            enddo
!
!            do k=int_kstop,int_kstart,-1
!               do j=int_jstart+2,int_jstop-1
!                  do i=int_istart+1,int_istop-1
!                     if(icellflag(i,j,k).eq.9)then	!actually interpolate
!                        v_x(i,j,k)=vo(int_istart,j,k)+((vo(int_istop,j,k)&
!                                   -vo(int_istart,j,k))/(int_istop-int_istart))*(i-int_istart)
!                        v_y(i,j,k)=vo(i,int_jstart+1,k)+((vo(i,int_jstop,k)&
!                                   -vo(i,int_jstart+1,k))/(int_jstop-int_jstart+1))*(j-int_jstart+1)
!                        v_z(i,j,k)= 0 +((vo(i,j,int_kstop)-vo(i,j,int_kstart))/(int_kstop-int_kstart))*(k-int_kstart)
!!	    v_z(i,j,k)=vo(i,j,int_kstart)+((vo(i,j,int_kstop)-vo(i,j,int_kstart))/(int_kstop-int_kstart))*(k-int_kstart)
!                        vo(i,j,k)=Cy*v_y(i,j,k)
!                        if(abs(vo(i,j,k)) .gt. max_velmag)then
!                           print*,'Parameterized V exceeds max in intersection',&
!                              vo(i,j,k),max_velmag,i,j,k
!                        endif
!!	    vo(i,j,k)=Cx*v_x(i,j,k)+Cy*v_y(i,j,k)+Cz*v_z(i,j,k)
!!			vo(i,j,k)=0
!!	print132,i,j,k,v_x(i,j,k),v_y(i,j,k),v_z(i,j,k),uo(i,j,k)
!
!!	print132,k,(int_kstop-int_kstart),(k-int_kstart),(uo(i,j,int_kstop)-uo(i,j,int_kstart)),uo(i,j,int_kstart),u_z(i,j,k)
!!132	format(3i5,4(f9.5,1x))
!                     endif
!                  enddo
!               enddo
!            enddo
!
!            do k=int_kstop,int_kstart,-1
!               do j=int_jstart+1,int_jstop-1
!                  do i=int_istart+1,int_istop-1
!                     if(icellflag(i,j,k).eq.9)then	!actually interpolate
!                        w_x(i,j,k)=wo(int_istart,j,k)+((wo(int_istop,j,k)&
!                                   -wo(int_istart,j,k))/(int_istop-int_istart))*(i-int_istart)
!                        w_y(i,j,k)=wo(i,int_jstart,k)+((wo(i,int_jstop,k)&
!                                   -wo(i,int_jstart,k))/(int_jstop-int_jstart))*(j-int_jstart)
!                        w_z(i,j,k)=wo(i,j,int_kstart)+((wo(i,j,int_kstart+1)&
!                                   -wo(i,j,int_kstart))/(int_kstop-int_kstart))*(k-int_kstart)
!                        wo(i,j,k)=Cx*w_x(i,j,k)+Cy*w_y(i,j,k)+Cz*w_z(i,j,k)
!                        wo(i,j,k)=0
!                     endif
!                  enddo
!               enddo
!            enddo
!
!         endif !are there any intersection nodes at all if?
!!end test
!
!!SUP over writing interpolated winds with inflow winds
!         do k=1,nz-1
!            do j=1,ny-1
!               do i=1,nx-1
!                  if(icellflag(i,j,k).eq.9) then
!				         uo(i,j,k)=uo_bl(k)
!				         vo(i,j,k)=vo_bl(k)
!				      endif
!               enddo
!            enddo
!         enddo

! end MAN 12/11/2007
         deallocate(intersect_1,intersect_2,intersect)
		   deallocate(E_W_flag,W_E_flag,N_S_flag,S_N_flag)
         deallocate(u_x,u_y,u_z,v_x,v_y,v_z,w_x,w_y,w_z)
!	close(74)
!	close(75)
         return
      end
