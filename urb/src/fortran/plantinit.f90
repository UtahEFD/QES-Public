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
      subroutine plantinit

         !this function initializes the velocity profile in the specified
         !vegetative area area using the MacDonald (2000) approach.
         !ca - attenuation coefficient
         !vk - von Karmen constant

         !TMB 7/18/03 internal plant canopy variables
         !ANU 6/2005 implemented

         use datamodule
         implicit none
   
         real bisect  !bisection function
         real x1,x2,x3,x4,y1,y2,y3,y4
         real xmin,xmax,ymin,ymax,x_c,y_c
         real xL1,xL2,yL3,yL4,slope
!         real d_check
         integer i_start,i_end,j_start,j_end,ican
         real vegvelfrac,avg_atten,num_atten
         
         canopy_top(:,:)=0.
         canopy_ktop(:,:)=0
         canopy_ustar(:,:)=0.
         canopy_zo(:,:)=0.
         canopy_d(:,:)=0.
         canopy_atten(:,:,:)=0.
         
         vk=.4 !VonKarmen Constant
!erp  add subroutine to calculate ustar and zo  using a least squares
! regression of the current initialization
         
         
         do ican=1,inumcanopy !for each canopy
            if(cgamma(ican) .eq. 0)then
               i_start=nint(cXfo(ican)/dx)
               i_end=nint((cXfo(ican)+cL(ican))/dx)
               j_start=nint((cYfo(ican)-0.5*cW(ican))/dy)
               j_end=nint((cYfo(ican)+0.5*cW(ican))/dy)
               do j=j_start+1,j_end
                  do i=i_start+1,i_end
                     if(i .gt. 0 .and. i .lt. nx .and. j .gt. 0 .and. j .lt. ny)then
                        if(cH(ican) .gt. canopy_top(i,j))then
                           canopy_top(i,j)=cH(ican)
                        endif
                        do k=2,nz-1
                           if(cZfo(ican) .lt. zm(k) .and. cH(ican) .gt. zm(k))then
                              canopy_atten(i,j,k)=ca(ican)
                           endif
                           if(cH(ican) .le. zm(k))exit
                        enddo    
                     endif
                  enddo
               enddo
            else
               x1=cxfo(ican)+0.5*cW(ican)*sin(cgamma(ican))
               y1=cyfo(ican)-0.5*cW(ican)*cos(cgamma(ican))
               x2=x1+cL(ican)*cos(cgamma(ican))
               y2=y1+cL(ican)*sin(cgamma(ican))
               x4=cxfo(ican)-0.5*cW(ican)*sin(cgamma(ican))
               y4=cyfo(ican)+0.5*cW(ican)*cos(cgamma(ican))
               x3=x4+cL(ican)*cos(cgamma(ican))
               y3=y4+cL(ican)*sin(cgamma(ican))
               if(cgamma(ican).gt.0)then
                  xmin=x4
                  xmax=x2
                  ymin=y1
                  ymax=y3
               endif
               if(cgamma(ican).lt.0)then
                  xmin=x1
                  xmax=x3
                  ymin=y2
                  ymax=y4
               endif
               do j=nint(ymin/dy)+1,nint(ymax/dy)+1
                  do i=nint(xmin/dx)+1,nint(xmax/dx)+1
                     if(i .gt. 0 .and. i .lt. nx .and. j .gt. 0 .and. j .lt. ny)then
                        x_c=(real(i)-0.5)*dx
                        y_c=(real(j)-0.5)*dy
                        slope = (y4-y1)/(x4-x1) !slope of L1
                        xL1 = x4 + (y_c-y4)/slope
                        slope = (y3-y2)/(x3-x2) !slope of L2
                        xL2 = x3 + (y_c-y3)/slope
                        slope = (y2-y1)/(x2-x1) !slope of L3
                        yL3 = y1 + slope*(x_c-x1)
                        slope = (y3-y4)/(x3-x4) !slope of L4
                        yL4 = y4 + slope*(x_c-x4)
                        if(x_c .gt. xL1 .and. x_c .lt. xL2 .and. y_c .gt. yL3 .and. y_c .lt. yL4)then
                           if(cH(ican) .gt. canopy_top(i,j))then
                              canopy_top(i,j)=cH(ican)
                           endif
                           do k=2,nz-1
                              if(cZfo(ican) .lt. zm(k) .and. cH(ican) .gt. zm(k))then
                                 canopy_atten(i,j,k)=ca(ican)
                              endif
                              if(cH(ican) .le. zm(k))exit
                           enddo
                        endif
                     endif
                  enddo
               enddo
            endif
         enddo
         
         call regress
         
         do j=1,ny-1
            do i=1,nx-1
               if(canopy_top(i,j) .gt. 0.)then
                  canopy_d(i,j) = bisect(canopy_ustar(i,j),canopy_zo(i,j), &
                        canopy_top(i,j),canopy_atten(i,j,canopy_ktop(i,j)),vk,0.)
                  if(canopy_d(i,j) .gt. 0.99*canopy_top(i,j))then
                     canopy_d(i,j)=0.7*canopy_top(i,j)
                     canopy_zo(i,j)=0.25*canopy_top(i,j)
                  endif
                  uH = (canopy_ustar(i,j)/vk)*log((canopy_top(i,j)-canopy_d(i,j))/canopy_zo(i,j))
                  do k=2,nz
                     if(zm(k) .le. canopy_top(i,j))then
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
                           vegvelfrac=log((canopy_top(i,j)-canopy_d(i,j))/canopy_zo(i,j))*&
                                 exp(avg_atten*((zm(k)/canopy_top(i,j))-1.))/&
                                 log(zm(k)/canopy_zo(i,j))
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
                           if(icellflag(i,j,k) .gt. 0)then
                              icellflag(i,j,k)=8
                           endif
                        endif
                     else
                        vegvelfrac=log((zm(k)-canopy_d(i,j))/canopy_zo(i,j))/&
                                 log(zm(k)/canopy_zo(i,j))
                        uo(i,j,k)=uo(i,j,k)*vegvelfrac
                        vo(i,j,k)=vo(i,j,k)*vegvelfrac
                        if(j .lt. ny-1)then
                           if(canopy_atten(i,j+1,canopy_ktop(i,j)) .eq. 0.)then
                              vo(i,j+1,k)=vo(i,j+1,k)*vegvelfrac
                           endif
                        endif
                        if(i .lt. nx-1)then
                           if(canopy_atten(i+1,j,canopy_ktop(i,j)) .eq. 0.)then
                              uo(i+1,j,k)=uo(i+1,j,k)*vegvelfrac
                           endif
                        endif
                     endif
                  enddo
               endif
            enddo
         enddo
         write(44,*)

         do ibuild=1,inumbuild
            write(44,*)'building #  ',bldnum(ibuild),'  building type  ',bldtype(ibuild)
            write(44,100)

            write(44,200)Ht(ibuild),Wti(ibuild),Lti(ibuild),&
               xfo(ibuild),yfo(ibuild),zfo(ibuild),gamma(ibuild),atten(ibuild)!erp 7/25/03
            write(44,*)
         enddo
 100     format(1x,'Height',3x,'Width',3x,'Length',3x,'xfo',3x,'yfo',3x,'zfo',3x,'gamma',3x,'AttenCoef')
 200     format(8(f6.2,2x))

         return
      end
