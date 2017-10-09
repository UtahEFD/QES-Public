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
      subroutine sort
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
! Subroutine sort - this subroutine sorts the buildings and reorders
! them in terms of height from smallest to tallest so that the 
! empirical algorithms in bcsetup.f90 apply them in that order
!	- called by main.f90
!	- calls none
! ERP 3/8/05
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

         use datamodule ! make data from module "datamodule" visible

         implicit none

         integer, allocatable :: imax(:),bldnum_orig(:),bldtype_orig(:),group_id_orig(:)
         real, allocatable :: Ht_orig(:),Wti_orig(:),Lti_orig(:),Htmax(:),aa_orig(:),bb_orig(:)
         real, allocatable :: xfo_orig(:),yfo_orig(:),zfo_orig(:),gamma_orig(:),atten_orig(:)
         allocate(Ht_orig(inumbuild),Wti_orig(inumbuild),Lti_orig(inumbuild))
         allocate(bldnum_orig(inumbuild),bldtype_orig(inumbuild))
         allocate(xfo_orig(inumbuild),yfo_orig(inumbuild),zfo_orig(inumbuild))
         allocate(gamma_orig(inumbuild),Htmax(inumbuild),aa_orig(inumbuild),bb_orig(inumbuild))
         allocate(imax(inumbuild),group_id_orig(inumbuild),atten_orig(inumbuild))

!define temporary arrays
         bldnum_orig=bldnum
         group_id_orig=group_id ! Add group ID
         bldtype_orig = bldtype
         Ht_orig=Ht
         Wti_orig=Wti
         Lti_orig=Lti
         xfo_orig=xfo
         yfo_orig=yfo
         zfo_orig=zfo
         gamma_orig=gamma
         atten_orig=atten
         aa_orig=aa
         bb_orig=bb



! initialize arrays
         do i=1,inumbuild
            Htmax(i) = 0.
            imax(i) = i

!erp 1/6/2006 remove zero height buildings that were from vegetation
            if(bldtype(i).eq.9)Ht(i)=9999		!a very large number
            if(bldtype(i).eq.0)Ht(i)=9998
         enddo

         do j=1,inumbuild
            do i=1,inumbuild
               if(Ht(i) .gt. Htmax(j))then
                  Htmax(j)=Ht(i)
                  imax(j)=i
               endif
            enddo
            Ht(imax(j))=-999
         enddo

         do i=1,inumbuild
            !write(48,*)bldnum_orig(imax(inumbuild+1-i))
            bldnum(i)=bldnum_orig(imax(inumbuild+1-i))
            bldtype(i)=bldtype_orig(imax(inumbuild+1-i))
            group_id(i)=group_id_orig(imax(inumbuild+1-i))
            Ht(i)=Ht_orig(imax(inumbuild+1-i))
            Wti(i)=Wti_orig(imax(inumbuild+1-i))
            Lti(i)=Lti_orig(imax(inumbuild+1-i))
            xfo(i)=xfo_orig(imax(inumbuild+1-i))
            yfo(i)=yfo_orig(imax(inumbuild+1-i))
            zfo(i)=zfo_orig(imax(inumbuild+1-i))
            gamma(i)=gamma_orig(imax(inumbuild+1-i))
            atten(i)=atten_orig(imax(inumbuild+1-i))
            aa(i)=aa_orig(imax(inumbuild+1-i))
            bb(i)=bb_orig(imax(inumbuild+1-i))
         enddo

!	do i=1,inumbuild
!	print*,bldnum_orig(i),bldnum(i),Ht_orig(i),Ht(i)
!	enddo
!MAN 8/30/2005 stacked building fix
         zfo_actual=zfo

         deallocate(Ht_orig,Wti_orig,Lti_orig,bldnum_orig,bldtype_orig,group_id_orig)
         deallocate(xfo_orig,yfo_orig,zfo_orig,gamma_orig,Htmax,aa_orig,bb_orig)

	

         inumbuild=inumbuild-inumcanopy !erp 1/3/2006 vegetative canopy mod


!	do i=1,inumbuild
!	write(44,*)'building #  ',bldnum(i),'  building type  ',bldtype(i)
!	write(44,100)
!
!	write(44,200)Ht(i),Wti(i),Lti(i),xfo(i),yfo(i),zfo(i),gamma(i),atten(i)!erp 7/25/03
!	write(44,*)
!	enddo
!100	format(1x,'Height',3x,'Width',3x,'Length',3x,'xfo',3x,'yfo',3x,'zfo',3x,'gamma',3x,'AttenCoef')
!200	format(8(f6.2,2x))

!	stop


         return
      end


