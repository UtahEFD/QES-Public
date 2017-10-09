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
      function bisect(ustar,zo,H,a,vk,psi_m)result(d)
         !this function uses the bisection method to find the zd displacement height
		implicit none
         integer iter
         real zo, H, a, vk, tol,uhc
         real d, fi, d1,d2, fnew, ustar, psi_m !fu,
	
!this function uses the bisection method to find the root of the specified
!equation 

         tol=zo/100
         fnew=tol*10

!initial two guesses
         d1 = zo;
         d2 = H;

         uhc = (ustar/vk)*(log((H-d1)/zo)+psi_m)
         fi = a*uhc*vk/ustar -  H/(H-d1)

! MAN 05/15/2007 fu never used so it is commented out
!         uhc = (ustar/vk)*log((H-d2)/zo)
!         fu = a*uhc*vk/ustar -  H/(H-d2)

         iter = 0;

         do while(iter.lt.200.and.abs(fnew).gt.tol)
            iter = iter + 1
            d = (d1 + d2) / 2 ! Algorithm for bisect method

            uhc = (ustar/vk)*(log((H-d)/zo)+psi_m)
            fnew = a*uhc*vk/ustar -  H/(H-d)
      
            if(fnew*fi.gt.0) then
               d1 = d
            elseif(fnew*fi.lt.0)then
               d2 = d
            endif
    

         enddo

      end
