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
      function zd_bisect(uH,uin,zref,pp,H,a,vk)result(zd)
         !this function uses the bisection method to find the zd displacement height
         !fnew
         !dnew
         !Tom Booth 2/23/04

         integer iter
         real uH, uin, zref, pp, H, a, vk, tol
         real zd, zd1, f1, d1, zd2, f2, d2, zdnew, fnew, dnew
	
         tol = 0.0001 ! Error tolerance

         !initial two guesses
         zd1 = 0.0000001
         zd2 = 2*H

         d1 = zd1 - uin/uH * H/a * exp(-a*(zd1/H -1)) / log(zref/pp)
         f1= zd1-H*(1/a*(log(abs(uin/uH *log(abs((zd1-d2)/pp))/log(abs(zref/pp)))))+1)

         d2 = zd2 - uin/uH * H/a * exp(-a*(zd2/H -1)) / log(zref/pp);
         f2= zd2-H*(1/a*(log(abs(uin/uH *log(abs((zd2-d2)/pp))/log(abs(zref/pp)))))+1)
	
         iter = 0
         fnew = 1.0 !to get the loop started


         do while(iter.lt.20000.and.abs(fnew).gt.tol)

            iter = iter + 1
            zdnew = (zd1 + zd2) / 2; ! Algorithm for bisect method

            dnew = zdnew - uin/uH * H/a * exp(-a*(zdnew/H -1)) / log(zref/pp);
            fnew=zdnew-H*(1/a*(log(abs(uin/uH *log(abs((zdnew-dnew)/pp))/log(abs(zref/pp)))))+1)

            if (fnew*f1.gt.0)then
               zd1 = zdnew
            endif
            if (fnew*f2.gt.0)then
               zd2 = zdnew
            endif

!	print*,zdnew

         enddo
	
         zd = zdnew

      end


