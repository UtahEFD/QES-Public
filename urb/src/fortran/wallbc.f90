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
      subroutine wallbc

         use datamodule ! make data from module "datamodule" visible
! operations  done a priori to speed up the code AAG & IS  07/03/06
         implicit none
         ! non boundary cells
!erp 6/12/2006 the following includes the redefinition of b.c. based on 
!C. Bathke work
         
lp044:   do k=2,nz-2
lp043:      do j=2,ny-2
lp042:         do i=2,nx-2
                  if(icellflag(i,j,k) .ne. 0)then
                     if(icellflag(i,j,k-1) .eq. 0)then
                        if(icellflag(i,j+1,k) .eq. 0)then
                           if(icellflag(i-1,j,k) .eq. 0)then
                              celltype(i,j,k)=15                      !left+front+below
                           elseif(icellflag(i+1,j,k) .eq. 0)then
                              celltype(i,j,k)=16                      !right+front+below
                           else
                              celltype(i,j,k)=7                       !below+front
                           endif
                        elseif(icellflag(i,j-1,k) .eq. 0)then
                           if(icellflag(i-1,j,k) .eq. 0)then
                              celltype(i,j,k)=18                      !left+behind+below   <----------------
                           elseif(icellflag(i+1,j,k) .eq. 0)then
                              celltype(i,j,k)=17                      !right+behind+below
                           else
                              celltype(i,j,k)=10                      !below+behind
                           endif
                        elseif(icellflag(i-1,j,k) .eq. 0)then
                           celltype(i,j,k)=8                          !below+left
                        elseif(icellflag(i+1,j,k) .eq. 0)then
                           celltype(i,j,k)=9                          !below+right
                        else
                           celltype(i,j,k)=1                          !wall below
                        endif
                     elseif(icellflag(i,j,k+1) .eq. 0)then
                        if(icellflag(i,j+1,k) .eq. 0)then
                           if(icellflag(i-1,j,k) .eq. 0)then
                              celltype(i,j,k)=19                      !left+front+above
                           elseif(icellflag(i+1,j,k) .eq. 0)then
                              celltype(i,j,k)=20                      !right+front+above
                           else
                              celltype(i,j,k)=23                      !above+front
                           endif
                        elseif(icellflag(i,j-1,k) .eq. 0)then
                           if(icellflag(i-1,j,k) .eq. 0)then
                              celltype(i,j,k)=22                      !left+behind+above
                           elseif(icellflag(i+1,j,k) .eq. 0)then
                              celltype(i,j,k)=21                      !right+behind+above
                           else
                              celltype(i,j,k)=26                      !above+behind
                           endif
                        elseif(icellflag(i-1,j,k) .eq. 0)then
                           celltype(i,j,k)=24                         !above+left
                        elseif(icellflag(i+1,j,k) .eq. 0)then
                           celltype(i,j,k)=25                         !above+right
                        else
                           celltype(i,j,k)=2                          !wall above
                        endif
                     elseif(icellflag(i-1,j,k) .eq. 0)then
                        if(icellflag(i,j+1,k) .eq. 0)then
                           celltype(i,j,k)=11                         !left+front
                        elseif(icellflag(i,j-1,k) .eq. 0)then
                           celltype(i,j,k)=14                         !left+behind
                        else
                           celltype(i,j,k)=4                          !wall to left
                        endif
                     elseif(icellflag(i+1,j,k) .eq. 0)then
                        if(icellflag(i,j+1,k) .eq. 0)then
                           celltype(i,j,k)=12                         !right+front
                        elseif(icellflag(i,j-1,k) .eq. 0)then
                           celltype(i,j,k)=13                         !right+behind
                        else
                           celltype(i,j,k)=3                          !wall to right
                        endif
                     elseif(icellflag(i,j+1,k) .eq. 0)then
                        celltype(i,j,k)=5                             !wall in front
                     elseif(icellflag(i,j-1,k) .eq. 0)then
                        celltype(i,j,k)=6                             !wall behind
                     endif
                  endif

               enddo   lp042      
            enddo   lp043      
         enddo   lp044      


lp047:   do k=1,nz-1
lp046:      do j=1,ny-1
lp045:         do i=1,nx-1
                  if(celltype(i,j,k).eq.40)then                        !fluid no boundary
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=1.
                     p(i,j,k)=1.
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.0)then  !p1 = 0 for inflow/outflow bc
                     e(i,j,k)=0.
                     f(i,j,k)=0.
                     g(i,j,k)=0.
                     h(i,j,k)=0.
                     m(i,j,k)=0.
                     n(i,j,k)=0.
                     o(i,j,k)=1.
                     p(i,j,k)=0.
                     q(i,j,k)=0.
                  elseif(celltype(i,j,k).eq.1)then                        !wall below
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=1.
                     p(i,j,k)=1.
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.2)then                        !wall above
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=1.
                     p(i,j,k)=1.
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.3)then                        !wall to right
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=1.
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.4)then                        !wall to left
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=1.
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.5)then                        !wall in front
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=1.
                     p(i,j,k)=.5
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.6)then                        !wall behind
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=1.
                     p(i,j,k)=.5
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.7)then                        !in front and below
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=1.
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.8)then                        !below and to the left
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=.5
                     p(i,j,k)=1.
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.9)then                        !below and to the right
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=.5
                     p(i,j,k)=1.
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.10)then                        !below and behind
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=1.
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.11)then                        !left+front
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.12)then                        !right+front
                     e(i,j,k)=.0
                     f(i,j,k)=1.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.13)then                        !right+behind
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.14)then                        !left+behind
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=1.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=1.
                  elseif(celltype(i,j,k).eq.15)then                        !left+front+below
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.16)then                        !right+front+below
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.17)then                        !right+behind+below
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.18)then                        !left+behind+below  <----------------------------
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=1.
                     n(i,j,k)=0.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.19)then                        !left+front+above
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.20)then                        !right+front+above
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.21)then                        !right+behind+above
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.22)then                        !left+behind+above
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.23)then                        !above+front
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=0.
                     h(i,j,k)=1.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=1.
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.24)then                        !above+left
                     e(i,j,k)=1.
                     f(i,j,k)=0.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=1.
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.25)then                        !above+right
                     e(i,j,k)=0.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=1.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=.5
                     p(i,j,k)=1.
                     q(i,j,k)=.5
                  elseif(celltype(i,j,k).eq.26)then                        !above+behind
                     e(i,j,k)=1.
                     f(i,j,k)=1.
                     g(i,j,k)=1.
                     h(i,j,k)=0.
                     m(i,j,k)=0.
                     n(i,j,k)=1.
                     o(i,j,k)=1.
                     p(i,j,k)=.5
                     q(i,j,k)=.5
                  endif


               enddo   lp045      
            enddo   lp046      
         enddo   lp047      
         
!erp 6/12/2006 end section modified based on C. Bathke work 
         return
      end
