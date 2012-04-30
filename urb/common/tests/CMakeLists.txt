if(QUIC_BUILD_TESTS)
  SET(PROGRAMS 
    absoluteDifference_tester 
    max_tester 
    max_timer 
    pack_tester
    sum_tester 
    sum_timer 
    zero_tester
    boundaryMatriceTester
  )
else()
  SET(PROGRAMS boundaryMatriceTester)
endif()

foreach(program ${PROGRAMS})
  CUDA_ADD_EXECUTABLE(${program} ${program}.cu)
  
  target_link_libraries (${program} CommonKernels)
endforeach(program)
