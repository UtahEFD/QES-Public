if(QUIC_BUILD_DEV)
  SET(PROGRAMS cuda_timer cuda_urb setup_urb viewer_urb)
else()
  SET(PROGRAMS cuda_urb )
endif()

foreach(program ${PROGRAMS})
  CUDA_ADD_EXECUTABLE(${program} ${program}.cpp)
  
  target_link_libraries (${program} UrbCUDA)
  target_link_libraries (${program} UrbKernels) 
  target_link_libraries (${program} CommonKernels)
  target_link_libraries (${program} CommonUtil)
  target_link_libraries (${program} CommonQUIC)
  target_link_libraries (${program} sive-quicutil) 
  target_link_libraries (${program} m)
  target_link_libraries (${program} gfortran)
  target_link_libraries (${program} Urbfortran)

endforeach(program)

# These really need CUDA_ADD_LIB?
if(QUIC_BUILD_DEV)
  CUDA_ADD_LIBRARY(UrbCUDA
    urbModule.cpp 
    urbTimer.cpp
    urbSetup.cpp 
    urbCUDA.cpp 
    urbParser.cpp
    urbViewer.cpp
    urbHost.cpp
    urbStopWatchList.cpp
  )
else()
  CUDA_ADD_LIBRARY(UrbCUDA
    urbModule.cpp 
    urbSetup.cpp 
    urbCUDA.cpp 
    urbParser.cpp
    urbStopWatchList.cpp
  )
endif()


