INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  

if(QUIC_BUILD_DEV)
  SET(PROGRAMS cuda_timer cuda_urb setup_urb viewer_urb)
else()
  SET(PROGRAMS cuda_urb )
endif()

foreach(program ${PROGRAMS})
  CUDA_ADD_EXECUTABLE(${program} ${program}.cpp)
  
  target_link_libraries (${program} UrbCUDA)
  target_link_libraries (${program} UrbKernels)
  target_link_libraries (${program} UrbUtil)
#  target_link_libraries (${program} sive-cudautil)
  target_link_libraries (${program} quicloader)
  target_link_libraries (${program} util)
  target_link_libraries (${program} m)
  # Order is IMPORTANT... gfortran is needed after Urbfortran
#  target_link_libraries (${program} Urbfortran)
#  target_link_libraries (${program} gfortran)

endforeach(program)

# These really need CUDA_ADD_LIB?
if(QUIC_BUILD_DEV)
  CUDA_ADD_LIBRARY(UrbCUDA
    building.cpp
    buildingList.cpp
    intersect.cpp
    pentagon.cpp
    #
    urbCUDA.cpp 
    urbHost.cpp
    urbModule.cpp 
    urbParser.cpp
    urbSetup.cpp 
    urbStopWatchList.cpp
    urbTimer.cpp
    urbViewer.cpp
  )
else()
  CUDA_ADD_LIBRARY(UrbCUDA
    building.cpp
    buildingList.cpp
    intersect.cpp
    pentagon.cpp
    #
    urbCUDA.cpp 
    urbModule.cpp 
    urbParser.cpp
    urbSetup.cpp 
    urbStopWatchList.cpp
  )
endif()



