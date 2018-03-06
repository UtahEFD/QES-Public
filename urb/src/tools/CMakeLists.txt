include_directories(/Developer/CUDA/common/inc/ )
include_directories(/usr/local/cuda/include)

include_directories(${CUDA_CUT_INCLUDE_DIR})

INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  

add_library(UrbTools matrixViewer.cpp)

if(QUIC_BUILD_DEV)
  SET(PROGRAMS 
    celltypeCompare
    matrixCompare
    matrixCompareEqual
    matrixDifference
    max
    min
    matrix_viewer
    velocityCompare
  )

  foreach(program ${PROGRAMS})
    cuda_add_executable(${program} ${program}.cpp)
    
    target_link_libraries (${program} UrbTools)
    target_link_libraries (${program} UrbKernels)
    target_link_libraries (${program} UrbUtil)
    target_link_libraries (${program} quicloader)
    target_link_libraries (${program} util)
  endforeach(program)

endif()
