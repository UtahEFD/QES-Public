include_directories(/Developer/CUDA/common/inc/ )
include_directories(/usr/local/cuda/include)

include_directories(${CUDA_CUT_INCLUDE_DIR})


add_library(CommonTools matrixViewer.cpp)

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
    
    target_link_libraries (${program} CommonTools)
    target_link_libraries (${program} CommonKernels)
    target_link_libraries (${program} CommonUtil)
  endforeach(program)

endif()
