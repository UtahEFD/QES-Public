# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

if(HAS_CUDA_SUPPORT)
  CUDA_ADD_EXECUTABLE( qesPlume
    qesPlumeMain.cpp
  )
else(HAS_CUDA_SUPPORT)
  ADD_EXECUTABLE( qesPlume
    qesPlumeMain.cpp
  )
endif(HAS_CUDA_SUPPORT)

target_link_libraries(qesPlume qesplumecore)

target_link_libraries(qesPlume qeswindscore)
IF (HAS_CUDA_SUPPORT)
  target_link_libraries(qesPlume qeswindsgpu)
ENDIF()
target_link_libraries(qesPlume qeswindscore)

target_link_libraries(qesPlume qesutil)

IF (HAS_OPTIX_SUPPORT)
  target_link_libraries(qesPlume qesOptix)
ENDIF()

link_external_libraries(qesPlume)
