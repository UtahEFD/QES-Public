# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesPlume
  qesPlumeMain.cpp
  )

target_link_libraries(qesPlume qesplumecore)
target_link_libraries(qesPlume qeswindscore)
target_link_libraries(qesPlume qesutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesPlume qesOptix)
ENDIF()

link_external_libraries(qesPlume)
