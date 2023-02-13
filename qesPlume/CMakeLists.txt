# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_EXECUTABLE( qesPlume
    qesPlumeMain.cpp
  )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
  ADD_EXECUTABLE( qesPlume
    qesPlumeMain.cpp
  )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

target_link_libraries(qesPlume qesplumecore)

target_link_libraries(qesPlume qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(qesPlume qeswindsgpu)
ENDIF()
target_link_libraries(qesPlume qeswindscore)

target_link_libraries(qesPlume qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesPlume qesOptix)
ENDIF()

link_external_libraries(qesPlume)
