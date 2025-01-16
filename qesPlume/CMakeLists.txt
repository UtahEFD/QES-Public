# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

#IF ($CACHE{HAS_CUDA_SUPPORT})
#  CUDA_ADD_EXECUTABLE( qesPlume
#          handlePlumeArgs.cpp
#          qesPlumeMain.cpp
#  )
#ELSE ($CACHE{HAS_CUDA_SUPPORT})
  ADD_EXECUTABLE( qesPlume
          handlePlumeArgs.cpp
          qesPlumeMain.cpp
  )
#ENDIF ($CACHE{HAS_CUDA_SUPPORT})

target_link_libraries(qesPlume PRIVATE qesplumecore)

IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(qesPlume PRIVATE qeswindsgpu)
ENDIF()
target_link_libraries(qesPlume PRIVATE qeswindscore)

target_link_libraries(qesPlume PRIVATE qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesPlume PRIVATE qesOptix)
ENDIF()

link_external_libraries(qesPlume)
