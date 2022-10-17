# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qes
		     handleQESArgs.cpp
                     qesMain.cpp	
		     )

target_link_libraries(qes qesplumecore)
target_link_libraries(qes qeswindscore)
target_link_libraries(qes qesutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qes qesOptix)
ENDIF()

link_external_libraries(qes)
