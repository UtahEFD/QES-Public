# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

#IF ($CACHE{HAS_CUDA_SUPPORT})
#	CUDA_ADD_EXECUTABLE( qes
#		     handleQESArgs.cpp
#                     qesMain.cpp	
#		     )
#ELSE ($CACHE{HAS_CUDA_SUPPORT})
	ADD_EXECUTABLE( qes
		     handleQESArgs.cpp
                     qesMain.cpp	
		     )
#ENDIF ($CACHE{HAS_CUDA_SUPPORT})

target_link_libraries(qes PRIVATE qesplumecore)
target_link_libraries(qes PRIVATE qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(qes PRIVATE qeswindsgpu)
ENDIF()
target_link_libraries(qes PRIVATE qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qes PRIVATE qesOptix)
ENDIF()

link_external_libraries(qes)
