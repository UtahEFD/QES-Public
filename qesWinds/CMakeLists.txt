# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

#IF ($CACHE{HAS_CUDA_SUPPORT})
#	CUDA_ADD_EXECUTABLE( qesWinds
#			handleWINDSArgs.cpp
#			qesWindsMain.cpp
#	)
#ELSE ($CACHE{HAS_CUDA_SUPPORT})
	ADD_EXECUTABLE( qesWinds
			handleWINDSArgs.cpp
			qesWindsMain.cpp
	)
#ENDIF ($CACHE{HAS_CUDA_SUPPORT})

# target_link_libraries(qesWinds curand)
target_link_libraries(qesWinds PRIVATE qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(qesWinds PRIVATE qeswindsgpu)
ENDIF()
target_link_libraries(qesWinds PRIVATE qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesWinds PRIVATE qesOptix)
ENDIF()

link_external_libraries(qesWinds)

