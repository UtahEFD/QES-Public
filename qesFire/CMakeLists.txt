
include_directories(${PROJECT_SOURCE_DIR}/src)

IF ($CACHE{HAS_CUDA_SUPPORT})
	CUDA_ADD_EXECUTABLE( qesFire
		handleQESArgs.cpp
  		qesFireMain.cpp
  )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
	ADD_EXECUTABLE( qesFire
		handleQESArgs.cpp
		qesFireMain.cpp
	)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

# target_link_libraries(qesFire curand)
target_link_libraries(qesFire qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
	target_link_libraries(qesFire qeswindsgpu)
	target_link_libraries(qesFire qesfiregpu)
ENDIF()
target_link_libraries(qesFire qesfirecore)
target_link_libraries(qesFire qesplumecore)
target_link_libraries(qesFire qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesFire qesOptix)
ENDIF()

link_external_libraries(qesFire)

