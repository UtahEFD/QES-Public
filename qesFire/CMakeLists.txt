
include_directories(${PROJECT_SOURCE_DIR}/src)

ADD_EXECUTABLE( qesFire
		handleQESArgs.cpp
		qesFireMain.cpp
	)

# target_link_libraries(qesFire curand)
target_link_libraries(qesFire PRIVATE qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
	target_link_libraries(qesFire PRIVATE qeswindsgpu)
	target_link_libraries(qesFire PRIVATE qesfiregpu)
ENDIF()
target_link_libraries(qesFire PRIVATE qesfirecore)
target_link_libraries(qesFire PRIVATE qesplumecore)
target_link_libraries(qesFire PRIVATE qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(qesFire PRIVATE qesOptix)
ENDIF()

link_external_libraries(qesFire)

