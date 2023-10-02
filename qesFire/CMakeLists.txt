include_directories(${CMAKE_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesFire
handleQESArgs.cpp
  qesFireMain.cpp
  )

target_link_libraries(qesFire curand)
target_link_libraries(qesFire qeswindscore)
target_link_libraries(qesFire qesfirecore)
target_link_libraries(qesFire qesplumecore)
target_link_libraries(qesFire qesutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesFire qesOptix)
ENDIF()

link_external_libraries(qesFire)

