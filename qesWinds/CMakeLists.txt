include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE( qesWinds
  qesWindsMain.cpp
  )

target_link_libraries(qesWinds curand)
target_link_libraries(qesWinds qeswindscore)
target_link_libraries(qesWinds qeswindsutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(qesWinds qesOptix)
ENDIF()

target_link_libraries(qesWinds ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(qesWinds ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(qesWinds ${GDAL_LIBRARY})
target_link_libraries(qesWinds ${NETCDF_LIBRARIES_CXX})
target_link_libraries(qesWinds ${NETCDF_LIBRARIES_C})
target_link_libraries(qesWinds cudadevrt)
target_link_libraries(qesWinds ${CUDA_LIBRARIES})

# sanity tests
IF (ENABLE_SANITY_TESTS OR ENABLE_ALL_TESTS)
add_test(NAME CPU_FlatTerrain COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/FlatTerrain.xml)
add_test(NAME GPU_FlatTerrain COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/FlatTerrain.xml -s 2)
add_test(NAME GPU_GaussianHill COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/GaussianHill.xml -s 2)

# longer sanity tests
IF (ENABLE_LONG_SANITY_TESTS OR ENABLE_ALL_TESTS)
add_test(NAME CPU_GaussianHill COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/GaussianHill.xml)
add_test(NAME CPU_GranitePeak COMMAND ${CMAKE_BINARY_DIR}/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/GranitePeak.xml)
add_test(NAME GPU_GranitePeak COMMAND ${CMAKE_BINARY_DIR}/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/GranitePeak.xml -s 2)

add_test(NAME CPU_OklahomaCity COMMAND ${CMAKE_BINARY_DIR}/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/Oklahoma_city.xml)
add_test(NAME GPU_OklahomaCity COMMAND ${CMAKE_BINARY_DIR}/qesWinds -q
  ${CMAKE_SOURCE_DIR}/data/InputFiles/Oklahoma_city.xml -s 2)
ENDIF()
ENDIF()
