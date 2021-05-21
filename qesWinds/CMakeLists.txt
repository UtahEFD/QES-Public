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
  IF (ENABLE_CPU_TESTS)
    add_test(NAME CPU_FlatTerrain COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/FlatTerrain.xml -z -o test_Flatterrain_cpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  ENDIF()

  IF (ENABLE_GPU_TESTS)
    add_test(NAME GPU_FlatTerrain COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/FlatTerrain.xml -s 2 -z -o test_Flatterrain_gpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    add_test(NAME GPU_GaussianHill COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/GaussianHill.xml -s 2 -z -o test_GaussianHill_gpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  ENDIF()

  # longer sanity tests
  IF (ENABLE_LONG_SANITY_TESTS OR ENABLE_ALL_TESTS)
    IF (ENABLE_CPU_TESTS)
      add_test(NAME CPU_GaussianHill COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/GaussianHill.xml -z -o test_GaussianHill_cpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      add_test(NAME CPU_GranitePeak COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/GranitePeak.xml -z -o test_GranitePeak_cpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      add_test(NAME CPU_OklahomaCity COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/Oklahoma_city.xml -z -o test_OKC_cpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    ENDIF()

    IF (ENABLE_GPU_TESTS)
      add_test(NAME GPU_OklahomaCity COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/Oklahoma_city.xml -s 2 -z -o test_OKC_gpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      add_test(NAME GPU_GranitePeak COMMAND ${CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${CMAKE_SOURCE_DIR}/data/InputFiles/GranitePeak.xml -s 2 -z -o test_GranitePeak_gpu WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    ENDIF()
  ENDIF()
ENDIF()
