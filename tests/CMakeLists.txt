include_directories(${CMAKE_SOURCE_DIR})

# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

CUDA_ADD_EXECUTABLE(compareSolvers compareSolvers.cpp)

#target_link_libraries(compareSolvers curand)
target_link_libraries(compareSolvers qeswindscore)
target_link_libraries(compareSolvers qeswindsutil)

IF (HAS_OPTIX_SUPPORT MATCHES ON)
  target_link_libraries(compareSolvers qesOptix)
ENDIF()

target_link_libraries(compareSolvers ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(compareSolvers ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(compareSolvers ${GDAL_LIBRARY})
target_link_libraries(compareSolvers ${NETCDF_LIBRARIES_CXX})
target_link_libraries(compareSolvers ${NETCDF_LIBRARIES_C})
#target_link_libraries(compareSolvers cudadevrt)
target_link_libraries(compareSolvers ${CUDA_LIBRARIES})

# easy to use functions to add sanity tests
## just specify the infile's name (w/o extension) as below
#function(add_sanity_test_CPU infile_prefix)
#  add_test(NAME CPU_${#infile_prefix} COMMAND ${#CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${#CMAKE_SOURCE_DIR}/data/InputFiles/${#infile_prefix}#.xml -z -o test_${#infile_prefix}_CPU WORKING_DIRECTORY ${#CMAKE_BINARY_DIR})
#endfunction()

#function(add_sanity_test_GPU infile_prefix)
#  add_test(NAME GPU_${#infile_prefix} COMMAND ${#CMAKE_BINARY_DIR}/qesWinds/qesWinds -q ${#CMAKE_SOURCE_DIR}/data/InputFiles/${#infile_prefix}#.xml -s 2 -z -o test_${#infile_prefix}_GPU WORKING_DIRECTORY ${#CMAKE_BINARY_DIR})
#endfunction()

# sanity tests
#IF (ENABLE_SANITY_TESTS OR ENABLE_ALL_TESTS)
#  IF (ENABLE_CPU_TESTS)
#    add_sanity_test_CPU(FlatTerrain)
#    IF (ENABLE_LONG_SANITY_TESTS)
#      add_sanity_test_CPU(GaussianHill)
#      add_sanity_test_CPU(GranitePeak)
#      add_sanity_test_CPU(Oklahoma_city)
#    ENDIF()
#  ENDIF()

#  IF (ENABLE_GPU_TESTS)
#    add_sanity_test_GPU(FlatTerrain)
#    add_sanity_test_GPU(GaussianHill)
#    IF (ENABLE_LONG_SANITY_TESTS)
#      add_sanity_test_GPU(GranitePeak)
#      add_sanity_test_GPU(Oklahoma_city)
#    ENDIF()
#  ENDIF()
#ENDIF()
