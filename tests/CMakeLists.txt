# needs to be relative to the current project when we
# hierarchically integrate projects together.
include_directories(${PROJECT_SOURCE_DIR}/src)

IF (ENABLE_UNITTESTS)
  Include(FetchContent)

  FetchContent_Declare(
          Catch2
          GIT_REPOSITORY https://github.com/catchorg/Catch2.git
          GIT_TAG        v3.0.1 # or a later release
  )

  FetchContent_MakeAvailable(Catch2)

  add_subdirectory(unitTests)
ENDIF()



IF ($CACHE{HAS_CUDA_SUPPORT})
	CUDA_ADD_EXECUTABLE(compareSolvers compareSolvers.cpp)
ELSE ($CACHE{HAS_CUDA_SUPPORT})
	ADD_EXECUTABLE(compareSolvers compareSolvers.cpp)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})


IF ($CACHE{HAS_CUDA_SUPPORT})
  # target_link_libraries(compareSolvers curand)
ENDIF()
target_link_libraries(compareSolvers qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(compareSolvers qeswindsgpu)
ENDIF()
target_link_libraries(compareSolvers qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(compareSolvers qesOptix)
ENDIF()

target_link_libraries(compareSolvers ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(compareSolvers ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(compareSolvers ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(compareSolvers ${GDAL_LIBRARY})
target_link_libraries(compareSolvers ${NETCDF_LIBRARIES_CXX})
target_link_libraries(compareSolvers ${NETCDF_LIBRARIES_C})
IF ($CACHE{HAS_CUDA_SUPPORT})
  # target_link_libraries(compareSolvers cudadevrt)
  target_link_libraries(compareSolvers ${CUDA_LIBRARIES})
ENDIF()


IF ($CACHE{HAS_CUDA_SUPPORT})
	CUDA_ADD_EXECUTABLE( compareWindsResults
                     handleCompareArgs.cpp
                     compareWindsResults.cpp
		     )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
	ADD_EXECUTABLE( compareWindsResults
                     handleCompareArgs.cpp
                     compareWindsResults.cpp
		     )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})


IF ($CACHE{HAS_CUDA_SUPPORT})
  # target_link_libraries(compareWindsResults curand)
ENDIF()
target_link_libraries(compareWindsResults qeswindscore)
IF ($CACHE{HAS_CUDA_SUPPORT})
  target_link_libraries(compareWindsResults qeswindsgpu)
ENDIF()
target_link_libraries(compareWindsResults qeswindscore)
target_link_libraries(compareWindsResults qesutil)

IF ($CACHE{HAS_OPTIX_SUPPORT})
  target_link_libraries(compareWindsResults qesOptix)
ENDIF()

target_link_libraries(compareWindsResults ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(compareWindsResults ${Boost_DATE_TIME_LIBRARIES})
target_link_libraries(compareWindsResults ${Boost_FILESYSTEM_LIBRARIES})
target_link_libraries(compareWindsResults ${GDAL_LIBRARY})
target_link_libraries(compareWindsResults ${NETCDF_LIBRARIES_CXX})
target_link_libraries(compareWindsResults ${NETCDF_LIBRARIES_C})
IF ($CACHE{HAS_CUDA_SUPPORT})
  # target_link_libraries(compareWindsResults cudadevrt)
  target_link_libraries(compareWindsResults ${CUDA_LIBRARIES})
ENDIF()

#Comparison tests function
function(add_comparison_test infile_prefix solver_arg solver_name)
  add_test(NAME ${solver_name}_COMPARISON_${infile_prefix} COMMAND ${CMAKE_BINARY_DIR}/tests/compareSolvers -q ${CMAKE_SOURCE_DIR}/data/InputFiles/${infile_prefix}.xml -s ${solver_arg} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()
    
IF(ENABLE_ALL_TESTS OR ENABLE_ALL_COMPARISON_TESTS)
  add_comparison_test(FlatTerrain 1 ALL)
  #add_comparison_test(GaussianHill 1)
  IF(ENABLE_LONG_COMPARISON_TESTS)
    add_comparison_test(GaussianHill 1 ALL)
  ENDIF()
  IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
    add_comparison_test(AskerveinHill 1 ALL)
  ENDIF()
ENDIF()

IF(ENABLE_DYNAMIC_PARALLELISM_COMPARISON_TESTS AND NOT (ENABLE_ALL_TESTS OR ENABLE_ALL_COMPARISON_TESTS))
  add_comparison_test(FlatTerrain 2 DYNAMIC)
  IF(ENABLE_LONG_COMPARISON_TESTS)
    add_comparison_test(GaussianHill 2 DYNAMIC)
  ENDIF()
  IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
    add_comparison_test(AskerveinHill 2 DYNAMIC)
  ENDIF()
ENDIF()

IF(ENABLE_GLOBAL_COMPARISON_TESTS AND NOT (ENABLE_ALL_TESTS OR ENABLE_ALL_COMPARISON_TESTS))
  add_comparison_test(FlatTerrain 3 GLOBAL)
  IF(ENABLE_LONG_COMPARISON_TESTS)
    add_comparison_test(GaussianHill 3 GLOBAL)
  ENDIF()
  IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)    
    add_comparison_test(AskerveinHill 3 GLOBAL)
  ENDIF()
ENDIF()

IF(ENABLE_SHARED_COMPARISON_TESTS AND NOT (ENABLE_ALL_TESTS OR ENABLE_ALL_COMPARISON_TESTS))
  add_comparison_test(FlatTerrain 4 SHARED)
  IF(ENABLE_LONG_COMPARISON_TESTS)
    add_comparison_test(GaussianHill 4 SHARED)
  ENDIF()
  IF(ENABLE_RIDICULOUSLY_LONG_COMPARISON_TESTS)
    add_comparison_test(AskerveinHill 4 SHARED)
  ENDIF()
ENDIF()



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
