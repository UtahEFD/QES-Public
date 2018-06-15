CMAKE_MINIMUM_REQUIRED (VERSION 2.8) 
project(cudaUrb)

SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH} )

SET(Boost_USE_STATIC_LIBS TRUE)

FIND_PACKAGE(Boost REQUIRED date_time program_options system thread unit_test_framework chrono timer)
IF(${Boost_FOUND})
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
ENDIF()

FIND_PACKAGE(NetCDF)
IF(NetCDF_FOUND)
  MESSAGE(STATUS "Found NetCDF: ${NetCDF_INCLUDE_DIRECTORIES}, ${NetCDF_LIBRARIES}")
ENDIF(NetCDF_FOUND)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")


include_directories(${CMAKE_SOURCE_DIR}/util)
include_directories(${CMAKE_SOURCE_DIR})

include_directories("/home/csgrads/patte720/Research/netcdf-cxx4-4.3.0.dist/include")

add_subdirectory( util )

link_directories("/home/csgrads/patte720/Research/netcdf-cxx4-4.3.0.dist/lib")

add_executable( cudaUrb
  handleURBArgs.cpp
  Solver.cpp
  CPUSolver.cpp
  NetCDFData.cpp
  main/cudaUrbMain.cpp
  )

target_link_libraries(cudaUrb qesutil)
target_link_libraries(cudaUrb ${Boost_PROGRAM_OPTIONS_LIBRARIES})
target_link_libraries(cudaUrb netcdf)
target_link_libraries(cudaUrb netcdf_c++4)
