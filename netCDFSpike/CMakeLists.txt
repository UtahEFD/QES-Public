cmake_minimum_required (VERSION 2.8) 
PROJECT (netCDF)

# 
#	CMake Module path for including additional CMake macros to make
#	finding libraries and other requirements easier.
#
SET( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" ${CMAKE_MODULE_PATH} )

#
#	NetCDF
#	It's not required unless you want to use the NetCDFParser class.
#
FIND_PACKAGE (NetCDF COMPONENTS "CXX" REQUIRED)
IF(NETCDF_FOUND)
  MESSAGE(STATUS "NetCDF Libs: ${NETCDF_LIBRARIES}")
ENDIF(NETCDF_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")

add_executable(netCDF
  main.cpp)
target_link_libraries(netCDF ${NETCDF_LIBRARIES})
