# - Find NetCDF
# Find the native NetCDF includes and library
#
#  NETCDF_INCLUDES    - where to find netcdf.h, etc
#  NETCDF_LIBRARIES   - Link these libraries when using NetCDF
#  NETCDF_FOUND       - True if NetCDF found including required interfaces (see below)
#
# Your package can require certain interfaces to be FOUND by setting these
#
#  NETCDF_CXX         - require the C++ interface and link the C++ library
#  NETCDF_F77         - require the F77 interface and link the fortran library
#  NETCDF_F90         - require the F90 interface and link the fortran library
#
# In some installations of NETCDF, the different interfaces may be installed in separate 
# directories.  When this is the case, the other interfaces may not be located. For each interface,
# you can set CMake variables to provide the include path of that interface:
# 
#     NETCDF_CXX_DIR
#
# The following are not for general use and are included in
# NETCDF_LIBRARIES if the corresponding options above are set.
#
#  NETCDF_INCLUDES_CXX   - C++ interface (include dir), if available
#  NETCDF_LIBRARIES_CXX  - C++ interface (library), if available
#
#  NETCDF_LIBRARIES_F77  - Fortran 77 interface, if available
#  NETCDF_LIBRARIES_F90  - Fortran 90 interface, if available
#
# Normal usage would be:
#  set (NETCDF_F90 "YES")
#  find_package (NetCDF REQUIRED)
#  target_link_libraries (uses_f90_interface ${NETCDF_LIBRARIES})
#  target_link_libraries (only_uses_c_interface ${NETCDF_LIBRARIES_C})
#
# Or, 
#  set (NETCDF_CXX "YES")
#  find_package (NetCDF REQUIRED)
#  target_link_libraries (uses_f90_interface ${NETCDF_LIBRARIES})
#  target_link_libraries (only_uses_c_interface ${NETCDF_LIBRARIES_C})
#
# Updated by Pete Willemsen, January 2019
#   - look for C++4 (netcdf_c++4) bindings rather than older C++ bindings (netcdf_c++)
#   - allows the C library and C++ libraries to be installed in different locations
# 
# Taken from: https://github.com/jedbrown/cmake-modules/blob/master/FindNetCDF.cmake
#
# Copyright Constantine Khroulev
#           Jed Brown
#           johnfettig
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#

if (NETCDF_INCLUDES AND NETCDF_LIBRARIES)
  # Already in cache, be silent
  set (NETCDF_FIND_QUIETLY TRUE)
endif (NETCDF_INCLUDES AND NETCDF_LIBRARIES)

# On some systems, netcdf and netcdf_cxx are installed in different
# directories.  When this is the case, a single NETCDF_DIR location
# will not be able to locate both interfaces, say C and CXX. 
#
# Use NETCDF_DIR to help locate the base C interface. If
# NETCDF_CXX_DIR is set, we can add it to the search HINTS for finding
# any CXX-related files.

# Search for NetCDF C interfaces using well-known locations as well as the
# NETCDF_DIR hint and/or environment variable
find_path (NETCDF_INCLUDES netcdf.h
	   HINTS ${NETCDF_DIR} ENV NETCDF_DIR)
# message(STATUS "FindNetCDF.cmake: NETCDF_INCLUDES=${NETCDF_INCLUDES}")

# Extract the Parent directory path for NETCDF_C
get_filename_component(NETCDF_C_DIR ${NETCDF_INCLUDES} DIRECTORY)
# message(STATUS "Parent dir = ${NETCDF_C_DIR}")

# Locate the NETCDF Libraries for the C interface
find_library (NETCDF_LIBRARIES       NAMES netcdf   PATHS ${NETCDF_C_DIR}/lib)
mark_as_advanced(NETCDF_LIBRARIES)
# message(STATUS "FindNetCDF.cmake: NETCDF_LIBRARIES_C=${NETCDF_LIBRARIES_C}")

set (NetCDF_has_interfaces "YES") # will be set to NO if we're missing any interfaces
set (NetCDF_libs "${NETCDF_LIBRARIES}")

get_filename_component (NetCDF_lib_dirs "${NETCDF_LIBRARIES}" PATH)

macro (NetCDF_check_interface lang header libs)
  if (NETCDF_${lang})

    message(STATUS "Searching for NETCDF_${lang} interfaces (via header file ${header} and library ${libs})...")
    set (NETCDF_${lang}_SEARCHPATH ${NETCDF_DIR})
    if (NETCDF_${lang}_DIR)
      set (NETCDF_${lang}_SEARCHPATH ${NETCDF_${lang}_DIR})
      message(STATUS "     looking in ${NETCDF_${lang}_SEARCHPATH}")
    endif (NETCDF_${lang}_DIR)

    find_path (NETCDF_INCLUDES_${lang} NAMES ${header}
      HINTS ${NETCDF_${lang}_SEARCHPATH})
    # message(STATUS "   result NETCDF_INCLUDES_${lang} = ${NETCDF_INCLUDES_${lang}}")

    # Extract the Parent directory path for NETCDF_${lang}
    get_filename_component(NETCDF_${lang}_PDIR ${NETCDF_INCLUDES_${lang}} DIRECTORY)
    # message(STATUS "   Parent ${lang} dir = ${NETCDF_${lang}_PDIR}")

    find_library (NETCDF_LIBRARIES_${lang}    NAMES ${libs}      PATHS ${NETCDF_${lang}_PDIR}/lib)
    # message(STATUS "   result NETCDF_LIBRARIES_${lang} = ${NETCDF_LIBRARIES_${lang}}")
    mark_as_advanced (NETCDF_INCLUDES_${lang} NETCDF_LIBRARIES_${lang})
    
    if (NETCDF_INCLUDES_${lang} AND NETCDF_LIBRARIES_${lang})
      list (INSERT NetCDF_libs 0 ${NETCDF_LIBRARIES_${lang}}) # prepend so that -lnetcdf is last
    else (NETCDF_INCLUDES_${lang} AND NETCDF_LIBRARIES_${lang})
      set (NetCDF_has_interfaces "NO")
      message (STATUS "Failed to find NetCDF interface for ${lang}")
    endif (NETCDF_INCLUDES_${lang} AND NETCDF_LIBRARIES_${lang})
  endif (NETCDF_${lang})
endmacro (NetCDF_check_interface)

NetCDF_check_interface (CXX netcdf netcdf_c++4)  # header for CXX does not have .h
NetCDF_check_interface (F77 netcdf.inc  netcdff)
NetCDF_check_interface (F90 netcdf.mod  netcdff)

message(STATUS "All NetCDF Libraries: ${NetCDF_libs}")
set (NETCDF_LIBRARIES "${NetCDF_libs}" CACHE STRING "All NetCDF libraries required for requested interfaces")

# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NetCDF DEFAULT_MSG NETCDF_LIBRARIES NETCDF_INCLUDES NetCDF_has_interfaces)

mark_as_advanced (NETCDF_LIBRARIES NETCDF_INCLUDES)
