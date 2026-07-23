# ----------------------------------------------------------
# pyQES pybind11 extension modules
#
# Built only when QES_BUILD_PYTHON=ON (driven by scikit-build-core
# when producing the Python wheel). Each module wraps a thin C++
# "runner" that replicates the corresponding qes*Main.cpp workflow.
# ----------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Helper: create a pybind11 module, link QES cores + external libs,
# and install it inside the pyQES package of the wheel.
function(add_pyqes_module name)
  set(multiValueArgs SOURCES LINK)
  cmake_parse_arguments(PYMOD "" "" "${multiValueArgs}" ${ARGN})

  pybind11_add_module(${name} ${PYMOD_SOURCES})
  target_link_libraries(${name} PRIVATE ${PYMOD_LINK})
  IF ($CACHE{HAS_OPTIX_SUPPORT})
    target_link_libraries(${name} PRIVATE qesOptix)
  ENDIF()
  link_external_libraries(${name})

  install(TARGETS ${name} LIBRARY DESTINATION pyQES)
endfunction()

# GPU cores are linked in when CUDA support is compiled.
set(WINDS_GPU_LIBS "")
set(PLUME_GPU_LIBS "")
set(FIRE_GPU_LIBS "")
IF ($CACHE{HAS_CUDA_SUPPORT})
  set(WINDS_GPU_LIBS qeswindsgpu qesutilgpu)
  set(PLUME_GPU_LIBS qesplumegpu)
  set(FIRE_GPU_LIBS qesfiregpu)
ENDIF()

add_pyqes_module(_util
  SOURCES util_bindings.cpp
  LINK qesutil)

add_pyqes_module(_winds
  SOURCES winds_bindings.cpp runners/run_winds.cpp
  LINK qeswindscore ${WINDS_GPU_LIBS} qesutil)

add_pyqes_module(_plume
  SOURCES plume_bindings.cpp runners/run_plume.cpp
  LINK qesplumecore ${PLUME_GPU_LIBS} qeswindscore ${WINDS_GPU_LIBS} qesutil)

add_pyqes_module(_fire
  SOURCES fire_bindings.cpp runners/run_fire.cpp
  LINK qesfirecore ${FIRE_GPU_LIBS} qesplumecore ${PLUME_GPU_LIBS} qeswindscore ${WINDS_GPU_LIBS} qesutil)
