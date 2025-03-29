include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(qesWindCoreSources
        ../qes/Domain.cpp ../qes/Domain.h
        Canopy.cpp
        CanopyElement.cpp
        CanopyHomogeneous.cpp
        CanopyIsolatedTree.cpp
        CanopyWindbreak.cpp
        CanopyROC.cpp
        Cell.cpp
        CutCell.cpp
        CutBuilding.cpp
        DTEHeightField.cpp
        LocalMixing.cpp
        LocalMixingDefault.cpp
        LocalMixingNetCDF.cpp
        LocalMixingSerial.cpp
        LocalMixingOptix.cpp
        PolyBuilding.cpp PolyBuilding.h
        Sensor.cpp
        Solver.cpp
        Solver_CPU.cpp
        Solver_CPU_RB.cpp
        SolverFactory.cpp
        TURBParams.h
        TURBGeneralData.cpp
        TURBGeneralData.h
        TURBOutput.cpp
        TURBWall.cpp
        TURBWallBuilding.cpp
        TURBWallTerrain.cpp
	WindProfilerSensorType.cpp
	WindProfilerBarnCPU.cpp
        WindProfilerBarnGPU.cpp
        WindProfilerWRF.cpp
	WindProfilerHRRR.cpp
        WINDSInputData.h
        WINDSGeneralData.cpp WINDSGeneralData.h
        WINDSOutputVisualization.cpp
        WINDSOutputWorkspace.cpp
        WRFInput.cpp WRFInput.h
        Wall.cpp Wall.h
        UpwindCavity.cpp
        NonLocalMixing.cpp
        PolygonWake.cpp
        StreetCanyon.cpp
        StreetCanyonModified.cpp
        Sidewall.cpp
        Rooftop.cpp
        StreetIntersection.cpp
        TimeSeries.h
	HRRRData.cpp
	HRRRInput.h)

ADD_LIBRARY(qeswindscore ${qesWindCoreSources})


IF ($CACHE{HAS_CUDA_SUPPORT})
  MESSAGE(STATUS "CUDA Toolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
  SET(qesWindGPUSources
          # While we get this verified on CUDA 12.8, we will
          # replace use of it with the GlobalMemory solver.
          #    Solver_GPU_DynamicParallelism.cu
          Solver_GPU_GlobalMemory.cu
          Solver_GPU_SharedMemory.cu
          WINDSGeneralData.cu
          TURBGeneralData.cu
          WindProfilerBarnGPU.cu)
  ADD_LIBRARY(qeswindsgpu ${qesWindGPUSources})
  TARGET_INCLUDE_DIRECTORIES(qeswindsgpu PRIVATE {CUDAToolkit_INCLUDE_DIRS})
  set_target_properties(qeswindsgpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})


IF ($CACHE{HAS_OPTIX_SUPPORT})
  MESSAGE(STATUS "Compiling in OptiX support.")

  INCLUDE_DIRECTORIES(${OptiX_INCLUDE})

  cuda_compile_and_embed(EMBEDDED_OPTIX ${CMAKE_CURRENT_SOURCE_DIR}/OptixRayTrace.cu OptixRayTrace)

  # Add the generated header file to your target's include directories
  include_directories(${CMAKE_BINARY_DIR}/ptx)

  MESSAGE(STATUS "Info about ptx embedded CPP file:  ${EMBEDDED_OPTIX}")
  add_library(qesOptix
          OptixRayTrace.cpp OptixRayTrace.h
          ${EMBEDDED_OPTIX})

ENDIF ($CACHE{HAS_OPTIX_SUPPORT})
