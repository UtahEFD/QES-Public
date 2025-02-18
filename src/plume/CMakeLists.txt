include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

SET(qesPlumeCoreSources
  
  ParticleIDGen.cpp
  SourceIDGen.cpp
  Random.cpp
  RandomSingleton.cpp
  
  Interp.cpp
  InterpNearestCell.cpp
  InterpPowerLaw.cpp
  InterpTriLinear.cpp    
  
  Concentration.cpp
  Deposition.cpp
  StatisticsDirector.cpp
  
  GLE_Solver.cpp
  PLUMEGeneralData.cpp
  ParticleOutput.cpp
  
  PI_BoundaryConditions.hpp
  DomainBoundaryConditions.cpp
  
  ParticleModel.cpp
  
  Source.cpp
  SourceGeometryCube.cpp
  SourceGeometryLine.cpp
  SourceGeometryPoint.cpp
  SourceGeometrySphereShell.cpp
  
  PI_Source.cpp
  
  PI_ReleaseType_instantaneous.cpp
  PI_ReleaseType_continuous.cpp
  PI_ReleaseType_duration.cpp
  
  PI_SourceGeometry_Cube.cpp PI_SourceGeometry_Cube.hpp
  PI_SourceGeometry_FullDomain.cpp PI_SourceGeometry_FullDomain.hpp
  PI_SourceGeometry_Line.cpp PI_SourceGeometry_Line.hpp
  PI_SourceGeometry_Point.cpp PI_SourceGeometry_Point.hpp
  PI_SourceGeometry_SphereShell.cpp PI_SourceGeometry_SphereShell.hpp
  
  WallReflection.cpp
  WallReflection_StairStep.cpp
  WallReflection_TriMesh.cpp)

ADD_LIBRARY(qesplumecore ${qesPlumeCoreSources})

IF ($CACHE{HAS_CUDA_SUPPORT})
  SET(qesPlumeGPUSources
    CUDA/Partition.cu
    CUDA/RandomGenerator.cpp
    CUDA/Interpolation.cu
    CUDA/Model.cu
    CUDA/Concentration.cu)
  
  ADD_LIBRARY(qesplumegpu ${qesPlumeGPUSources})
  TARGET_INCLUDE_DIRECTORIES(qesplumegpu PRIVATE {CUDAToolkit_INCLUDE_DIRS})
  SET_TARGET_PROPERTIES(qesplumegpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

