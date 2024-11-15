include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
#LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

SET( qesPlumeCoreSources
   
    IDGenerator.cpp
    Random.cpp
    RandomSingleton.cpp

    Interp.cpp
    InterpNearestCell.cpp
    InterpPowerLaw.cpp
    InterpTriLinear.cpp    

    Deposition.cpp
    StatisticsDirector.cpp

    GLE_Solver.cpp
    PLUMEGeneralData.cpp
    ParticleOutput.cpp

    PI_BoundaryConditions.hpp
    DomainBoundaryConditions.cpp

    TracerParticle_Model.cpp
    TracerParticle_Source.cpp
    TracerParticle_Concentration.cpp
    
    #HeavyParticle_Model.cpp
    #HeavyParticle_Source.cpp
    #HeavyParticle_Concentration.cpp

    PI_Source.cpp

    PI_ReleaseType.hpp
    PI_ReleaseType_instantaneous.hpp
    PI_ReleaseType_continuous.hpp
    PI_ReleaseType_duration.hpp

    PI_SourceGeometry.hpp
    PI_SourceGeometry_Cube.cpp PI_SourceGeometry_Cube.hpp
    PI_SourceGeometry_FullDomain.cpp PI_SourceGeometry_FullDomain.hpp
    PI_SourceGeometry_Line.cpp PI_SourceGeometry_Line.hpp
    PI_SourceGeometry_Point.cpp PI_SourceGeometry_Point.hpp
    PI_SourceGeometry_SphereShell.cpp PI_SourceGeometry_SphereShell.hpp

    WallReflection.cpp
    WallReflection_StairStep.cpp
    WallReflection_TriMesh.cpp)

IF ($CACHE{HAS_CUDA_SUPPORT})
   CUDA_ADD_LIBRARY( qesplumecore
      ${qesPlumeCoreSources}
   )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
   ADD_LIBRARY( qesplumecore
      ${qesPlumeCoreSources}
   )
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

IF ($CACHE{HAS_CUDA_SUPPORT})
  CUDA_ADD_LIBRARY(qesplumegpu
    cuda/Partition.cu
    cuda/RandomGenerator.cpp
    cuda/Interpolation.cu
    cuda/Model.cu
    cuda/Concentration.cu)
ENDIF ($CACHE{HAS_CUDA_SUPPORT})

#target_link_libraries (qesplumecore quicloader) 
#target_link_libraries (qesplumecore util)
#target_link_libraries (qesplumecore ${Boost_PROGRAM_OPTIONS_LIBRARIES} )  
#
#if (APPLE)
#  find_package(OpenGL)
#  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
#  find_package(GLUT)
#  INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})
#  target_link_libraries (qesplumecore ${OPENGL_LIBRARIES})
#  target_link_libraries (qesplumecore ${GLUT_LIBRARIES})
#else (APPLE)
#  target_link_libraries (qesplumecore GL)
#  target_link_libraries (qesplumecore glut)
#  target_link_libraries (qesplumecore GLU)
#endif (APPLE)
#target_link_libraries (qesplumecore GLEW) 
