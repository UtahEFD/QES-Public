include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
#LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

SET( qesPlumeCoreSources
    Random.cpp
    RandomSingleton.cpp

    Interp.cpp
    InterpNearestCell.cpp
    InterpPowerLaw.cpp
    InterpTriLinear.cpp    

    Deposition.cpp
    Concentration.cpp
    StatisticsDirector.cpp

    Plume.cpp
    AdvectParticle.cpp
    DepositParticle.cpp

    PLUMEGeneralData.cpp

    PlumeOutput.cpp
    PlumeOutputParticleData.cpp

    BoundaryConditions.hpp
    DomainBoundaryConditions.cpp

    ParticleManager.cpp
    ParticleContainers.cpp

    TracerParticle_Model.cpp
    TracerParticle_Source.cpp
    TracerParticle_Statistics.cpp

    HeavyParticle_Model.cpp
    HeavyParticle_Source.cpp

    Particle.hpp
    ParticleFactories.hpp
    Particle_Tracer.hpp
    Particle_Heavy.hpp
    ParticleFactories.cpp

    ReleaseType.hpp
    ReleaseType_instantaneous.hpp
    ReleaseType_continuous.hpp
    ReleaseType_duration.hpp

    PI_Source.cpp
    SourceGeometry.hpp
    SourceGeometry_Cube.cpp SourceGeometry_Cube.hpp
    SourceGeometry_FullDomain.cpp SourceGeometry_FullDomain.hpp
    SourceGeometry_Line.cpp SourceGeometry_Line.hpp
    SourceGeometry_Point.cpp SourceGeometry_Point.hpp
    SourceGeometry_SphereShell.cpp SourceGeometry_SphereShell.hpp

    Source.cpp
    Source_TracerParticles.cpp
    Source_HeavyParticles.cpp

    WallReflection.cpp
    WallReflection_StairStep.cpp
    WallReflection_TriMesh.cpp
    
)

IF ($CACHE{HAS_CUDA_SUPPORT})
   CUDA_ADD_LIBRARY( qesplumecore
      ${qesPlumeCoreSources}
   )
ELSE ($CACHE{HAS_CUDA_SUPPORT})
   ADD_LIBRARY( qesplumecore
      ${qesPlumeCoreSources}
   )
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
