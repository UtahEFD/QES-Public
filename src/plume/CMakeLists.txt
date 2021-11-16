include_directories(${CMAKE_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../util)

#INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
#LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

CUDA_ADD_LIBRARY( qesplumecore
    
    Random.cpp
        
    Args.cpp
    
    Eulerian.cpp
    EulerianPowerLaw.cpp
    
    Plume.cpp
    AdvectParticle.cpp
    DepositParticle.cpp
    WallReflection.cpp
    
    PlumeOutput.cpp
    PlumeOutputParticleData.cpp
    PlumeOutputEulerian.cpp

    BoundaryConditions.hpp
    Particle.hpp
    
    ReleaseType.hpp
    ReleaseType_instantaneous.hpp
    ReleaseType_continuous.hpp
    ReleaseType_duration.hpp
    
    Sources.hpp
    SourceType.hpp
    SourcePoint.cpp SourcePoint.hpp
    SourceLine.cpp SourceLine.hpp
    SourceCircle.cpp SourceCircle.hpp
    SourceCube.cpp SourceCube.hpp
    SourceFullDomain.cpp SourceFullDomain.hpp

)

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
