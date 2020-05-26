include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
#LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

CUDA_ADD_LIBRARY( cudaplumecore
    
    Random.cpp
    Vector3.h
    
    Args.cpp
    Input.cpp
    NetCDFInput.cpp
    Urb.cpp
    Turb.cpp
    URBGeneralData.cpp
    TURBGeneralData.cpp
    Wall.cpp
    
    Eulerian.cpp
    Plume.cpp
    AdvectParticle.cpp
    WallReflection.cpp
    
    NetCDFOutput.cpp
    QESNetCDFOutput.cpp
    PlumeOutputLagrToEul.cpp
    PlumeOutputLagrangian.cpp
    PlumeOutputEulerian.cpp

    BoundaryConditions.hpp
    Particle.hpp

    Sources.hpp
    SourceKind.hpp
    
    ReleaseType.hpp
    ReleaseType_instantaneous.hpp
    ReleaseType_continuous.hpp
    ReleaseType_duration.hpp
    
    SourcePoint.cpp SourcePoint.hpp
    SourceLine.cpp SourceLine.hpp
    SourceCircle.cpp SourceCircle.hpp
    SourceCube.cpp SourceCube.hpp
    SourceFullDomain.cpp SourceFullDomain.hpp

)

#target_link_libraries (cudaplumecore quicloader) 
#target_link_libraries (cudaplumecore util)
#target_link_libraries (cudaplumecore ${Boost_PROGRAM_OPTIONS_LIBRARIES} )  
#
#if (APPLE)
#  find_package(OpenGL)
#  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
#  find_package(GLUT)
#  INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})
#  target_link_libraries (cudaplumecore ${OPENGL_LIBRARIES})
#  target_link_libraries (cudaplumecore ${GLUT_LIBRARIES})
#else (APPLE)
#  target_link_libraries (cudaplumecore GL)
#  target_link_libraries (cudaplumecore glut)
#  target_link_libraries (cudaplumecore GLU)
#endif (APPLE)
#target_link_libraries (cudaplumecore GLEW) 
