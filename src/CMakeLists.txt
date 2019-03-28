include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/quicutil)  
LINK_DIRECTORIES (${CMAKE_BINARY_DIR}/quicutil/)

CUDA_ADD_LIBRARY( cudaplumecore
    shader/particle/shaders.cpp kernel/kernel_interface.cu 
    shader/particle/render_particles.cpp  
    plumeSystem.cpp  
    Eulerian.cpp
    Util.cpp
    #advectPar.cpp
    Random.cpp
    #Dispersion.cpp
    Args.cpp
    Input.cpp
    Output.cpp
    Urb.cpp
    Turb.cpp
)

target_link_libraries (cudaplumecore quicloader) 
target_link_libraries (cudaplumecore util)
target_link_libraries (cudaplumecore ${Boost_PROGRAM_OPTIONS_LIBRARIES} )  

if (APPLE)
  find_package(OpenGL)
  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
  find_package(GLUT)
  INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})
  target_link_libraries (cudaplumecore ${OPENGL_LIBRARIES})
  target_link_libraries (cudaplumecore ${GLUT_LIBRARIES})
else (APPLE)
  target_link_libraries (cudaplumecore GL)
  target_link_libraries (cudaplumecore glut)
  target_link_libraries (cudaplumecore GLU)
endif (APPLE)
target_link_libraries (cudaplumecore GLEW) 