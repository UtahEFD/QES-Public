add_library (util ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  fileHandling.h
#  handleGraphicsArgs.cpp handleGraphicsArgs.h
#  handleNetworkArgs.cpp handleNetworkArgs.h
  handlePlumeArgs.cpp handlePlumeArgs.h
  handleQUICArgs.cpp handleQUICArgs.h
  logstream.cpp logstream.h
  Vector3D.cpp Vector3D.h
  Random.cpp Random.h
  ResourcesHandler.cpp ResourcesHandler.h
  Timer.cpp Timer.h
  
  # From QUIC_URB
  angle.h
  angle.cpp
  normal.h
  normal.cpp
  GLSL.h
  GLSL.cpp
  StopWatch.h
  StopWatch.cpp
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS util LIBRARY ARCHIVE DESTINATION lib)

