add_library (sive-util ${LIB_TYPE}
  ArgumentParsing.cpp ArgumentParsing.h
  fileHandling.h
  handleGraphicsArgs.cpp handleGraphicsArgs.h
  handleNetworkArgs.cpp handleNetworkArgs.h
  handlePlumeArgs.cpp handlePlumeArgs.h
  handleQUICArgs.cpp handleQUICArgs.h
  logstream.cpp logstream.h
  Vector3D.cpp Vector3D.h
  Random.cpp Random.h
  Timer.cpp Timer.h
  
  # From QUIC_URB
  angle.cpp
  normal.cpp
  GLSL.cpp
  StopWatch.cpp
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/util)
INSTALL(TARGETS sive-util LIBRARY ARCHIVE DESTINATION lib)

