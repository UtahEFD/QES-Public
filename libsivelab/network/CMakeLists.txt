add_library (sive-network ${LIB_TYPE}
  IPAddress.cpp IPAddress.h
  Socket.cpp Socket.h
  UDPSocket.cpp UDPSocket.h 
  ServerSocket.cpp ServerSocket.h
  ClientSocket.cpp ClientSocket.h 
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/network)
INSTALL(TARGETS sive-network LIBRARY ARCHIVE DESTINATION lib)
