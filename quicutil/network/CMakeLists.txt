add_library (sive-network ${LIB_TYPE}
  ClientSocket.cpp ClientSocket.h 
  HTTPRequest.cpp HTTPRequest.h
  IPAddress.cpp IPAddress.h
  ProtocolMessage.cpp ProtocolMessage.h
  ProtocolMessageBuffer.cpp ProtocolMessageBuffer.h
  Socket.cpp Socket.h
  ServerSocket.cpp ServerSocket.h
  UDPSocket.cpp UDPSocket.h 
)

FILE(GLOB files "*.h")
INSTALL(FILES ${files} DESTINATION include/network)
INSTALL(TARGETS sive-network LIBRARY ARCHIVE DESTINATION lib)
