#include "ServerSocket.h"

using namespace sivelab;

ServerSocket::ServerSocket()
{
  m_family = AF_INET6;
}

ServerSocket::ServerSocket(int port)
  : m_portNum( port )
{
  m_family = AF_INET6;

  m_socket_fd = socket(m_family, SOCK_STREAM, 0);
  if (m_socket_fd == -1)
    {
      throw SocketException("Socket creation failure!");
    }

  memset(&m_servAddr, 0, sizeof(m_servAddr));
  m_servAddr.sin6_family = m_family;
  m_servAddr.sin6_port = htons(m_portNum);
  m_servAddr.sin6_addr = in6addr_any;
  
  if (bind(m_socket_fd, (const sockaddr*)&m_servAddr, sizeof(m_servAddr)) == -1)
    {
      throw SocketException("Socket bind failure!");
    }

  if (listen(m_socket_fd, 512) == -1)
    {
      throw SocketException("Socket listen failure!");
    }
}

ClientSocket *ServerSocket::accept() const
{
  struct sockaddr_in6 clientAddr;
  socklen_t addrLength = sizeof(clientAddr);

  // Make sure to call the system call, thus the :: prefixing the accept call.
  int clientSocket_fd = ::accept(m_socket_fd, (sockaddr*)&clientAddr, &addrLength);
  if (clientSocket_fd == 0)
    {
      throw SocketException("Socket accept failure!");
    }

  std::cout << "Connect from client at port: " << ntohs( clientAddr.sin6_port ) << ", family: " << clientAddr.sin6_family << std::endl;

  return new ClientSocket( clientSocket_fd, clientAddr );
}

