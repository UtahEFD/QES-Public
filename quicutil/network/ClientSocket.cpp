#include "ClientSocket.h"

using namespace sivelab;

ClientSocket::ClientSocket()
  : m_portNum(-1), m_sockIP(""), m_peerIP("")
{
  m_family = AF_INET6;
}

ClientSocket::ClientSocket(const std::string &host, int port)
  : m_portNum( port ), m_sockIP(""), m_peerIP("")
{
  m_family = AF_INET6;

  // ideally, evaluate the host. determine if host ip (v4 or v6) or
  // hostname

  struct addrinfo hints, *res;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  // Convert the port number into a string to be sent into the
  // getaddrinfo function.
  std::ostringstream portNumString;
  portNumString << port;

  // Use getaddrinfo here so that all of the socket structures are
  // filled appropriately.  This essentially makes it easier to
  // connect to servers since we can use the domain name system to get
  // the address structures.
  int retVal = getaddrinfo(host.c_str(), portNumString.str().c_str(), &hints, &res);
  if (retVal != 0)
    {
      throw sivelab::SocketException("Unable to locate server host name or port!");
      // fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(retVal));
    }

  // Using the result from getaddrinfo, walk over the results, looking
  // for sockets that will allow for the connection to be made.
  int sockfd;
  struct addrinfo *currResult;
  
  bool socketNotFound = true;

  currResult = res;
  while (currResult && socketNotFound) 
    {
      // Attempt to create a socket...
      if ((sockfd = socket(currResult->ai_family, currResult->ai_socktype, currResult->ai_protocol)) == -1) 
	{
	  // if it wasn't created, try the next result
	  perror("server: socket");
	  continue;
        }

      int yes = 1;
#ifdef WIN32
	  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(int)) == -1) 
#else
	  if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1) 
#endif
	  {
	  perror("setsockopt");
	  // exit(1);
	  // continue;
	}

      // Using the socket that was created, try to connect to the
      // server... if this fails, try the next result, but be sure to
      // close the socket first!
      if (connect(sockfd, res->ai_addr, res->ai_addrlen) == 0)
	{
	  std::cout << "Connected!" << std::endl;
	  std::cout << "\tFamily: " << ((res->ai_family == AF_INET) ? "AF_INET" : "AF_INET6") << std::endl;

	  socketNotFound = false;
	  m_socket_fd = sockfd;
	}
      else
	{
	  // throw sivelab::SocketException("Socket connection failure!");
	  perror("couldn't connect!");
	  continue;
	}
      
      currResult = currResult->ai_next;
    }

  sockname();
  peername();

#if 0
  // make a socket:
  int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (sockfd == -1)
    {
      throw sivelab::SocketException("Socket creation failure!");
    }

  // connect!
  if (connect(sockfd, res->ai_addr, res->ai_addrlen) == -1)
    {
      throw sivelab::SocketException("Socket connection failure!");
    }
#endif
}

ClientSocket::ClientSocket(int socketfd, struct sockaddr_in6 &addr)
{
  m_socket_fd = socketfd;
  m_family = AF_INET6;

  memcpy(&m_clientAddr, &addr, sizeof(m_clientAddr));
}

int ClientSocket::send( std::string &msg )
{
  int numBytesSent = ::send( m_socket_fd, msg.c_str(), msg.length(), 0 );
  // std::cout << "sent " << numBytesSent << std::endl;

  if (numBytesSent == -1)
    {
      std::ostringstream errs;
      errs << "Socket send failure: (" << errno << ") " << strerror(errno);
      throw sivelab::SocketException(errs.str());    
    }

  return numBytesSent;
}

/** @brief Socket data sending
 * 
 * Function that sends a message buffer.
 */
int ClientSocket::send( ProtocolMessageBuffer &msg )
{
  //  std::cout << "m_socket_fd = " << m_socket_fd << std::endl;
  //  sockname();
  //  peername();


  // std::cout << "data = " << msg.byteArray() << ", length=" << msg.length() << std::endl;
  
  int numBytesSent = ::send(m_socket_fd, msg.byteArray(), msg.length(), 0);
  // std::cout << "sent " << numBytesSent << std::endl;

  if (numBytesSent == -1)
    {
      std::ostringstream errs;
      errs << "Socket send failure: (" << errno << ") " << strerror(errno);
      throw sivelab::SocketException(errs.str());    
    }

  return numBytesSent;
}


int ClientSocket::receive( std::string &msg ) { return 0; }

int ClientSocket::receive( ProtocolMessageBuffer &msg, bool allowBlocking )
{
  //  std::cout << "m_socket_fd = " << m_socket_fd << std::endl;
  //  sockname();
  //  peername();

  // Modify to read chars into this until there are no others...
  const int MSGSZ = 64000;
  char *buf = new char[MSGSZ];
  
  int selectRetVal = 0;

  // check the socket to see if it has more input to read 
  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(m_socket_fd, &rfds);

  // timer to wait is set to 0 to not wait
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 0;

  int numBytesReceived = 0, totalBytesReceived = 0;

  // use select to see if there's any data to be read... this is the
  // main entrance into this functionality.  
  if (!allowBlocking)
    selectRetVal = select(m_socket_fd+1, &rfds, NULL, NULL, &tv);

  while (selectRetVal || allowBlocking)
    {
      numBytesReceived = 0;
      numBytesReceived = ::recv( m_socket_fd, buf, MSGSZ, 0 );
      totalBytesReceived += numBytesReceived;
      
      //
      // package buffer into protocol message
      // 
      msg.addBufferData(buf, numBytesReceived);

      // use select to see if there's more data in this pass to read.
      // This is not the same as the blocking entry that got us into
      // this loop.  We want to keep reading data while it's being
      // sent.
      allowBlocking = false;
      selectRetVal = select(m_socket_fd+1, &rfds, NULL, NULL, &tv);
    } 

  std::cout << "PMBuf: " << msg.toString() << std::endl;

  delete [] buf;

  return totalBytesReceived;
}

std::string ClientSocket::peername()
{
  if (m_peerIP == "")
    {
      struct sockaddr_storage saddr;
      socklen_t sSz = sizeof(saddr); 

      memset(&saddr, 0, sSz);

      if (getpeername(m_socket_fd, (struct sockaddr *)&saddr, &sSz) == 0)
	{
	  char ipAddrStr[1024];

	  if (saddr.ss_family == AF_INET) {
	    struct sockaddr_in *tmp_saddr = (struct sockaddr_in*)&saddr;
	    inet_ntop(saddr.ss_family, (const void*)&(tmp_saddr->sin_addr), ipAddrStr, 1024);
	  }
	  else if (saddr.ss_family == AF_INET6) {
	    struct sockaddr_in6 *tmp_saddr = (struct sockaddr_in6*)&saddr;
	    inet_ntop(saddr.ss_family, (const void*)&(tmp_saddr->sin6_addr), ipAddrStr, 1024);
	  }

	  m_peerIP = ipAddrStr;
	}
      else
	std::cout << "getsockname error!" << std::endl;
    }
  
  return m_peerIP;
}


std::string ClientSocket::sockname()
{
  if (m_sockIP == "")
    {
      struct sockaddr_storage saddr;
      socklen_t sSz = sizeof(saddr); 

      memset(&saddr, 0, sSz);

      if (getsockname(m_socket_fd, (struct sockaddr *)&saddr, &sSz) == 0)
	{
	  char ipAddrStr[1024];

	  if (saddr.ss_family == AF_INET) {
	    struct sockaddr_in *tmp_saddr = (struct sockaddr_in*)&saddr;
	    inet_ntop(saddr.ss_family, (const void*)&(tmp_saddr->sin_addr), ipAddrStr, 1024);
	  }
	  else if (saddr.ss_family == AF_INET6) {
	    struct sockaddr_in6 *tmp_saddr = (struct sockaddr_in6*)&saddr;
	    inet_ntop(saddr.ss_family, (const void*)&(tmp_saddr->sin6_addr), ipAddrStr, 1024);
	  }

	  m_sockIP = ipAddrStr;
	}
      else
	std::cout << "getsockname error!" << std::endl;
    }
  
  return m_sockIP;
}
